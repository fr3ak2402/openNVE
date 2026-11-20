"""Frame and Tile (results of the frame engine) and splitting frames into tiles."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from .config import TileConfig



@dataclasses.dataclass
class Tile:
    """One tile of a frame.

    Attributes:
        id: Tile id, row by row, starting at 1 (id = (row - 1) * columns + col).
        row, col: Tile row and column, starting at 1.
        x, y: Position of the top-left pixel of the tile in the frame.
        width, height: Tile size (smaller at the right and bottom edges).
        event_count: Number of events in the tile (None if unknown, see split_into_tiles).
        data: Frame data of the tile, shape (height, width, ...) of the split array
            (height x width x 2, float32 for an unchanged frame).
    """

    id: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    event_count: typing.Optional[int]
    data: np.ndarray

    def to_dict(self) -> dict[str, typing.Any]:
        """Same format as the original FrameEngine: tile_id, row, col, tile_data."""
        return {"tile_id": self.id, "row": self.row, "col": self.col, "tile_data": self.data}


@dataclasses.dataclass
class Frame:
    """A frame built from the events of one time window.

    Attributes:
        camera: Camera name.
        name: Name of the frame configuration (None if it has none).
        start_t, end_t: Time window in µs (camera time).
        mode: "time_surface", "binary" or "count".
        decay: "linear" or "exponential" for time surfaces, else None.
        data: Frame, shape (height, width, 2), float32 (channel = polarity).
        event_count: Number of events used (inside the frame and the time window).
        tiles: Selected tiles (empty list if tiles are disabled).
        tile_rows, tile_cols: Size of the tile grid (0 if tiles are disabled).
        window: The time window (window.events: the events in the output format).
    """

    camera: str
    start_t: int
    end_t: int
    mode: str
    data: np.ndarray
    event_count: int
    tiles: list[Tile]
    tile_rows: int = 0
    tile_cols: int = 0
    window: typing.Any = dataclasses.field(default=None, repr=False)
    decay: typing.Optional[str] = None
    # positions of the events used for the frame (for split)
    event_x: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    event_y: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    name: typing.Optional[str] = None

    def tile_event_counts(self, tile_size: int) -> np.ndarray:
        """Number of events per tile, shape (tile rows, tile columns)."""
        if self.event_x is None or self.event_y is None:
            raise ValueError("this frame has no event positions")
        rows, cols = tile_grid(self.width, self.height, tile_size)
        counts = np.bincount(
            (self.event_y.astype(np.int64) // tile_size) * cols + self.event_x.astype(np.int64) // tile_size,
            minlength=rows * cols,
        )
        return counts.reshape(rows, cols)

    def split(
        self,
        tile_size: typing.Optional[int] = None,
        min_events: typing.Optional[int] = None,
        data: typing.Optional[np.ndarray] = None,
    ) -> list[Tile]:
        """Splits the frame into tiles, for instance after processing it.

        tile_size: tile size in pixels (default: the "tiles" configuration, else 80).
        min_events: only tiles with at least this many events (default: configuration, else 1;
            0 = all tiles). The events are those of the frame, also if data was changed.
        data: array to split instead of frame.data, for instance a processed frame. Its first
            two dimensions must be height x width (any further dimensions are kept).

        The tiles are also stored in frame.tiles (and tile_rows / tile_cols are updated).
        """
        if tile_size is None:
            tile_size = self._tile_size or 80
        if min_events is None:
            min_events = self._min_events if self._tile_size else 1
        TileConfig(size=tile_size, min_events=min_events).validate()
        data = self.data if data is None else np.asarray(data)
        if data.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"data must have the frame size {self.height} x {self.width}, got {data.shape[0]} x {data.shape[1]}"
            )
        self.tiles = split_into_tiles(data, tile_size, self.tile_event_counts(tile_size), min_events)
        self.tile_rows, self.tile_cols = tile_grid(self.width, self.height, tile_size)
        self._tile_size, self._min_events = tile_size, min_events
        return self.tiles

    _tile_size: typing.Optional[int] = dataclasses.field(default=None, repr=False)
    _min_events: int = dataclasses.field(default=1, repr=False)

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def events(self) -> np.ndarray:
        """Events of the time window in the output format (for instance x, y, p, t)."""
        if self.window is None:
            raise ValueError("this frame was not built from a time window")
        return self.window.events

    def tile(self, tile_id: int) -> typing.Optional[Tile]:
        """Selected tile with this id, or None."""
        for tile in self.tiles:
            if tile.id == tile_id:
                return tile
        return None

    def tile_dicts(self) -> list[dict[str, typing.Any]]:
        """Tiles in the format of the original FrameEngine.generate_frame:
        [{"tile_id": 1, "row": 1, "col": 1, "tile_data": ndarray}, ...]"""
        return [tile.to_dict() for tile in self.tiles]

    def to_image(self, draw_tiles: bool = True) -> np.ndarray:
        """BGR image (uint8) for display: polarity 1 in green, polarity 0 in red,
        brightness = frame value (newer events are brighter in time_surface mode)."""
        data = self.data
        if self.mode == "count":
            data = np.clip(data / 4.0, 0, 1)  # 4 or more events = full brightness
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :, 1] = np.clip(data[:, :, 1] * 255, 0, 255).astype(np.uint8)
        image[:, :, 2] = np.clip(data[:, :, 0] * 255, 0, 255).astype(np.uint8)
        if draw_tiles:
            for tile in self.tiles:
                image[tile.y, tile.x : tile.x + tile.width] = (255, 255, 255)
                image[tile.y + tile.height - 1, tile.x : tile.x + tile.width] = (255, 255, 255)
                image[tile.y : tile.y + tile.height, tile.x] = (255, 255, 255)
                image[tile.y : tile.y + tile.height, tile.x + tile.width - 1] = (255, 255, 255)
        return image


# ------------------------------------------------------------------ tiles


def tile_grid(width: int, height: int, tile_size: int) -> tuple[int, int]:
    """(rows, columns) of the tile grid; the tiles at the right and bottom edges may be smaller."""
    return (height + tile_size - 1) // tile_size, (width + tile_size - 1) // tile_size


def split_into_tiles(
    data: np.ndarray,
    tile_size: int,
    event_counts: typing.Optional[np.ndarray] = None,
    min_events: int = 0,
) -> list[Tile]:
    """Splits any array of shape (height, width, ...) into tiles of tile_size x tile_size pixels.

    event_counts: optional events per tile, shape (rows, columns), see Frame.tile_event_counts.
        With event_counts, only tiles with at least min_events events are returned.
        Without, all tiles are returned and Tile.event_count is None.
    Tiles are ordered row by row, ids / rows / columns start at 1.
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("data must have at least 2 dimensions (height, width, ...)")
    TileConfig(size=int(tile_size), min_events=int(min_events)).validate()
    height, width = data.shape[:2]
    rows, cols = tile_grid(width, height, tile_size)
    if event_counts is not None:
        event_counts = np.asarray(event_counts)
        if event_counts.shape != (rows, cols):
            raise ValueError(f"event_counts must have the shape ({rows}, {cols})")
        selected = np.flatnonzero(event_counts.reshape(-1) >= min_events)
    elif min_events > 0:
        raise ValueError("min_events needs event_counts (use Frame.split to count the events of a frame)")
    else:
        selected = np.arange(rows * cols)
    tiles = []
    for index in selected:
        row, col = divmod(int(index), cols)
        y_start, x_start = row * tile_size, col * tile_size
        y_end, x_end = min(y_start + tile_size, height), min(x_start + tile_size, width)
        tiles.append(
            Tile(
                id=int(index) + 1,
                row=row + 1,
                col=col + 1,
                x=x_start,
                y=y_start,
                width=x_end - x_start,
                height=y_end - y_start,
                event_count=None if event_counts is None else int(event_counts.reshape(-1)[index]),
                data=data[y_start:y_end, x_start:x_end].copy(),
            )
        )
    return tiles
