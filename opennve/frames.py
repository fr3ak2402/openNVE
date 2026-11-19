"""Frame engine: converts the events of a time window into frames (tensors) and tiles.

JSON example (camera configuration):

    "frame": {
        "width": 1280,
        "height": 720,
        "mode": "time_surface",
        "tiles": {"size": 80, "min_events": 1}
    }

Modes (all frames have the shape height x width x 2, float32, channel = polarity 0 / 1):

    time_surface: linear time surface, (t_last - start_t) / (end_t - start_t) of the latest
                  event per pixel and polarity, 0 where there was no event
    binary:       1.0 where at least one event occurred, else 0.0
    count:        number of events per pixel and polarity

Tiles split the frame into a grid of size x size pixels (row by row, the tiles at the right and
bottom edges are smaller if the frame size is not a multiple of the tile size). Tile ids, rows
and columns start at 1. Only tiles with at least min_events events are returned
(min_events = 0 returns all tiles, including empty ones).
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

FRAME_MODES = ("time_surface", "binary", "count")


# ------------------------------------------------------------------ configuration


@dataclasses.dataclass
class TileConfig:
    """Tile grid: square tiles of size x size pixels."""

    size: int = 80
    min_events: int = 1

    def validate(self):
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("frame.tiles.size must be a positive integer, for instance 80")
        if isinstance(self.min_events, bool) or not isinstance(self.min_events, int) or self.min_events < 0:
            raise ValueError("frame.tiles.min_events must be an integer >= 0")


@dataclasses.dataclass
class FrameConfig:
    """Frame settings: size (None = sensor size), mode and optional tiles."""

    width: typing.Optional[int] = None
    height: typing.Optional[int] = None
    mode: str = "time_surface"
    tiles: typing.Optional[TileConfig] = None

    def validate(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 65535):
                raise ValueError(f"frame.{name} must be a positive integer or null (sensor size)")
        if self.mode not in FRAME_MODES:
            raise ValueError(f"frame.mode must be one of {', '.join(FRAME_MODES)}")
        if self.tiles is not None:
            self.tiles.validate()

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "tiles": None if self.tiles is None else dataclasses.asdict(self.tiles),
        }

    @classmethod
    def from_dict(cls, data: typing.Optional[dict[str, typing.Any]]) -> "FrameConfig":
        data = dict(data or {})
        unknown = set(data) - {"width", "height", "mode", "tiles"}
        if unknown:
            raise ValueError(f"unknown frame setting(s): {', '.join(sorted(unknown))}")
        tiles = data.get("tiles")
        if tiles is not None:
            if isinstance(tiles, int) and not isinstance(tiles, bool):
                tiles = {"size": tiles}
            if not isinstance(tiles, dict):
                raise ValueError("frame.tiles must be an object, for instance {\"size\": 80}, or null")
            unknown = set(tiles) - {"size", "min_events"}
            if unknown:
                raise ValueError(f"unknown frame.tiles setting(s): {', '.join(sorted(unknown))}")
            tiles = TileConfig(**tiles)
        result = cls(
            width=data.get("width"),
            height=data.get("height"),
            mode=str(data.get("mode", "time_surface")).lower(),
            tiles=tiles,
        )
        result.validate()
        return result


# ------------------------------------------------------------------ results


@dataclasses.dataclass
class Tile:
    """One tile of a frame.

    Attributes:
        id: Tile id, row by row, starting at 1 (id = (row - 1) * columns + col).
        row, col: Tile row and column, starting at 1.
        x, y: Position of the top-left pixel of the tile in the frame.
        width, height: Tile size (smaller at the right and bottom edges).
        event_count: Number of events in the tile.
        data: Frame data of the tile, shape (height, width, 2), float32.
    """

    id: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    event_count: int
    data: np.ndarray

    def to_dict(self) -> dict[str, typing.Any]:
        """Same format as the original FrameEngine: tile_id, row, col, tile_data."""
        return {"tile_id": self.id, "row": self.row, "col": self.col, "tile_data": self.data}


@dataclasses.dataclass
class Frame:
    """A frame built from the events of one time window.

    Attributes:
        camera: Camera name.
        start_t, end_t: Time window in µs (camera time).
        mode: "time_surface", "binary" or "count".
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


# ------------------------------------------------------------------ engine


def _columns(events) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """x, y, p, t as numpy arrays from a structured array (fields x, y, t and p or on),
    a pandas DataFrame or a dict of arrays."""
    names = getattr(getattr(events, "dtype", None), "names", None)
    if names is None and hasattr(events, "columns"):
        names = tuple(events.columns)  # pandas DataFrame
    elif names is None and isinstance(events, dict):
        names = tuple(events)
    if names is None:
        raise TypeError("events must be a numpy structured array, a pandas DataFrame or a dict of arrays")
    missing = {"x", "y", "t"} - set(names)
    if missing or not ({"p", "on"} & set(names)):
        raise ValueError(f"events need the fields x, y, t and p (or on), got {', '.join(names)}")

    def column(name, dtype):
        value = events[name]
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        return np.asarray(value).astype(dtype, copy=False)

    polarity = column("p" if "p" in names else "on", np.int64)
    return column("x", np.int64), column("y", np.int64), polarity, column("t", np.float64)


class FrameEngine:
    """Builds frames and tiles from events.

    Args:
        width, height: Frame size in pixels. Events outside the frame are ignored.
        tile_size: Tile size in pixels, or None for no tiles.
        mode: "time_surface", "binary" or "count".
        min_events: Minimum number of events of a returned tile (0 = all tiles).

    The original arguments frame_width, frame_height and frame_tile_size are accepted as well,
    so FrameEngine(frame_width=1280, frame_height=720, frame_tile_size=80) works as before.
    """

    def __init__(
        self,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        tile_size: typing.Optional[int] = None,
        mode: str = "time_surface",
        min_events: int = 1,
        *,
        frame_width: typing.Optional[int] = None,
        frame_height: typing.Optional[int] = None,
        frame_tile_size: typing.Optional[int] = None,
    ):
        width = frame_width if width is None else width
        height = frame_height if height is None else height
        tile_size = frame_tile_size if tile_size is None else tile_size
        if width is None or height is None:
            raise ValueError("width and height are required")
        config = FrameConfig(
            width=int(width),
            height=int(height),
            mode=mode,
            tiles=None if tile_size is None else TileConfig(size=int(tile_size), min_events=int(min_events)),
        )
        config.validate()
        self.config = config
        self.width = config.width
        self.height = config.height
        self.mode = config.mode
        self.tile_size = None if config.tiles is None else config.tiles.size
        self.min_events = 1 if config.tiles is None else config.tiles.min_events
        size = self.tile_size or 1
        self.tile_cols = (self.width + size - 1) // size if self.tile_size else 0
        self.tile_rows = (self.height + size - 1) // size if self.tile_size else 0

    @classmethod
    def from_config(cls, config: FrameConfig, sensor_width: int, sensor_height: int) -> "FrameEngine":
        """Engine for a FrameConfig; width/height None means the sensor size."""
        return cls(
            config.width or sensor_width,
            config.height or sensor_height,
            tile_size=None if config.tiles is None else config.tiles.size,
            mode=config.mode,
            min_events=1 if config.tiles is None else config.tiles.min_events,
        )

    # Frames

    def build(
        self,
        events,
        start_t: int,
        end_t: int,
        camera: str = "",
        window: typing.Any = None,
        mode: typing.Optional[str] = None,
    ) -> Frame:
        """Builds the frame and the tiles of the events in [start_t, end_t)."""
        mode = mode or self.mode
        if mode not in FRAME_MODES:
            raise ValueError(f"mode must be one of {', '.join(FRAME_MODES)}")
        if end_t <= start_t:
            raise ValueError("end_t must be greater than start_t")
        x, y, p, t = _columns(events)
        valid = (
            (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            & (p >= 0) & (p <= 1) & (t >= start_t) & (t < end_t)
        )
        if not np.all(valid):
            x, y, p, t = x[valid], y[valid], p[valid], t[valid]
        data = self._frame(x, y, p, t, float(start_t), float(end_t - start_t), mode)
        tiles = self._tiles(data, x, y)
        return Frame(
            camera=camera,
            start_t=int(start_t),
            end_t=int(end_t),
            mode=mode,
            data=data,
            event_count=len(x),
            tiles=tiles,
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            window=window,
        )

    def build_window(self, window, mode: typing.Optional[str] = None) -> Frame:
        """Builds the frame of an EventWindow."""
        return self.build(
            window.polarity_events, window.start_t, window.end_t, camera=window.camera, window=window, mode=mode
        )

    def _frame(self, x, y, p, t, start_t: float, duration: float, mode: str) -> np.ndarray:
        size = self.height * self.width * 2
        index = (y * self.width + x) * 2 + p
        if mode == "time_surface":
            last = np.full(size, -1.0, dtype=np.float64)
            np.maximum.at(last, index, t - start_t)  # latest event, also for unsorted events
            data = np.where(last >= 0, last / duration, 0.0).astype(np.float32)
        elif mode == "binary":
            data = np.zeros(size, dtype=np.float32)
            data[index] = 1.0
        else:
            data = np.bincount(index, minlength=size).astype(np.float32)
        return data.reshape(self.height, self.width, 2)

    def _tiles(self, data: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[Tile]:
        if not self.tile_size:
            return []
        size = self.tile_size
        counts = np.bincount(
            (y // size) * self.tile_cols + (x // size), minlength=self.tile_rows * self.tile_cols
        )
        tiles = []
        for index in np.flatnonzero(counts >= self.min_events):
            row, col = divmod(int(index), self.tile_cols)
            y_start, x_start = row * size, col * size
            y_end, x_end = min(y_start + size, self.height), min(x_start + size, self.width)
            tiles.append(
                Tile(
                    id=int(index) + 1,
                    row=row + 1,
                    col=col + 1,
                    x=x_start,
                    y=y_start,
                    width=x_end - x_start,
                    height=y_end - y_start,
                    event_count=int(counts[index]),
                    data=data[y_start:y_end, x_start:x_end, :].copy(),
                )
            )
        return tiles

    # Interface of the original FrameEngine (frame_data: pandas DataFrame or array with x, y, p, t)

    @staticmethod
    def create_linear_time_surface(frame_data, frame_width: int, frame_height: int, frame_start: int, frame_end: int):
        """Linear time surface, shape (height, width, 2), float32 (same as the original FrameEngine)."""
        engine = FrameEngine(frame_width, frame_height)
        if len(_columns(frame_data)[0]) == 0:
            if frame_end <= frame_start:
                raise ValueError("frame_end must be greater than frame_start")
            return np.zeros((engine.height, engine.width, 2), dtype=np.float32)
        return engine.build(frame_data, frame_start, frame_end).data

    def generate_frame(
        self,
        frame_data,
        threshold=0,
        select_insects_points=True,
        use_time_surface=True,
        tau=100_000,
        frame_end=None,
        activity_threshold=0.04,
        frame_start=None,
    ) -> list[dict[str, typing.Any]]:
        """Tiles as [{"tile_id", "row", "col", "tile_data"}] (same as the original FrameEngine).

        tau and activity_threshold are accepted for compatibility and ignored. Empty tiles are
        skipped; with select_insects_points, tiles with fewer than threshold events are skipped too.
        Without tiles configured, the whole frame is returned as one tile.
        """
        del tau, activity_threshold
        _, _, _, t = _columns(frame_data)
        if len(t) == 0:
            return []
        if frame_end is None:
            frame_end = int(t.max()) + 1
        if frame_start is None:
            frame_start = int(t.min())
        if frame_end <= frame_start:
            raise ValueError("invalid time window (frame_end <= frame_start)")
        minimum = max(1, int(threshold)) if select_insects_points else 1
        engine = FrameEngine(
            self.width,
            self.height,
            tile_size=self.tile_size or max(self.width, self.height),
            mode="time_surface" if use_time_surface else "binary",
            min_events=minimum,
        )
        frame = engine.build(frame_data, frame_start, frame_end)
        return frame.tile_dicts()
