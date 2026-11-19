"""Frame engine: converts the events of a time window into frames (tensors) and tiles.

Frame configurations are separate from the camera configurations. A frame configuration file
contains one or more named frame configurations (like the cameras in a camera configuration file):

    {
        "frames": [
            {
                "name": "full",
                "width": 1280,
                "height": 720,
                "mode": "time_surface",
                "time_surface": {"decay": "linear", "tau_us": 100000},
                "tiles": null
            },
            {
                "name": "tiles80",
                "mode": "time_surface",
                "time_surface": {"decay": "exponential", "tau_us": 20000},
                "tiles": {"size": 80, "min_events": 1}
            }
        ]
    }

Load it with load_frame_configs("frames.json") (list) or load_frame_config("frames.json", "tiles80"),
and pass a configuration to camera.frames(config), raw.frames(config) or window.frame(config).
Several configurations at once: camera.frame_sets(configs) or window.frames(configs).
width / height null = sensor size of the camera or file.

Modes (all frames have the shape height x width x 2, float32, channel = polarity 0 / 1):

    time_surface: time surface of the latest event per pixel and polarity, 0 where there was no event
                  decay "linear":      (t_last - start_t) / (end_t - start_t)
                  decay "exponential": exp(-(end_t - t_last) / tau_us)
    binary:       1.0 where at least one event occurred, else 0.0
    count:        number of events per pixel and polarity

Tiles split the frame into a grid of size x size pixels (row by row, the tiles at the right and
bottom edges are smaller if the frame size is not a multiple of the tile size). Tile ids, rows
and columns start at 1. Only tiles with at least min_events events are returned
(min_events = 0 returns all tiles, including empty ones).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

import numpy as np

PathLike = typing.Union[str, pathlib.Path]

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


TIME_SURFACE_DECAYS = ("linear", "exponential")


@dataclasses.dataclass
class TimeSurfaceConfig:
    """Time surface settings (mode "time_surface").

    decay "linear":      (t_last - start_t) / (end_t - start_t), 0 for an event at start_t
    decay "exponential": exp(-(end_t - t_last) / tau_us), close to 1 for recent events
    tau_us is only used by the exponential decay.
    """

    decay: str = "linear"
    tau_us: float = 100_000

    def validate(self):
        if self.decay not in TIME_SURFACE_DECAYS:
            raise ValueError(f"frame.time_surface.decay must be one of {', '.join(TIME_SURFACE_DECAYS)}")
        if isinstance(self.tau_us, bool) or not isinstance(self.tau_us, (int, float)) or not self.tau_us > 0:
            raise ValueError("frame.time_surface.tau_us must be a positive number (microseconds)")


@dataclasses.dataclass
class FrameConfig:
    """One frame configuration: name, size (None = sensor size), mode, time surface and optional tiles."""

    width: typing.Optional[int] = None
    height: typing.Optional[int] = None
    mode: str = "time_surface"
    tiles: typing.Optional[TileConfig] = None
    time_surface: TimeSurfaceConfig = dataclasses.field(default_factory=TimeSurfaceConfig)
    name: typing.Optional[str] = None

    def validate(self):
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("frame.name must be a non-empty string or null")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 65535):
                raise ValueError(f"frame.{name} must be a positive integer or null (sensor size)")
        if self.mode not in FRAME_MODES:
            raise ValueError(f"frame.mode must be one of {', '.join(FRAME_MODES)}")
        if self.tiles is not None:
            self.tiles.validate()
        self.time_surface.validate()

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "time_surface": dataclasses.asdict(self.time_surface),
            "tiles": None if self.tiles is None else dataclasses.asdict(self.tiles),
        }

    @classmethod
    def from_dict(cls, data: typing.Optional[dict[str, typing.Any]]) -> "FrameConfig":
        data = dict(data or {})
        unknown = set(data) - {"name", "width", "height", "mode", "tiles", "time_surface"}
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
        time_surface = data.get("time_surface")
        if time_surface is None:
            time_surface = {}
        elif isinstance(time_surface, str):
            time_surface = {"decay": time_surface}  # "time_surface": "exponential" is accepted as well
        if not isinstance(time_surface, dict):
            raise ValueError("frame.time_surface must be an object, for instance {\"decay\": \"exponential\"}")
        unknown = set(time_surface) - {"decay", "tau_us"}
        if unknown:
            raise ValueError(f"unknown frame.time_surface setting(s): {', '.join(sorted(unknown))}")
        if "decay" in time_surface:
            time_surface = {**time_surface, "decay": str(time_surface["decay"]).lower()}
        result = cls(
            width=data.get("width"),
            height=data.get("height"),
            mode=str(data.get("mode", "time_surface")).lower(),
            tiles=tiles,
            time_surface=TimeSurfaceConfig(**time_surface),
            name=data.get("name"),
        )
        result.validate()
        return result

    def to_json(self, indent: typing.Optional[int] = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "FrameConfig":
        return cls.from_dict(json.loads(text))

    def copy(self, **changes) -> "FrameConfig":
        """Copy with changes, for instance config.copy(name="small", width=640, height=360)."""
        return FrameConfig.from_dict({**self.to_dict(), **changes})


def as_frame_config(config: typing.Union[FrameConfig, dict, str, None]) -> FrameConfig:
    """FrameConfig from a FrameConfig, a dict, a JSON string or None (default configuration)."""
    if config is None:
        return FrameConfig()
    if isinstance(config, FrameConfig):
        config.validate()
        return config
    if isinstance(config, dict):
        return FrameConfig.from_dict(config)
    if isinstance(config, str):
        return FrameConfig.from_json(config)
    raise TypeError(
        f"expected a FrameConfig, a dict or a JSON string, got {type(config).__name__} "
        "(the window duration is passed as duration_ms=...)"
    )


def as_frame_configs(configs) -> list[FrameConfig]:
    """List of FrameConfig with unique names from a list (or a dict name -> configuration).
    Configurations without a name are called frame1, frame2, ..."""
    if isinstance(configs, (FrameConfig, str)) or (isinstance(configs, dict) and not _is_name_mapping(configs)):
        configs = [configs]
    elif isinstance(configs, dict):
        configs = [
            dataclasses.replace(as_frame_config(config), name=name) for name, config in configs.items()
        ]
    result = []
    for index, config in enumerate(configs):
        config = as_frame_config(config)
        if config.name is None:
            config = dataclasses.replace(config, name=f"frame{index + 1}")
        result.append(config)
    names = [config.name for config in result]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate frame configuration name(s): {', '.join(duplicates)}")
    if not result:
        raise ValueError("no frame configuration")
    return result


def _is_name_mapping(data: dict) -> bool:
    return bool(data) and all(isinstance(value, (dict, FrameConfig)) for value in data.values()) and not (
        set(data) & {"name", "width", "height", "mode", "tiles", "time_surface"}
    )


# ------------------------------------------------------------------ frame configuration files


def frame_configs_from_data(data: typing.Any, source: str = "frame configuration") -> list[FrameConfig]:
    """Frame configurations from parsed JSON: one object, a list or {"frames": [...]}."""
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected a frame configuration object, a list or {{\"frames\": [...]}}")
    configs = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{source}: frame configuration {index} must be an object")
        try:
            configs.append(FrameConfig.from_dict(item))
        except ValueError as error:
            raise ValueError(f"{source}: frame configuration {index}: {error}") from error
    try:
        return as_frame_configs(configs)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def load_frame_configs(path: PathLike) -> list[FrameConfig]:
    """Loads a frame configuration file with one configuration, a list or {"frames": [...]}.
    Configurations without a name are called frame1, frame2, ..."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return frame_configs_from_data(data, str(path))


def load_frame_config(path: PathLike, name: typing.Optional[str] = None) -> FrameConfig:
    """Loads one frame configuration from a file (by name if the file has several)."""
    configs = load_frame_configs(path)
    if name is None:
        if len(configs) != 1:
            raise ValueError(f"{path} contains {len(configs)} frame configurations, specify a name")
        return configs[0]
    for config in configs:
        if config.name == name:
            return config
    names = ", ".join(config.name for config in configs)
    raise ValueError(f"{path}: no frame configuration named {name!r} (available: {names})")


def save_frame_configs(path: PathLike, configs: typing.Union[FrameConfig, typing.Sequence[FrameConfig]]):
    """Saves one frame configuration (as an object) or several (as {"frames": [...]})."""
    if isinstance(configs, FrameConfig):
        configs.validate()
        data: typing.Any = configs.to_dict()
    else:
        data = {"frames": [config.to_dict() for config in as_frame_configs(list(configs))]}
    pathlib.Path(path).write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ results


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
        decay: Time surface decay, "linear" or "exponential".
        tau_us: Time constant of the exponential decay in µs.

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
        decay: str = "linear",
        tau_us: float = 100_000,
        *,
        frame_width: typing.Optional[int] = None,
        frame_height: typing.Optional[int] = None,
        frame_tile_size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
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
            time_surface=TimeSurfaceConfig(decay=decay, tau_us=tau_us),
            name=name,
        )
        config.validate()
        self.config = config
        self.name = name
        self.width = config.width
        self.height = config.height
        self.mode = config.mode
        self.decay = config.time_surface.decay
        self.tau_us = float(config.time_surface.tau_us)
        self.tile_size = None if config.tiles is None else config.tiles.size
        self.min_events = 1 if config.tiles is None else config.tiles.min_events
        size = self.tile_size or 1
        self.tile_cols = (self.width + size - 1) // size if self.tile_size else 0
        self.tile_rows = (self.height + size - 1) // size if self.tile_size else 0

    @classmethod
    def from_config(
        cls, config: typing.Union[FrameConfig, dict, str, None], sensor_width: int, sensor_height: int
    ) -> "FrameEngine":
        """Engine for a FrameConfig (or dict / JSON); width/height None means the sensor size."""
        config = as_frame_config(config)
        return cls(
            config.width or sensor_width,
            config.height or sensor_height,
            tile_size=None if config.tiles is None else config.tiles.size,
            mode=config.mode,
            min_events=1 if config.tiles is None else config.tiles.min_events,
            decay=config.time_surface.decay,
            tau_us=config.time_surface.tau_us,
            name=config.name,
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
        decay: typing.Optional[str] = None,
    ) -> Frame:
        """Builds the frame and the tiles of the events in [start_t, end_t).
        mode and decay override the engine settings for this frame."""
        mode = mode or self.mode
        decay = decay or self.decay
        if mode not in FRAME_MODES:
            raise ValueError(f"mode must be one of {', '.join(FRAME_MODES)}")
        if decay not in TIME_SURFACE_DECAYS:
            raise ValueError(f"decay must be one of {', '.join(TIME_SURFACE_DECAYS)}")
        if end_t <= start_t:
            raise ValueError("end_t must be greater than start_t")
        x, y, p, t = _columns(events)
        valid = (
            (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            & (p >= 0) & (p <= 1) & (t >= start_t) & (t < end_t)
        )
        if not np.all(valid):
            x, y, p, t = x[valid], y[valid], p[valid], t[valid]
        data = self._frame(x, y, p, t, float(start_t), float(end_t - start_t), mode, decay)
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
            decay=decay if mode == "time_surface" else None,
            event_x=x.astype(np.uint16),
            event_y=y.astype(np.uint16),
            _tile_size=self.tile_size,
            _min_events=self.min_events,
            name=self.name,
        )

    def build_window(self, window, mode: typing.Optional[str] = None, decay: typing.Optional[str] = None) -> Frame:
        """Builds the frame of an EventWindow."""
        return self.build(
            window.polarity_events,
            window.start_t,
            window.end_t,
            camera=window.camera,
            window=window,
            mode=mode,
            decay=decay,
        )

    def _frame(self, x, y, p, t, start_t: float, duration: float, mode: str, decay: str = "linear") -> np.ndarray:
        size = self.height * self.width * 2
        index = (y * self.width + x) * 2 + p
        if mode == "time_surface":
            last = np.full(size, -1.0, dtype=np.float64)
            np.maximum.at(last, index, t - start_t)  # latest event, also for unsorted events
            present = last >= 0
            if decay == "linear":
                values = last / duration
            else:
                # exp(-(end_t - t_last) / tau): 1 for an event at the end of the window
                values = np.exp((last - duration) / self.tau_us)
            data = np.where(present, values, 0.0).astype(np.float32)
        elif mode == "binary":
            data = np.zeros(size, dtype=np.float32)
            data[index] = 1.0
        else:
            data = np.bincount(index, minlength=size).astype(np.float32)
        return data.reshape(self.height, self.width, 2)

    def _tiles(self, data: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[Tile]:
        if not self.tile_size:
            return []
        counts = np.bincount(
            (y // self.tile_size) * self.tile_cols + (x // self.tile_size), minlength=self.tile_rows * self.tile_cols
        )
        return split_into_tiles(
            data, self.tile_size, counts.reshape(self.tile_rows, self.tile_cols), self.min_events
        )

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
