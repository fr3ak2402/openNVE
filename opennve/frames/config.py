"""Frame configurations: FrameConfig, TileConfig, TimeSurfaceConfig and frame configuration files."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing


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
