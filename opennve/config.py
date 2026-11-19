"""JSON camera configuration.

A camera configuration describes one camera:

    {
        "name": "left",
        "model": "SE1-S4",
        "serial": "05110634",
        "settings": {
            "biases": {"diff_on": 20, "diff_off": 20},
            "clock": "internal",
            "rate_limiter": null,
            "enable_external_trigger": true
        },
        "filter": {
            "polarity": "both",
            "background_activity": {"enabled": true, "time_us": 5000, "min_neighbors": 1}
        },
        "output": {"fields": ["x", "y", "p", "t"]}
    }

- name: free name of the camera in your setup
- model: camera model (see opennve.models.MODELS), for instance "SE1-S4", "EVK4", "EVK3-HD"
- serial: serial number, or null for the first camera of this model
- settings: hardware settings, passed to the camera (fields of the model's
  neuromorphic_drivers Configuration; omitted fields keep their default value)
- filter: software filters applied to the events (see opennve.filters)
- output: fields of the output events, in order (see opennve.output):
  x, y, p = polarity (1 ON / 0 OFF), t = timestamp in µs
  (the camera name is stored once per packet, not per event)

A configuration file contains either one camera object, a list of camera objects,
or {"cameras": [...]}.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import types
import typing

import numpy as np

from .filters import FilterConfig
from .output import OutputConfig
from .models import CameraModel, get_model

PathLike = typing.Union[str, pathlib.Path]

# Large bit masks are only written to JSON if they differ from their default value
_COMPACT_FIELDS = {"x_mask", "y_mask", "pixel_mask", "mask_intersection_only"}


# ------------------------------------------------------------------
# Conversion between JSON values and neuromorphic_drivers dataclasses
# ------------------------------------------------------------------


def _union_arguments(annotation) -> typing.Optional[tuple]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        return typing.get_args(annotation)
    return None


def _to_native(value: typing.Any, annotation: typing.Any, path: str) -> typing.Any:
    """Converts a JSON value to the type expected by a neuromorphic_drivers field."""
    arguments = _union_arguments(annotation)
    if arguments is not None:
        if value is None:
            if type(None) in arguments:
                return None
            raise ValueError(f"{path} must not be null")
        non_none = [argument for argument in arguments if argument is not type(None)]
        # serde integer types are Union[numpy.intX, int]
        numpy_types = [
            argument
            for argument in non_none
            if isinstance(argument, type) and issubclass(argument, np.integer)
        ]
        if numpy_types:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path} must be an integer")
            information = np.iinfo(numpy_types[0])
            if not information.min <= value <= information.max:
                raise ValueError(f"{path} must be in [{information.min}, {information.max}]")
            return value
        if len(non_none) == 1:
            return _to_native(value, non_none[0], path)
        raise ValueError(f"{path}: unsupported type {annotation}")
    if dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        return _dataclass_from_dict(annotation, value, path)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        key = value.strip().upper()
        if key not in annotation.__members__:
            choices = ", ".join(name.lower() for name in annotation.__members__)
            raise ValueError(f"{path} must be one of {choices}")
        return annotation[key]
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be true or false")
        return value
    if typing.get_origin(annotation) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a list")
        item_types = typing.get_args(annotation)
        if len(value) != len(item_types):
            raise ValueError(f"{path} must have {len(item_types)} items")
        return tuple(
            _to_native(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, item_types))
        )
    return value


def _dataclass_from_dict(cls: type, data: dict, path: str, base: typing.Any = None):
    hints = typing.get_type_hints(cls)
    names = {field.name for field in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(
            f"unknown setting(s) in {path}: {', '.join(sorted(unknown))} "
            f"(supported: {', '.join(sorted(names))})"
        )
    if base is None:
        try:
            base = cls()
        except TypeError:
            base = None
    values = {}
    for field in dataclasses.fields(cls):
        if field.name in data:
            value = data[field.name]
            if base is not None and dataclasses.is_dataclass(hints[field.name]) and isinstance(value, dict):
                # Partial nested objects (for instance only two biases) keep the other defaults
                nested_base = getattr(base, field.name)
                values[field.name] = _dataclass_from_dict(
                    hints[field.name], value, f"{path}.{field.name}", nested_base
                )
            else:
                values[field.name] = _to_native(value, hints[field.name], f"{path}.{field.name}")
        elif base is not None:
            values[field.name] = getattr(base, field.name)
        else:
            raise ValueError(f"{path}.{field.name} is required")
    return cls(**values)


def _to_json_value(value: typing.Any) -> typing.Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.name.lower()
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def settings_to_dict(settings: typing.Any) -> dict[str, typing.Any]:
    """neuromorphic_drivers Configuration -> JSON object (bit masks only if not default)."""
    default = type(settings)()
    result = {}
    for field in dataclasses.fields(settings):
        value = getattr(settings, field.name)
        if field.name in _COMPACT_FIELDS and value == getattr(default, field.name):
            continue
        result[field.name] = _to_json_value(value)
    return result


def settings_from_dict(model: CameraModel, data: typing.Optional[dict]) -> typing.Any:
    """JSON object -> neuromorphic_drivers Configuration of the model (missing fields keep defaults)."""
    if data is None:
        return model.configuration_type()
    if not isinstance(data, dict):
        raise ValueError("settings must be an object")
    return _dataclass_from_dict(model.configuration_type, data, "settings")


# ------------------------------------------------------------------
# Camera configuration
# ------------------------------------------------------------------


@dataclasses.dataclass
class CameraConfig:
    """Configuration of one camera (see module documentation for the JSON format)."""

    model: CameraModel
    name: typing.Optional[str] = None
    serial: typing.Optional[str] = None
    settings: typing.Any = None
    filter: FilterConfig = dataclasses.field(default_factory=FilterConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = get_model(self.model)
        if self.settings is None:
            self.settings = self.model.configuration_type()
        elif isinstance(self.settings, dict):
            self.settings = settings_from_dict(self.model, self.settings)
        elif not isinstance(self.settings, self.model.configuration_type):
            raise TypeError(
                f"settings must be a {self.model.module.__name__}.Configuration for {self.model.id}"
            )
        if isinstance(self.filter, dict):
            self.filter = FilterConfig.from_dict(self.filter)
        if isinstance(self.output, dict):
            self.output = OutputConfig.from_dict(self.output)
        if self.name is None:
            self.name = self.model.id if self.serial is None else f"{self.model.id}-{self.serial}"

    # JSON

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "model": self.model.id,
            "serial": self.serial,
            "settings": settings_to_dict(self.settings),
            "filter": self.filter.to_dict(),
            "output": self.output.to_dict(),
        }

    def to_json(self, indent: typing.Optional[int] = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> "CameraConfig":
        if not isinstance(data, dict):
            raise ValueError("a camera configuration must be an object")
        unknown = set(data) - {"name", "model", "serial", "settings", "filter", "output"}
        if unknown:
            raise ValueError(f"unknown camera setting(s): {', '.join(sorted(unknown))}")
        if "model" not in data:
            raise ValueError("model is required")
        serial = data.get("serial")
        if serial is not None and not isinstance(serial, str):
            raise ValueError("serial must be a string (for instance \"05110634\") or null")
        model = get_model(data["model"])
        return cls(
            model=model,
            name=data.get("name"),
            serial=serial,
            settings=settings_from_dict(model, data.get("settings")),
            filter=FilterConfig.from_dict(data.get("filter")),
            output=OutputConfig.from_dict(data.get("output")),
        )

    @classmethod
    def from_json(cls, text: str) -> "CameraConfig":
        return cls.from_dict(json.loads(text))

    def copy(self, **changes) -> "CameraConfig":
        """Copy with changes, for instance config.copy(serial="05110636")."""
        return CameraConfig.from_dict({**self.to_dict(), **{k: v for k, v in changes.items()}})


def load_configs(path: PathLike) -> list[CameraConfig]:
    """Loads a configuration file with one camera, a list of cameras or {"cameras": [...]}."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cameras" in data:
        data = data["cameras"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a camera object, a list or {{\"cameras\": [...]}}")
    configs = []
    for index, item in enumerate(data):
        try:
            configs.append(CameraConfig.from_dict(item))
        except ValueError as error:
            raise ValueError(f"{path}: camera {index}: {error}") from error
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate camera name(s): {', '.join(duplicates)}")
    return configs


def load_config(path: PathLike, name: typing.Optional[str] = None) -> CameraConfig:
    """Loads one camera from a configuration file (by name if the file has several cameras)."""
    configs = load_configs(path)
    if name is None:
        if len(configs) != 1:
            raise ValueError(f"{path} contains {len(configs)} cameras, specify a name")
        return configs[0]
    for config in configs:
        if config.name == name:
            return config
    raise ValueError(f"{path}: no camera named {name!r}")


def save_configs(path: PathLike, configs: typing.Union[CameraConfig, typing.Sequence[CameraConfig]]):
    """Saves one camera (as an object) or several cameras (as {"cameras": [...]})."""
    if isinstance(configs, CameraConfig):
        data: typing.Any = configs.to_dict()
    else:
        data = {"cameras": [config.to_dict() for config in configs]}
    pathlib.Path(path).write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
