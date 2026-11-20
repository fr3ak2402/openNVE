"""Registry of network architectures and Keras custom objects.

New architectures are added with a builder function:

    import opennve

    @opennve.register_network("MyNet")
    def build_my_net(parameters: opennve.NetworkParameters, **options):
        import keras
        inputs = keras.Input((parameters.tile_height, parameters.tile_width, parameters.channels))
        ...
        return keras.Model(inputs, outputs)

    network = opennve.Network.build("MyNet", parameters)

Custom Keras layers that saved models need for loading are registered with
register_custom_objects({"MyLayer": MyLayer}); they are passed to keras.models.load_model
automatically (by the agents and by Network.load).
"""

from __future__ import annotations

import dataclasses
import importlib
import typing


@dataclasses.dataclass
class NetworkParameters:
    """Input and output size of a network.

    number_of_classes: output channels (classes) of the network.
    tile_height, tile_width: input size (for instance 80 x 80 tiles).
    channels: input channels (2: one per polarity, see the frame engine).
    """

    number_of_classes: int
    tile_height: int = 80
    tile_width: int = 80
    channels: int = 2

    def __post_init__(self):
        for name in ("number_of_classes", "tile_height", "tile_width", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"network parameter {name} must be a positive integer")

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.tile_height, self.tile_width, self.channels)

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> "NetworkParameters":
        """Also accepts the upper-case names (NUMBER_OF_CLASSES, TILE_HEIGHT, TILE_WIDTH, CHANNELS)."""
        values = {key.lower(): value for key, value in data.items()}
        unknown = set(values) - {"number_of_classes", "tile_height", "tile_width", "channels"}
        if unknown:
            raise ValueError(f"unknown network parameter(s): {', '.join(sorted(unknown))}")
        return cls(**values)


Builder = typing.Callable[..., typing.Any]

_BUILDERS: dict[str, Builder] = {}
# built-in architectures, imported on first use (they need TensorFlow)
_LAZY_BUILDERS: dict[str, tuple[str, str]] = {
    "UNet": ("opennve.networks.unet", "build_unet"),
}
_CUSTOM_OBJECTS: dict[str, typing.Any] = {}
_LAZY_CUSTOM_OBJECTS = ("opennve.networks.unet",)


def register_network(name: str, builder: typing.Optional[Builder] = None):
    """Registers an architecture: builder(parameters, **options) -> keras.Model.
    Usable as a decorator (@register_network("MyNet")) or directly."""
    if not isinstance(name, str) or not name:
        raise ValueError("the network name must be a non-empty string")

    def decorator(function: Builder) -> Builder:
        _BUILDERS[name] = function
        _LAZY_BUILDERS.pop(name, None)
        return function

    return decorator if builder is None else decorator(builder)


def unregister_network(name: str):
    _BUILDERS.pop(name, None)
    _LAZY_BUILDERS.pop(name, None)


def list_networks() -> list[str]:
    """Names of the registered architectures."""
    return sorted(set(_BUILDERS) | set(_LAZY_BUILDERS))


def get_builder(name: str) -> Builder:
    if name in _BUILDERS:
        return _BUILDERS[name]
    if name in _LAZY_BUILDERS:
        module, attribute = _LAZY_BUILDERS[name]
        builder = getattr(importlib.import_module(module), attribute)
        _BUILDERS[name] = builder
        return builder
    raise ValueError(f"unknown network {name!r} (registered: {', '.join(list_networks())})")


def build_network(name: str, parameters: typing.Union[NetworkParameters, dict], **options):
    """Builds a Keras model of a registered architecture."""
    if isinstance(parameters, dict):
        parameters = NetworkParameters.from_dict(parameters)
    return get_builder(name)(parameters, **options)


def register_custom_objects(objects: dict[str, typing.Any], package: typing.Optional[str] = None):
    """Registers Keras classes / functions needed to load saved models.

    Each object is registered under its name and, if given, under "package>name" (the name
    Keras stores for classes registered with register_keras_serializable(package=...)).
    """
    for name, value in objects.items():
        _CUSTOM_OBJECTS[name] = value
        if package:
            _CUSTOM_OBJECTS[f"{package}>{name}"] = value


def custom_objects() -> dict[str, typing.Any]:
    """All registered custom objects (including the built-in UNet layers), for keras.models.load_model."""
    for module in _LAZY_CUSTOM_OBJECTS:
        try:
            importlib.import_module(module)
        except ImportError:
            pass
    return dict(_CUSTOM_OBJECTS)
