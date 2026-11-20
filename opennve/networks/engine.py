"""NeuralEngine: a named collection of networks (build, load, save, convert, transfer weights)."""

from __future__ import annotations

import pathlib
import typing

from .network import Network, PathLike, WeightComparison, compare_weights
from .registry import NetworkParameters

QUANTIZATION_SUFFIXES = {
    "none": "",
    "dynamic": "_dynamic",
    "float32": "_float32",
    "float16": "_float16",
    "float8_e4m3fn": "_float8_e4m3fn",
    "float8_e5m2": "_float8_e5m2",
    "int8": "_int8",
}


class NeuralEngine:
    """Manages networks by name (or index, in the order they were added).

        engine = opennve.NeuralEngine()
        engine.add_network("unet", "UNet", parameters=opennve.NetworkParameters(number_of_classes=3))
        engine.add_network("hsnr", path="models/hsnr_unet.keras")
        engine.transfer_weights("unet", "hsnr")
        engine.quantize("unet", "int8", representative_dataset=...)
    """

    def __init__(self, model_directory: PathLike = "."):
        self.model_directory = pathlib.Path(model_directory)
        self._networks: dict[str, Network] = {}

    # collection

    def add_network(
        self,
        name: str,
        architecture: typing.Optional[str] = None,
        parameters: typing.Union[NetworkParameters, dict, None] = None,
        path: typing.Optional[PathLike] = None,
        safe_mode: bool = True,
        **options,
    ) -> Network:
        """Builds a network (architecture and parameters) or loads it from path. Returns the Network."""
        if name in self._networks:
            raise ValueError(f"a network named {name!r} already exists")
        if parameters is not None:
            if architecture is None:
                raise ValueError("building a network needs the architecture, for instance \"UNet\"")
            network = Network.build(architecture, parameters, name=name, **options)
        elif path is not None:
            network = Network.load(path, name=name, safe_mode=safe_mode)
            network.architecture = architecture
        else:
            raise ValueError("give parameters (to build a network) or path (to load one)")
        self._networks[name] = network
        return network

    def add(self, network: Network, name: typing.Optional[str] = None) -> Network:
        """Adds an existing Network (or Keras model)."""
        if not isinstance(network, Network):
            network = Network(network, name=name)
        name = name or network.name
        if name in self._networks:
            raise ValueError(f"a network named {name!r} already exists")
        network.name = name
        self._networks[name] = network
        return network

    def remove(self, key: typing.Union[str, int]):
        del self._networks[self.get(key).name]

    def get(self, key: typing.Union[str, int]) -> Network:
        """Network by name or index."""
        if isinstance(key, int):
            return list(self._networks.values())[key]
        if key not in self._networks:
            raise KeyError(f"no network named {key!r} (available: {', '.join(self._networks) or 'none'})")
        return self._networks[key]

    def __getitem__(self, key) -> Network:
        return self.get(key)

    def __contains__(self, name) -> bool:
        return name in self._networks

    def __iter__(self):
        return iter(self._networks.values())

    def __len__(self) -> int:
        return len(self._networks)

    @property
    def names(self) -> list[str]:
        return list(self._networks)

    @property
    def networks(self) -> list[Network]:
        return list(self._networks.values())

    # operations

    def save(self, key, path: PathLike):
        self.get(key).save(path)

    def load_weights(self, key, path: PathLike):
        self.get(key).load_weights(path)

    def save_weights(self, key, path: PathLike):
        self.get(key).save_weights(path)

    def summary(self, key, expand_nested: bool = True):
        self.get(key).summary(expand_nested=expand_nested)

    def quantize(
        self,
        key,
        quantization: str,
        path: typing.Optional[PathLike] = None,
        representative_dataset=None,
    ) -> pathlib.Path:
        """Converts a network to TensorFlow Lite. Default path: <model_directory>/<name><suffix>.tflite,
        for instance unet_int8.tflite."""
        network = self.get(key)
        if path is None:
            if quantization not in QUANTIZATION_SUFFIXES:
                raise ValueError(f"quantization must be one of {', '.join(QUANTIZATION_SUFFIXES)}")
            path = self.model_directory / f"{network.name}{QUANTIZATION_SUFFIXES[quantization]}.tflite"
        return network.to_tflite(path, quantization, representative_dataset)

    def transfer_weights(self, destination, source, method: str = "hsnr_unet", **options) -> WeightComparison:
        """Copies the weights of source into destination (see Network.transfer_weights_from)."""
        return self.get(destination).transfer_weights_from(self.get(source), method, **options)

    def compare_weights(self, a, b) -> WeightComparison:
        return compare_weights(self.get(a), self.get(b))
