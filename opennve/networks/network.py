"""Network: a Keras model with building, loading, saving, prediction and TensorFlow Lite export."""

from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import typing

import numpy as np

from .registry import NetworkParameters, build_network, custom_objects

PathLike = typing.Union[str, pathlib.Path]

# quantization -> description (see Network.to_tflite)
QUANTIZATIONS = {
    "none": "float32 model, no optimization",
    "dynamic": "dynamic range quantization: int8 weights, float32 inputs and outputs",
    "float32": "float32 types with the default optimization (as dynamic range quantization)",
    "float16": "float16 weights",
    "float8_e4m3fn": "float8 (e4m3fn) weights (experimental)",
    "float8_e5m2": "float8 (e5m2) weights (experimental)",
    "int8": "full integer quantization: int8 weights, activations, inputs and outputs (needs a representative dataset)",
}

_WEIGHT_TRANSFERS: dict[str, typing.Any] = {}
_LAZY_WEIGHT_TRANSFERS = {"hsnr_unet": ("opennve.networks.unet", "transfer_unet_weights")}


def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import keras  # type: ignore
    except ImportError as error:
        raise ImportError(
            "networks need TensorFlow (Keras 3): pip install tensorflow "
            "(for Python 3.14: pip install tf-nightly, see the openNVE documentation)"
        ) from error
    return keras


def _tensorflow():
    _keras()
    import tensorflow as tf  # type: ignore

    return tf


def load_keras_model(path: PathLike, safe_mode: bool = True, objects: typing.Optional[dict] = None, compile: bool = False):
    """keras.models.load_model with the registered custom objects (UNet layers, ...)."""
    keras = _keras()
    merged = custom_objects()
    if objects:
        merged.update(objects)
    return keras.models.load_model(os.fspath(path), custom_objects=merged, safe_mode=safe_mode, compile=compile)


def register_weight_transfer(name: str, function=None):
    """Registers a weight transfer function(source_model, destination_model, **options),
    used by Network.transfer_weights_from(..., method=name)."""

    def decorator(transfer):
        _WEIGHT_TRANSFERS[name] = transfer
        _LAZY_WEIGHT_TRANSFERS.pop(name, None)
        return transfer

    return decorator if function is None else decorator(function)


def _weight_transfer(name: str):
    if name not in _WEIGHT_TRANSFERS and name in _LAZY_WEIGHT_TRANSFERS:
        module, attribute = _LAZY_WEIGHT_TRANSFERS[name]
        _WEIGHT_TRANSFERS[name] = getattr(importlib.import_module(module), attribute)
    if name not in _WEIGHT_TRANSFERS:
        names = sorted(set(_WEIGHT_TRANSFERS) | set(_LAZY_WEIGHT_TRANSFERS))
        raise ValueError(f"unknown weight transfer {name!r} (registered: {', '.join(names)})")
    return _WEIGHT_TRANSFERS[name]


def transfer_weights_by_name(source_model, destination_model, mapping: dict[str, str]):
    """Copies weights layer by layer: mapping = {destination layer path: source layer name}.
    A destination path can go into nested layers with dots, for instance "u_net.bottleneck.conv1"
    (attributes) or "u_net.encoder_blocks.0.conv1" (list items)."""
    for destination_path, source_name in mapping.items():
        parts = destination_path.split(".")
        target = destination_model.get_layer(parts[0])
        for part in parts[1:]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        target.set_weights(source_model.get_layer(source_name).get_weights())


@dataclasses.dataclass
class WeightDifference:
    """Comparison of one weight array of two models (in the order of model.get_weights())."""

    index: int
    shape_a: tuple
    shape_b: tuple
    max_difference: typing.Optional[float]

    @property
    def same_shape(self) -> bool:
        return self.shape_a == self.shape_b


@dataclasses.dataclass
class WeightComparison:
    """Result of compare_weights: counts and one WeightDifference per array pair."""

    count_a: int
    count_b: int
    differences: list[WeightDifference]

    @property
    def equal(self) -> bool:
        """Same number and shapes of arrays and identical values."""
        return self.count_a == self.count_b and all(
            difference.same_shape and difference.max_difference == 0 for difference in self.differences
        )

    @property
    def max_difference(self) -> float:
        values = [d.max_difference for d in self.differences if d.max_difference is not None]
        return max(values) if values else 0.0

    def report(self) -> str:
        lines = [f"weight arrays: {self.count_a} / {self.count_b}"]
        for difference in self.differences:
            if difference.same_shape:
                lines.append(f"[{difference.index}] shape {difference.shape_a}, max difference {difference.max_difference}")
            else:
                lines.append(f"[{difference.index}] different shapes: {difference.shape_a} / {difference.shape_b}")
        return "\n".join(lines)

    def print(self):
        print(self.report())


def compare_weights(model_a, model_b) -> WeightComparison:
    """Compares the weights of two models (or Networks) array by array."""
    weights_a = (model_a.model if isinstance(model_a, Network) else model_a).get_weights()
    weights_b = (model_b.model if isinstance(model_b, Network) else model_b).get_weights()
    differences = []
    for index, (a, b) in enumerate(zip(weights_a, weights_b)):
        a, b = np.asarray(a), np.asarray(b)
        difference = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else (0.0 if a.shape == b.shape else None)
        differences.append(WeightDifference(index, tuple(a.shape), tuple(b.shape), difference))
    return WeightComparison(len(weights_a), len(weights_b), differences)


def _pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    padded = np.zeros((height, width) + image.shape[2:], dtype=np.float32)
    padded[: min(height, image.shape[0]), : min(width, image.shape[1])] = image[:height, :width]
    return padded


def representative_dataset(
    data,
    input_shape: typing.Optional[tuple[int, int, int]] = None,
    max_samples: int = 200,
    min_events: int = 1,
):
    """Representative dataset for int8 quantization (Network.to_tflite(quantization="int8")).

    data: frames (for instance raw.frames(tiles80) or a list of Frame), tiles, arrays
        (height x width x channels), or a camera / RawReader together with a frame configuration
        as (source, frame_config).
    input_shape: (height, width, channels) of the network; smaller tiles are padded.
    Returns a function that yields [array of shape (1, height, width, channels)] (as expected by
    the TensorFlow Lite converter).
    """
    from ..frames import Frame, Tile

    if isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], "frames"):
        source, frame_config = data
        data = source.frames(frame_config)

    def samples():
        count = 0
        for item in data:
            if isinstance(item, Frame):
                images = [tile.data for tile in item.tiles if (tile.event_count or 0) >= min_events] if item.tiles else [item.data]
            elif isinstance(item, Tile):
                images = [item.data]
            else:
                images = [np.asarray(item, dtype=np.float32)]
            for image in images:
                image = np.asarray(image, dtype=np.float32)
                if image.ndim == 4:
                    image = image[0]
                if input_shape is not None:
                    image = _pad(image, input_shape[0], input_shape[1])
                yield [image[None, ...]]
                count += 1
                if count >= max_samples:
                    return

    cached = list(samples())
    if not cached:
        raise ValueError("the representative dataset is empty (no frames / tiles with events)")

    def generator():
        yield from cached

    return generator


class Network:
    """A Keras model with its name, architecture and parameters.

        network = opennve.Network.build("UNet", opennve.NetworkParameters(number_of_classes=3))
        network = opennve.Network.load("unet.keras")
        network.save("unet.keras"); network.save_weights("unet.weights.h5")
        network.to_tflite("unet_int8.tflite", "int8", representative_dataset=...)
        agent = network.agent(prediction={"task": "segmentation", "activation": "softmax"})
    """

    def __init__(
        self,
        model,
        name: typing.Optional[str] = None,
        architecture: typing.Optional[str] = None,
        parameters: typing.Optional[NetworkParameters] = None,
        path: typing.Optional[PathLike] = None,
    ):
        self.model = model
        self.name = name or getattr(model, "name", None) or "network"
        self.architecture = architecture
        self.parameters = parameters
        self.path = None if path is None else pathlib.Path(path)

    # creation

    @classmethod
    def build(
        cls,
        architecture: str,
        parameters: typing.Union[NetworkParameters, dict],
        name: typing.Optional[str] = None,
        **options,
    ) -> "Network":
        """Builds a new (untrained) network of a registered architecture (see list_networks())."""
        if isinstance(parameters, dict):
            parameters = NetworkParameters.from_dict(parameters)
        _keras()
        model = build_network(architecture, parameters, **options)
        return cls(model, name=name or architecture, architecture=architecture, parameters=parameters)

    @classmethod
    def load(
        cls,
        path: PathLike,
        name: typing.Optional[str] = None,
        safe_mode: bool = True,
        custom_objects: typing.Optional[dict] = None,
        compile: bool = False,
    ) -> "Network":
        """Loads a Keras model (.keras, .h5). The registered custom objects (UNet layers, ...) are
        passed automatically. safe_mode=False allows models with Lambda layers."""
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        model = load_keras_model(path, safe_mode=safe_mode, objects=custom_objects, compile=compile)
        return cls(model, name=name or path.stem, path=path)

    # Keras model

    def save(self, path: PathLike):
        """Saves the model (.keras or .h5)."""
        try:
            self.model.save(os.fspath(path))
        except (OSError, ValueError):
            if self.parameters is None:
                raise
            self.model.build((None,) + self.parameters.input_shape)
            self.model.save(os.fspath(path))
        self.path = pathlib.Path(path)

    def load_weights(self, path: PathLike):
        self.model.load_weights(os.fspath(path))

    def save_weights(self, path: PathLike):
        """Saves the weights (the file name must end with .weights.h5)."""
        self.model.save_weights(os.fspath(path))

    def summary(self, expand_nested: bool = False, **kwargs):
        self.model.summary(expand_nested=expand_nested, **kwargs)

    def get_weights(self) -> list[np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, weights: list[np.ndarray]):
        self.model.set_weights(weights)

    @property
    def input_shape(self) -> tuple:
        shape = self.model.input_shape
        return tuple(shape[0] if isinstance(shape, list) else shape)

    @property
    def output_shape(self) -> tuple:
        shape = self.model.output_shape
        return tuple(shape[0] if isinstance(shape, list) else shape)

    def predict(self, data, batch_size: typing.Optional[int] = None, verbose: int = 0) -> np.ndarray:
        """model.predict on a batch (array: batch x height x width x channels), a list of tiles or a frame
        (its tiles, padded to the network input size)."""
        from ..frames import Frame, Tile

        if isinstance(data, Frame):
            data = data.tiles if data.tiles else [data.data]
        if isinstance(data, (list, tuple)):
            height, width = self.input_shape[1], self.input_shape[2]
            images = [np.asarray(item.data if isinstance(item, Tile) else item, dtype=np.float32) for item in data]
            if height and width:
                images = [_pad(image, height, width) for image in images]
            data = np.stack(images) if images else np.zeros((0,) + tuple(self.input_shape[1:]), np.float32)
        data = np.asarray(data, dtype=np.float32)
        if len(data) == 0:
            return np.zeros((0,) + tuple(s or 0 for s in self.output_shape[1:]), np.float32)
        return np.asarray(self.model.predict(data, batch_size=batch_size, verbose=verbose))

    def compare_weights(self, other) -> WeightComparison:
        return compare_weights(self, other)

    def transfer_weights_from(self, source, method: str = "hsnr_unet", **options) -> WeightComparison:
        """Copies the weights of another network with a registered transfer method
        ("hsnr_unet": functional U-Net with enc_conv{i}-1 ... layers into a UNet) and returns the
        weight comparison."""
        source_model = source.model if isinstance(source, Network) else source
        _weight_transfer(method)(source_model, self.model, **options)
        return compare_weights(source_model, self.model)

    # TensorFlow Lite

    def to_tflite(
        self,
        path: PathLike,
        quantization: str = "none",
        representative_dataset=None,
    ) -> pathlib.Path:
        """Converts the network to TensorFlow Lite and writes it to path.

        quantization: see QUANTIZATIONS ("none", "dynamic", "float32", "float16", "float8_e4m3fn",
            "float8_e5m2", "int8"). "int8" needs representative_dataset (see representative_dataset()).
        """
        if quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {', '.join(QUANTIZATIONS)}")
        tf = _tensorflow()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if quantization != "none":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization in ("float8_e4m3fn", "float8_e5m2"):
            converter.target_spec.supported_types = [getattr(tf.experimental, quantization)]
        elif quantization == "float32":
            converter.target_spec.supported_types = [tf.float32]
        elif quantization == "int8":
            if representative_dataset is None:
                raise ValueError("int8 quantization needs a representative dataset (opennve.representative_dataset(...))")
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(converter.convert())
        return path

    # agents

    def agent(self, name: typing.Optional[str] = None, **settings):
        """Agent that runs this network directly (model.predict), for instance
        network.agent(frame="tiles80", prediction={"task": "segmentation", "activation": "softmax"})."""
        from ..agents import Agent

        return Agent.from_model(self.model, name=name or self.name, **settings)

    def __repr__(self) -> str:
        return f"Network(name={self.name!r}, architecture={self.architecture!r}, input={self.input_shape})"
