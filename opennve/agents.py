"""Agents: run TensorFlow and TensorFlow Lite models on frames and tiles.

An agent combines a model with the settings needed to feed it: which input it gets (the tiles
of a frame, or the whole frame), which frame configuration builds that input, how the data is
prepared and how the outputs are read. Agent configurations are kept in their own file, like
the camera and frame configurations:

    {
        "agents": [
            {
                "name": "insects",
                "model": "models/insects.tflite",
                "backend": "auto",
                "input": "tiles",
                "frame": "tiles80",
                "batch_size": 64,
                "threads": 2,
                "preprocess": {"channels": "both", "pad": true, "resize": null, "layout": "nhwc", "scale": 1.0},
                "prediction": {"activation": "none", "labels": ["background", "insect"], "threshold": 0.5}
            }
        ]
    }

Backends (backend "auto" chooses by the model path):

    tflite:     .tflite files, run with LiteRT (pip install ai-edge-litert), tflite_runtime
                or tf.lite (TensorFlow)
    tensorflow: Keras models (.keras, .h5) run with model.predict(...), and SavedModel
                directories run with their serving signature (pip install "opennve[tensorflow]")

Use:

    agents = opennve.load_agents("agents.json")
    frame_configs = opennve.load_frame_configs("frames.json")
    for step in opennve.run_agents(camera, agents, frame_configs):
        for prediction in step.results["insects"].selected:
            print(prediction.tile.id, prediction.label, prediction.score)

or, for one frame: result = agent.predict(frame).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
import typing

import numpy as np

from .frames import Frame, FrameConfig, Tile, as_frame_configs

PathLike = typing.Union[str, pathlib.Path]

_BACKENDS: dict[str, typing.Callable[..., "Backend"]] = {}  # name -> factory(config), see register_backend
_BACKEND_SUFFIXES: dict[str, str] = {}  # model file suffix -> backend name
_TASKS: dict[str, typing.Callable[..., list]] = {}  # name -> decoder, see register_task
INPUTS = ("tiles", "frame")
CHANNELS = ("both", "on", "off", "sum", "max")
LAYOUTS = ("nhwc", "nchw")
ACTIVATIONS = ("none", "softmax", "sigmoid")
CALLS = ("predict", "call")


# ------------------------------------------------------------------ configuration


@dataclasses.dataclass
class PreprocessConfig:
    """How frames and tiles are converted to model inputs.

    channels: "both" (2 channels: polarity 0, polarity 1), "on" / "off" (1 channel: polarity 1 / 0),
        "sum" or "max" (1 channel: both polarities combined).
    pad: pad the smaller tiles at the right and bottom edges with zeros to the full tile size.
    resize: [width, height] to resize the input to, or None: the model's input size if the
        model has a fixed size, otherwise unchanged.
    layout: "nhwc" (batch, height, width, channels; TensorFlow default) or "nchw".
    scale: factor applied to the values (for instance 255 for models trained on 0-255 images).
    """

    channels: str = "both"
    pad: bool = True
    resize: typing.Optional[tuple[int, int]] = None
    layout: str = "nhwc"
    scale: float = 1.0

    def validate(self):
        if self.channels not in CHANNELS:
            raise ValueError(f"agent.preprocess.channels must be one of {', '.join(CHANNELS)}")
        if not isinstance(self.pad, bool):
            raise ValueError("agent.preprocess.pad must be true or false")
        if self.resize is not None:
            if (
                not isinstance(self.resize, (list, tuple))
                or len(self.resize) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.resize)
            ):
                raise ValueError("agent.preprocess.resize must be [width, height] or null")
            self.resize = (int(self.resize[0]), int(self.resize[1]))
        if self.layout not in LAYOUTS:
            raise ValueError(f"agent.preprocess.layout must be one of {', '.join(LAYOUTS)}")
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)) or self.scale == 0:
            raise ValueError("agent.preprocess.scale must be a non-zero number")

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "channels": self.channels,
            "pad": self.pad,
            "resize": None if self.resize is None else list(self.resize),
            "layout": self.layout,
            "scale": self.scale,
        }


@dataclasses.dataclass
class PredictionConfig:
    """How the model outputs are read.

    task: how outputs are interpreted (see register_task):
        "classification": one output vector per input (class id, label, score).
        "segmentation": one class per pixel (output height x width x classes, for instance a U-Net):
            class map, pixels of the target classes with frame coordinates and scores.
    activation: "none" (the model already returns probabilities or scores), "softmax" or "sigmoid"
        (applied to the first output along the class axis, for models that return logits).
    labels: class names, index = class id.
    threshold: classification: selected if score >= threshold (None: all are selected).
        segmentation: a pixel counts for a target class only if its probability >= threshold
        (None: the class with the highest probability).
    target_classes: segmentation: classes to detect, for instance [2] (None: all classes except 0).
    min_pixels: segmentation: a tile is selected if it has at least this many target pixels.
    mask_inactive: segmentation: ignore target pixels without events in the input (all channels
        <= activity_threshold).
    activity_threshold: segmentation: a pixel is active if one of its input channels is above this value.
    """

    activation: str = "none"
    labels: typing.Optional[list[str]] = None
    threshold: typing.Optional[float] = None
    task: str = "classification"
    target_classes: typing.Optional[list[int]] = None
    min_pixels: int = 1
    mask_inactive: bool = False
    activity_threshold: float = 0.0

    def validate(self):
        if self.task not in _TASKS:
            raise ValueError(f"agent.prediction.task must be one of {', '.join(list_tasks())}")
        if self.target_classes is not None:
            if not isinstance(self.target_classes, (list, tuple)) or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in self.target_classes
            ):
                raise ValueError("agent.prediction.target_classes must be a list of class ids or null")
            self.target_classes = [int(value) for value in self.target_classes]
        if isinstance(self.min_pixels, bool) or not isinstance(self.min_pixels, int) or self.min_pixels < 0:
            raise ValueError("agent.prediction.min_pixels must be an integer >= 0")
        if not isinstance(self.mask_inactive, bool):
            raise ValueError("agent.prediction.mask_inactive must be true or false")
        if isinstance(self.activity_threshold, bool) or not isinstance(self.activity_threshold, (int, float)):
            raise ValueError("agent.prediction.activity_threshold must be a number")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"agent.prediction.activation must be one of {', '.join(ACTIVATIONS)}")
        if self.labels is not None and (
            not isinstance(self.labels, (list, tuple)) or not all(isinstance(label, str) for label in self.labels)
        ):
            raise ValueError("agent.prediction.labels must be a list of strings or null")
        if self.labels is not None:
            self.labels = list(self.labels)
        if self.threshold is not None and (
            isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float))
        ):
            raise ValueError("agent.prediction.threshold must be a number or null")

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "task": self.task,
            "activation": self.activation,
            "labels": self.labels,
            "threshold": self.threshold,
            "target_classes": self.target_classes,
            "min_pixels": self.min_pixels,
            "mask_inactive": self.mask_inactive,
            "activity_threshold": self.activity_threshold,
        }


_AGENT_FIELDS = (
    "name", "model", "backend", "input", "frame", "batch_size", "threads", "call", "signature",
    "safe_mode", "preprocess", "prediction",
)


@dataclasses.dataclass
class AgentConfig:
    """Configuration of one agent (see the module documentation for the JSON format).

    model: path of the model (.tflite, .keras, .h5 or a SavedModel directory). Relative paths
        are relative to the agent configuration file (base_dir).
    backend: "auto", "tflite" or "tensorflow".
    input: "tiles" (one model input per tile) or "frame" (the whole frame).
    frame: name of the frame configuration that builds the input (see run_agents), or None.
    batch_size: inputs per model call (None: all tiles of a frame at once).
    threads: CPU threads of the TensorFlow Lite interpreter (None: default).
    call: TensorFlow (Keras) models: "predict" (model.predict) or "call" (model(x), lower latency
        for small batches).
    signature: SavedModel signature (default "serving_default").
    """

    model: str
    name: typing.Optional[str] = None
    backend: str = "auto"
    input: str = "tiles"
    frame: typing.Optional[str] = None
    batch_size: typing.Optional[int] = None
    threads: typing.Optional[int] = None
    call: str = "predict"
    signature: str = "serving_default"
    safe_mode: bool = True
    preprocess: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)
    prediction: PredictionConfig = dataclasses.field(default_factory=PredictionConfig)
    base_dir: typing.Optional[pathlib.Path] = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if isinstance(self.preprocess, dict):
            self.preprocess = _preprocess_from_dict(self.preprocess)
        if isinstance(self.prediction, dict):
            self.prediction = _prediction_from_dict(self.prediction)
        if self.model is not None and not isinstance(self.model, str):
            self.model = os.fspath(self.model)
        if self.name is None and isinstance(self.model, str) and self.model:
            self.name = pathlib.Path(self.model).stem
        self.validate()

    def validate(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("agent.model must be the path of a model file")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("agent.name must be a non-empty string")
        if self.backend != "auto" and self.backend not in _BACKENDS:
            raise ValueError(f"agent.backend must be one of auto, {', '.join(sorted(_BACKENDS))}")
        if not isinstance(self.safe_mode, bool):
            raise ValueError("agent.safe_mode must be true or false")
        if self.input not in INPUTS:
            raise ValueError(f"agent.input must be one of {', '.join(INPUTS)}")
        if self.frame is not None and (not isinstance(self.frame, str) or not self.frame):
            raise ValueError("agent.frame must be the name of a frame configuration or null")
        for name in ("batch_size", "threads"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"agent.{name} must be a positive integer or null")
        if self.call not in CALLS:
            raise ValueError(f"agent.call must be one of {', '.join(CALLS)}")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("agent.signature must be a string")
        self.preprocess.validate()
        self.prediction.validate()

    @property
    def model_path(self) -> pathlib.Path:
        """Absolute model path (relative paths are resolved against base_dir)."""
        path = pathlib.Path(self.model).expanduser()
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / path
        return path

    @property
    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        path = self.model_path
        if path.suffix.lower() in _BACKEND_SUFFIXES:
            return _BACKEND_SUFFIXES[path.suffix.lower()]
        if path.is_dir():
            return "tensorflow"
        raise ValueError(
            f"agent {self.name}: cannot choose a backend for {self.model!r} "
            "(.tflite, .keras, .h5 or a SavedModel directory), set \"backend\""
        )

    # JSON

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "model": self.model,
            "backend": self.backend,
            "input": self.input,
            "frame": self.frame,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "call": self.call,
            "signature": self.signature,
            "safe_mode": self.safe_mode,
            "preprocess": self.preprocess.to_dict(),
            "prediction": self.prediction.to_dict(),
        }

    def to_json(self, indent: typing.Optional[int] = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any], base_dir: typing.Optional[PathLike] = None) -> "AgentConfig":
        if not isinstance(data, dict):
            raise ValueError("an agent configuration must be an object")
        unknown = set(data) - set(_AGENT_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown agent setting(s): {', '.join(sorted(unknown))} (supported: {', '.join(_AGENT_FIELDS)})"
            )
        if "model" not in data:
            raise ValueError("agent.model is required")
        values = {key: data[key] for key in _AGENT_FIELDS if key in data and key not in ("preprocess", "prediction")}
        for key in ("backend", "input", "call"):
            if isinstance(values.get(key), str):
                values[key] = values[key].lower()
        return cls(
            **values,
            preprocess=_preprocess_from_dict(data.get("preprocess")),
            prediction=_prediction_from_dict(data.get("prediction")),
            base_dir=None if base_dir is None else pathlib.Path(base_dir),
        )

    @classmethod
    def from_json(cls, text: str, base_dir: typing.Optional[PathLike] = None) -> "AgentConfig":
        return cls.from_dict(json.loads(text), base_dir)

    def copy(self, **changes) -> "AgentConfig":
        """Copy with changes, for instance config.copy(name="fast", batch_size=128)."""
        return AgentConfig.from_dict({**self.to_dict(), **changes}, self.base_dir)


def _preprocess_from_dict(data) -> PreprocessConfig:
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("agent.preprocess must be an object")
    unknown = set(data) - {"channels", "pad", "resize", "layout", "scale"}
    if unknown:
        raise ValueError(f"unknown agent.preprocess setting(s): {', '.join(sorted(unknown))}")
    data = dict(data)
    for key in ("channels", "layout"):
        if isinstance(data.get(key), str):
            data[key] = data[key].lower()
    config = PreprocessConfig(**data)
    config.validate()
    return config


def _prediction_from_dict(data) -> PredictionConfig:
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("agent.prediction must be an object")
    unknown = set(data) - {f.name for f in dataclasses.fields(PredictionConfig)}
    if unknown:
        raise ValueError(f"unknown agent.prediction setting(s): {', '.join(sorted(unknown))}")
    data = dict(data)
    for key in ("activation", "task"):
        if isinstance(data.get(key), str):
            data[key] = data[key].lower()
    config = PredictionConfig(**data)
    config.validate()
    return config


# ------------------------------------------------------------------ files


def agent_configs_from_data(
    data: typing.Any, source: str = "agent configuration", base_dir: typing.Optional[PathLike] = None
) -> list[AgentConfig]:
    """Agent configurations from parsed JSON: one object, a list or {"agents": [...]}."""
    if isinstance(data, dict) and "agents" in data:
        data = data["agents"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected an agent object, a list or {{\"agents\": [...]}}")
    configs = []
    for index, item in enumerate(data):
        try:
            configs.append(AgentConfig.from_dict(item, base_dir))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{source}: agent {index}: {error}") from error
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate agent name(s): {', '.join(duplicates)}")
    return configs


def load_agent_configs(path: PathLike) -> list[AgentConfig]:
    """Loads an agent configuration file (one agent, a list or {"agents": [...]}).
    Relative model paths are relative to the file."""
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return agent_configs_from_data(data, str(path), path.resolve().parent)


def load_agent_config(path: PathLike, name: typing.Optional[str] = None) -> AgentConfig:
    """Loads one agent configuration (by name if the file has several)."""
    configs = load_agent_configs(path)
    if name is None:
        if len(configs) != 1:
            raise ValueError(f"{path} contains {len(configs)} agents, specify a name")
        return configs[0]
    for config in configs:
        if config.name == name:
            return config
    names = ", ".join(config.name for config in configs)
    raise ValueError(f"{path}: no agent named {name!r} (available: {names})")


def save_agent_configs(path: PathLike, configs: typing.Union[AgentConfig, typing.Sequence[AgentConfig]]):
    """Saves one agent (as an object) or several (as {"agents": [...]})."""
    if isinstance(configs, AgentConfig):
        data: typing.Any = configs.to_dict()
    else:
        data = {"agents": [config.to_dict() for config in configs]}
    pathlib.Path(path).write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def load_agents(path: PathLike) -> list["Agent"]:
    """Agents of a configuration file. The models are loaded on first use (or with agent.load())."""
    return [Agent(config) for config in load_agent_configs(path)]


def load_agent(path: PathLike, name: typing.Optional[str] = None) -> "Agent":
    """One agent of a configuration file (by name if the file has several)."""
    return Agent(load_agent_config(path, name))


# ------------------------------------------------------------------ backends


@dataclasses.dataclass
class ModelInput:
    """Input of a model: shape (None = any size) and dtype."""

    shape: tuple[typing.Optional[int], ...]
    dtype: np.dtype
    name: str = ""


class Backend:
    """Runs a model on a batch: predict(batch) -> list of output arrays (first dimension = batch)."""

    input: ModelInput

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self):
        pass


def _tflite_interpreter_class():
    errors = []
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError as error:
        errors.append(str(error))
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter
    except ImportError as error:
        errors.append(str(error))
    try:
        import tensorflow as tf  # type: ignore

        return tf.lite.Interpreter
    except ImportError as error:
        errors.append(str(error))
    raise ImportError(
        "TensorFlow Lite models need LiteRT: pip install ai-edge-litert "
        "(installed with openNVE where wheels exist), or TensorFlow: pip install \"opennve[tensorflow]\""
    )


class TFLiteBackend(Backend):
    """TensorFlow Lite model (.tflite), run with LiteRT, tflite_runtime or tf.lite.

    Quantized models (int8 / uint8 inputs and outputs) are quantized and dequantized automatically.
    Models with a fixed batch size are called several times, models with a dynamic batch
    dimension are resized to the batch.
    """

    def __init__(self, path: PathLike, threads: typing.Optional[int] = None):
        Interpreter = _tflite_interpreter_class()
        kwargs = {"model_path": os.fspath(path)}
        if threads is not None:
            kwargs["num_threads"] = threads
        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        details = self.interpreter.get_input_details()
        if len(details) != 1:
            raise ValueError(f"{path}: only models with one input are supported, this one has {len(details)}")
        self._input = details[0]
        signature = self._input.get("shape_signature", self._input["shape"])
        self._dynamic_batch = len(signature) > 0 and int(signature[0]) == -1
        self.input = ModelInput(
            shape=tuple(None if int(size) == -1 else int(size) for size in signature),
            dtype=np.dtype(self._input["dtype"]),
            name=self._input.get("name", ""),
        )
        self._batch = int(self._input["shape"][0]) if len(self._input["shape"]) else 1

    def _resize(self, batch_size: int):
        if self._batch != batch_size:
            shape = list(self._input["shape"])
            shape[0] = batch_size
            self.interpreter.resize_tensor_input(self._input["index"], shape)
            self.interpreter.allocate_tensors()
            self._input = self.interpreter.get_input_details()[0]
            self._batch = batch_size

    def _run(self, batch: np.ndarray) -> list[np.ndarray]:
        dtype = np.dtype(self._input["dtype"])
        scale, zero_point = self._input.get("quantization", (0.0, 0))
        if np.issubdtype(dtype, np.integer) and scale:
            info = np.iinfo(dtype)
            batch = np.clip(np.round(batch / scale + zero_point), info.min, info.max)
        self.interpreter.set_tensor(self._input["index"], np.ascontiguousarray(batch, dtype=dtype))
        self.interpreter.invoke()
        outputs = []
        for detail in sorted(self.interpreter.get_output_details(), key=lambda item: item["index"]):
            output = self.interpreter.get_tensor(detail["index"])
            scale, zero_point = detail.get("quantization", (0.0, 0))
            if np.issubdtype(output.dtype, np.integer) and scale:
                output = (output.astype(np.float32) - zero_point) * scale
            outputs.append(np.array(output))
        return outputs

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        count = len(batch)
        if count == 0:
            return []
        if self._dynamic_batch:
            self._resize(count)
            return self._run(batch)
        # fixed batch size (often 1): run in chunks, pad the last chunk
        size = self._batch
        results: list[list[np.ndarray]] = []
        for start in range(0, count, size):
            chunk = batch[start : start + size]
            if len(chunk) < size:
                chunk = np.concatenate([chunk, np.zeros((size - len(chunk),) + chunk.shape[1:], chunk.dtype)])
            outputs = self._run(chunk)
            results.append([output[: min(size, count - start)] for output in outputs])
        return [np.concatenate([result[index] for result in results]) for index in range(len(results[0]))]

    def close(self):
        self.interpreter = None


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as error:
        raise ImportError(
            "TensorFlow models need TensorFlow: pip install \"opennve[tensorflow]\" "
            "(installs tf-nightly, see the openNVE documentation), "
            "or convert the model to TensorFlow Lite (.tflite) and use LiteRT"
        ) from error
    return tf


def _load_keras_model(path: str, safe_mode: bool = True):
    """Loads a Keras model with the registered custom objects (UNet layers, register_custom_objects)."""
    _import_tensorflow()
    from .networks import load_keras_model

    return load_keras_model(path, safe_mode=safe_mode)


class TensorFlowBackend(Backend):
    """TensorFlow model: Keras files (.keras, .h5) run with model.predict (or model(x) with
    call="call"), SavedModel directories run with their serving signature."""

    def __init__(
        self, path: PathLike, call: str = "predict", signature: str = "serving_default", safe_mode: bool = True
    ):
        path = pathlib.Path(path)
        self.call = call
        self.model = None
        self._function = None
        if path.is_dir():
            tf = _import_tensorflow()
            loaded = tf.saved_model.load(os.fspath(path))
            if signature not in loaded.signatures:
                names = ", ".join(loaded.signatures) or "none"
                raise ValueError(f"{path}: no signature {signature!r} (available: {names})")
            self._loaded = loaded  # keeps the variables alive
            self._function = loaded.signatures[signature]
            args, kwargs = self._function.structured_input_signature
            specs = list(kwargs.items())
            if len(specs) != 1:
                raise ValueError(f"{path}: only signatures with one input are supported")
            self._input_name, spec = specs[0]
            self._output_names = sorted(self._function.structured_outputs)
            self.input = ModelInput(
                shape=tuple(spec.shape.as_list()), dtype=np.dtype(spec.dtype.as_numpy_dtype), name=self._input_name
            )
            self._tf = tf
        else:
            self.model = _load_keras_model(os.fspath(path), safe_mode)
            inputs = self.model.inputs
            if len(inputs) != 1:
                raise ValueError(f"{path}: only models with one input are supported")
            dtype = getattr(inputs[0], "dtype", "float32")
            dtype = getattr(dtype, "as_numpy_dtype", dtype)
            self.input = ModelInput(shape=tuple(inputs[0].shape), dtype=np.dtype(dtype), name=getattr(inputs[0], "name", ""))

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        if len(batch) == 0:
            return []
        batch = np.ascontiguousarray(batch, dtype=self.input.dtype)
        if self._function is not None:
            outputs = self._function(**{self._input_name: self._tf.constant(batch)})
            return [np.asarray(outputs[name]) for name in self._output_names]
        if self.call == "call":
            outputs = self.model(batch, training=False)
        else:
            outputs = self.model.predict(batch, verbose=0, batch_size=len(batch))
        if isinstance(outputs, dict):
            return [np.asarray(outputs[name]) for name in sorted(outputs)]
        if isinstance(outputs, (list, tuple)):
            return [np.asarray(output) for output in outputs]
        return [np.asarray(outputs)]

    def close(self):
        self.model = None
        self._function = None


class ModelBackend(Backend):
    """Any Python object with predict(batch), for instance a Keras model created in code."""

    def __init__(self, model: typing.Any, input_shape: typing.Optional[tuple] = None, dtype: typing.Any = np.float32):
        if not hasattr(model, "predict") and not callable(model):
            raise TypeError("the model must have a predict(batch) method or be callable")
        self.model = model
        if input_shape is None:
            input_shape = getattr(model, "input_shape", None)
            if isinstance(input_shape, list):
                input_shape = input_shape[0] if len(input_shape) == 1 else None
        self.input = ModelInput(shape=tuple(input_shape) if input_shape else (None, None, None, None), dtype=np.dtype(dtype))

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        if len(batch) == 0:
            return []
        batch = np.ascontiguousarray(batch, dtype=self.input.dtype)
        if hasattr(self.model, "predict"):
            try:
                outputs = self.model.predict(batch, verbose=0)
            except TypeError:
                outputs = self.model.predict(batch)
        else:
            outputs = self.model(batch)
        if isinstance(outputs, dict):
            return [np.asarray(outputs[name]) for name in sorted(outputs)]
        if isinstance(outputs, (list, tuple)):
            return [np.asarray(output) for output in outputs]
        return [np.asarray(outputs)]


def register_backend(name: str, factory=None, suffixes: typing.Sequence[str] = ()):
    """Registers a model runtime: factory(config: AgentConfig) -> Backend.
    suffixes: model file suffixes chosen by backend "auto", for instance [".onnx"].
    Usable as a decorator: @register_backend("onnx", suffixes=[".onnx"])."""

    def decorator(function):
        _BACKENDS[name] = function
        for suffix in suffixes:
            _BACKEND_SUFFIXES[suffix.lower()] = name
        return function

    return decorator if factory is None else decorator(factory)


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


register_backend("tflite", lambda config: TFLiteBackend(config.model_path, config.threads), [".tflite"])
register_backend(
    "tensorflow",
    lambda config: TensorFlowBackend(config.model_path, config.call, config.signature, config.safe_mode),
    [".keras", ".h5", ".hdf5"],
)


def create_backend(config: AgentConfig) -> Backend:
    """Loads the model of an agent configuration with its backend."""
    path = config.model_path
    if not path.exists():
        raise FileNotFoundError(f"agent {config.name}: model not found: {path}")
    return _BACKENDS[config.resolved_backend](config)


# ------------------------------------------------------------------ results


@dataclasses.dataclass
class Prediction:
    """Prediction of an agent for one tile (or for the whole frame).

    Attributes:
        tile: The tile (None for "input": "frame").
        output: Output of the model for this input (first model output, after the activation).
        outputs: All model outputs for this input.
        class_id: Index of the highest value if the output is a vector (else None).
        label: prediction.labels[class_id] if labels are configured (else None).
        score: Highest value of the output vector (one value: that value), else None.
        selected: score >= threshold (True if no threshold is configured).
    """

    tile: typing.Optional[Tile]
    output: np.ndarray
    outputs: list[np.ndarray]
    class_id: typing.Optional[int] = None
    label: typing.Optional[str] = None
    score: typing.Optional[float] = None
    selected: bool = True
    class_map: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    pixels: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    pixel_scores: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    class_counts: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    activity_counts: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    labels: typing.Optional[list[str]] = dataclasses.field(default=None, repr=False)

    @property
    def tile_id(self) -> typing.Optional[int]:
        return None if self.tile is None else self.tile.id

    @property
    def pixel_count(self) -> int:
        """Segmentation: number of target pixels."""
        return 0 if self.pixels is None else len(self.pixels)

    def class_name(self, class_id: int) -> str:
        if self.labels and class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    def pixel_list(self) -> list[dict[str, typing.Any]]:
        """Segmentation: target pixels as dictionaries {x, y, class_id, score, scores: {class name: probability}}
        (x, y in frame coordinates)."""
        if self.pixels is None:
            return []
        names = [self.class_name(index) for index in range(self.pixel_scores.shape[1])] if len(self.pixels) else []
        return [
            {
                "x": int(pixel["x"]),
                "y": int(pixel["y"]),
                "class_id": int(pixel["class_id"]),
                "score": float(pixel["score"]),
                "scores": {name: float(value) for name, value in zip(names, scores)},
            }
            for pixel, scores in zip(self.pixels, self.pixel_scores)
        ]

    def to_dict(self) -> dict[str, typing.Any]:
        """JSON-friendly summary (without the tile data)."""
        result: dict[str, typing.Any] = {}
        if self.tile is not None:
            result.update(
                {
                    "tile_id": self.tile.id,
                    "row": self.tile.row,
                    "col": self.tile.col,
                    "x": self.tile.x,
                    "y": self.tile.y,
                    "width": self.tile.width,
                    "height": self.tile.height,
                }
            )
        result.update({"class_id": self.class_id, "label": self.label, "score": self.score})
        if self.class_map is None:
            result["output"] = self.output.tolist()
        else:
            result["class_counts"] = {
                self.class_name(index): int(count) for index, count in enumerate(self.class_counts)
            }
            result["pixels"] = self.pixel_list()
        return result


@dataclasses.dataclass
class AgentResult:
    """Result of one agent for one frame.

    Attributes:
        agent: Agent name.
        frame: The frame (None if the agent was called with tiles or an array).
        predictions: One prediction per tile (or one for the whole frame), in tile order.
        outputs: Raw model outputs for all inputs (first dimension = number of inputs).
        inference_ms: Time spent in the model (without preprocessing).
    """

    agent: str
    frame: typing.Optional[Frame]
    predictions: list[Prediction]
    outputs: list[np.ndarray]
    inference_ms: float = 0.0

    @property
    def selected(self) -> list[Prediction]:
        """Predictions with score >= threshold (all predictions if no threshold is set)."""
        return [prediction for prediction in self.predictions if prediction.selected]

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self):
        return iter(self.predictions)

    def by_tile(self) -> dict[int, Prediction]:
        """{tile id: prediction}."""
        return {prediction.tile.id: prediction for prediction in self.predictions if prediction.tile is not None}

    def statistics(self) -> dict[str, typing.Any]:
        """Segmentation statistics over all predictions: pixels per class, active / inactive pixels per
        class (active: input has events), target pixel probabilities (mean, median, min, max)."""
        segmented = [p for p in self.predictions if p.class_counts is not None]
        if not segmented:
            return {"inputs": len(self.predictions), "selected": len(self.selected)}
        classes = max(len(p.class_counts) for p in segmented)
        class_counts = np.zeros(classes, np.int64)
        activity = np.zeros((2, classes), np.int64)
        for prediction in segmented:
            class_counts[: len(prediction.class_counts)] += prediction.class_counts
            activity[:, : prediction.activity_counts.shape[1]] += prediction.activity_counts
        scores = [p.pixels["score"] for p in segmented if p.pixels is not None and len(p.pixels)]
        scores = np.concatenate(scores) if scores else np.zeros(0, np.float32)
        names = [segmented[0].class_name(index) for index in range(classes)]
        return {
            "inputs": len(self.predictions),
            "selected": len(self.selected),
            "pixels": int(class_counts.sum()),
            "active_pixels": int(activity[1].sum()),
            "inactive_pixels": int(activity[0].sum()),
            "class_pixels": dict(zip(names, class_counts.tolist())),
            "active_class_pixels": dict(zip(names, activity[1].tolist())),
            "inactive_class_pixels": dict(zip(names, activity[0].tolist())),
            "target_pixels": int(len(scores)),
            "target_score": None
            if not len(scores)
            else {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "min": float(scores.min()),
                "max": float(scores.max()),
            },
        }

    def statistics_report(self) -> str:
        """The statistics as text (for debugging a segmentation model)."""
        stats = self.statistics()
        lines = [f"agent {self.agent}: {stats['inputs']} inputs, {stats['selected']} selected"]
        if "pixels" not in stats:
            return lines[0]
        total, active, inactive = stats["pixels"], stats["active_pixels"], stats["inactive_pixels"]
        lines.append(f"pixels: {total}, active: {active}, inactive: {inactive}")

        def share(count, whole):
            return f"{count} ({count / whole:.2%})" if whole else f"{count}"

        for title, key, whole in (
            ("active pixels", "active_class_pixels", active),
            ("inactive pixels", "inactive_class_pixels", inactive),
            ("all pixels", "class_pixels", total),
        ):
            lines.append(f"{title}: " + ", ".join(f"{name} {share(count, whole)}" for name, count in stats[key].items()))
        score = stats["target_score"]
        if score is None:
            lines.append("target pixels: none")
        else:
            lines.append(
                f"target pixels: {stats['target_pixels']}, probability mean {score['mean']:.4f}, "
                f"median {score['median']:.4f}, min {score['min']:.4f}, max {score['max']:.4f}"
            )
        return "\n".join(lines)

    def print_statistics(self):
        print(self.statistics_report(), flush=True)

    def to_dict(self, selected_only: bool = False) -> dict[str, typing.Any]:
        predictions = self.selected if selected_only else self.predictions
        return {
            "agent": self.agent,
            "camera": None if self.frame is None else self.frame.camera,
            "frame": None if self.frame is None else self.frame.name,
            "start_t": None if self.frame is None else self.frame.start_t,
            "end_t": None if self.frame is None else self.frame.end_t,
            "inference_ms": self.inference_ms,
            "predictions": [prediction.to_dict() for prediction in predictions],
        }


# ------------------------------------------------------------------ agent


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    import cv2

    channels = image.shape[2]
    if channels <= 4:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        resized = np.stack(
            [cv2.resize(image[:, :, c], (width, height), interpolation=cv2.INTER_AREA) for c in range(channels)],
            axis=2,
        )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized


class Agent:
    """A model with its configuration: agent.predict(frame) runs it on the tiles (or the frame).

    The model is loaded on first use, or with agent.load(). An already loaded model (for
    instance a Keras model created in code) can be passed with model=...
    """

    def __init__(self, config: typing.Union[AgentConfig, dict, str], model: typing.Any = None):
        if isinstance(config, str):
            config = AgentConfig.from_json(config)
        elif isinstance(config, dict):
            config = AgentConfig.from_dict(config)
        config.validate()
        self.config = config
        self._backend: typing.Optional[Backend] = None
        if model is not None:
            self._backend = model if isinstance(model, Backend) else ModelBackend(model)

    @classmethod
    def from_model(cls, model: typing.Any, name: str = "model", **settings) -> "Agent":
        """Agent for a model object with predict(batch), for instance a Keras model.
        settings: further agent settings, for instance input="frame" or prediction={...}."""
        return cls(AgentConfig(model=f"<{name}>", name=name, **settings), model=model)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> "Agent":
        """Loads the model now (it is otherwise loaded on first use)."""
        if self._backend is None:
            self._backend = create_backend(self.config)
        return self

    @property
    def backend(self) -> Backend:
        return self.load()._backend

    @property
    def input(self) -> ModelInput:
        """Model input (shape and dtype)."""
        return self.backend.input

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> "Agent":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.config.model!r}, input={self.config.input!r})"

    # preprocessing

    def _model_size(self) -> tuple[typing.Optional[int], typing.Optional[int], typing.Optional[int]]:
        """(height, width, channels) of the model input, None where not fixed."""
        shape = self.input.shape
        if len(shape) != 4:
            return None, None, None
        if self.config.preprocess.layout == "nchw":
            return shape[2], shape[3], shape[1]
        return shape[1], shape[2], shape[3]

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            data = data[:, :, None]
        channels = self.config.preprocess.channels
        if data.shape[2] != 2:
            return data if channels == "both" else data[:, :, :1]
        if channels == "both":
            return data
        if channels == "on":
            return data[:, :, 1:2]
        if channels == "off":
            return data[:, :, 0:1]
        if channels == "sum":
            return data.sum(axis=2, keepdims=True)
        return data.max(axis=2, keepdims=True)

    def prepare(self, images: typing.Sequence[np.ndarray], pad_to: typing.Optional[tuple[int, int]] = None) -> np.ndarray:
        """Converts frames or tiles (height x width x channels) to a model input batch."""
        preprocess = self.config.preprocess
        model_height, model_width, model_channels = self._model_size()
        if preprocess.resize is not None:
            target = (preprocess.resize[1], preprocess.resize[0])
        elif model_height is not None and model_width is not None:
            target = (model_height, model_width)
        else:
            target = None
        prepared = []
        for image in images:
            image = np.asarray(image, dtype=np.float32)
            image = self._select_channels(image)
            if pad_to is not None and preprocess.pad and image.shape[:2] != pad_to:
                padded = np.zeros(pad_to + image.shape[2:], dtype=np.float32)
                padded[: image.shape[0], : image.shape[1]] = image
                image = padded
            if target is not None:
                image = _resize(image, target[1], target[0])
            prepared.append(image)
        if not prepared:
            height, width = target if target is not None else (pad_to or (0, 0))
            channels = model_channels or (2 if preprocess.channels == "both" else 1)
            batch = np.zeros((0, height or 0, width or 0, channels), dtype=np.float32)
        else:
            shapes = {image.shape for image in prepared}
            if len(shapes) != 1:
                raise ValueError(
                    f"agent {self.name}: inputs of different sizes {sorted(shapes)}; set "
                    "\"preprocess\": {\"pad\": true} or \"resize\": [width, height]"
                )
            batch = np.stack(prepared)
        if model_channels is not None and len(batch) and batch.shape[3] != model_channels:
            raise ValueError(
                f"agent {self.name}: the model expects {model_channels} channel(s), the input has {batch.shape[3]} "
                f"(\"preprocess\": {{\"channels\": ...}}: both = 2 channels, on / off / sum / max = 1 channel)"
            )
        if preprocess.scale != 1.0:
            batch = batch * np.float32(preprocess.scale)
        if preprocess.layout == "nchw":
            batch = batch.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(batch)

    # prediction

    def predict_batch(self, batch: np.ndarray) -> list[np.ndarray]:
        """Runs the model on a prepared batch (in chunks of batch_size), returns the raw outputs."""
        backend = self.backend
        size = self.config.batch_size or len(batch)
        if len(batch) == 0:
            return []
        chunks = [backend.predict(batch[start : start + size]) for start in range(0, len(batch), size)]
        return [np.concatenate([chunk[index] for chunk in chunks]) for index in range(len(chunks[0]))]

    def predict(
        self, data: typing.Union[Frame, typing.Sequence[Tile], np.ndarray]
    ) -> AgentResult:
        """Runs the model on a frame (its tiles or the whole frame, see "input"), a list of
        tiles, or an array (one input of height x width x channels, or a batch)."""
        frame = data if isinstance(data, Frame) else None
        tiles: typing.Optional[list[Tile]] = None
        pad_to = None
        if frame is not None:
            if self.config.input == "tiles":
                if not frame.tiles and frame.tile_rows == 0:
                    raise ValueError(
                        f"agent {self.name} needs tiles: use a frame configuration with \"tiles\" "
                        "(or frame.split(size) first), or set \"input\": \"frame\""
                    )
                tiles = list(frame.tiles)
                size = frame._tile_size or max([tile.width for tile in tiles] + [tile.height for tile in tiles] + [1])
                pad_to = (size, size)
                images = [tile.data for tile in tiles]
            else:
                images = [frame.data]
        elif isinstance(data, np.ndarray):
            images = [data] if data.ndim <= 3 else list(data)
        else:
            tiles = list(data)
            if tiles and not all(isinstance(tile, Tile) for tile in tiles):
                raise TypeError("predict expects a Frame, a list of Tile objects or a numpy array")
            size = max([tile.width for tile in tiles] + [tile.height for tile in tiles] + [1])
            pad_to = (size, size)
            images = [tile.data for tile in tiles]
        batch = self.prepare(images, pad_to)
        start = time.perf_counter()
        outputs = self.predict_batch(batch)
        inference_ms = (time.perf_counter() - start) * 1000
        prediction_config = self.config.prediction
        if outputs and prediction_config.activation != "none":
            first = outputs[0].astype(np.float32)
            first = _softmax(first) if prediction_config.activation == "softmax" else 1 / (1 + np.exp(-first))
            outputs = [first] + outputs[1:]
        context = TaskInput(
            agent=self,
            frame=frame,
            tiles=tiles,
            images=[np.asarray(image) for image in images],
            pad_to=pad_to,
            outputs=outputs,
            count=len(batch),
        )
        predictions = _TASKS[prediction_config.task](context)
        return AgentResult(
            agent=self.name, frame=frame, predictions=predictions, outputs=outputs, inference_ms=inference_ms
        )


# ------------------------------------------------------------------ tasks (output decoders)


@dataclasses.dataclass
class TaskInput:
    """What a task decoder gets: the agent, the frame / tiles, the input images (before
    preprocessing, height x width x channels), the padded size and the model outputs (after the
    activation, first dimension = inputs)."""

    agent: "Agent"
    frame: typing.Optional[Frame]
    tiles: typing.Optional[list[Tile]]
    images: list[np.ndarray]
    pad_to: typing.Optional[tuple[int, int]]
    outputs: list[np.ndarray]
    count: int

    @property
    def config(self) -> PredictionConfig:
        return self.agent.config.prediction

    def tile(self, index: int) -> typing.Optional[Tile]:
        return None if self.tiles is None else self.tiles[index]


def register_task(name: str, decoder=None):
    """Registers a task: decoder(TaskInput) -> list[Prediction] (one per input).
    Usable as a decorator: @register_task("detection")."""

    def wrapper(function):
        _TASKS[name] = function
        return function

    return wrapper if decoder is None else wrapper(decoder)


def list_tasks() -> list[str]:
    return sorted(_TASKS)


@register_task("classification")
def _classification(task: TaskInput) -> list[Prediction]:
    prediction_config = task.config
    outputs = task.outputs
    predictions = []
    for index in range(task.count):
        per_input = [output[index] for output in outputs]
        output = per_input[0] if per_input else np.zeros(0)
        class_id = label = score = None
        values = np.asarray(output).reshape(-1) if np.asarray(output).ndim <= 1 else None
        if values is not None and values.size > 0:
            if values.size == 1:
                score = float(values[0])
                class_id = int(score >= 0.5)
            else:
                class_id = int(np.argmax(values))
                score = float(values[class_id])
            labels = prediction_config.labels
            if labels and class_id is not None and class_id < len(labels):
                label = labels[class_id]
        selected = (
            True
            if prediction_config.threshold is None
            else (score is not None and score >= prediction_config.threshold)
        )
        predictions.append(
            Prediction(
                tile=task.tile(index),
                output=np.asarray(output),
                outputs=per_input,
                class_id=class_id,
                label=label,
                score=score,
                selected=selected,
                labels=prediction_config.labels,
            )
        )
    return predictions


_PIXEL_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("class_id", "u1"), ("score", "<f4")])


@register_task("segmentation")
def _segmentation(task: TaskInput) -> list[Prediction]:
    """One class per pixel: output (height, width, classes) per input, for instance a U-Net."""
    config = task.config
    layout = task.agent.config.preprocess.layout
    predictions = []
    for index in range(task.count):
        per_input = [output[index] for output in task.outputs]
        probabilities = np.asarray(per_input[0], dtype=np.float32)
        if probabilities.ndim == 2:  # one class map channel
            probabilities = probabilities[:, :, None]
        if probabilities.ndim != 3:
            raise ValueError(
                f"agent {task.agent.name}: segmentation needs outputs of height x width x classes, "
                f"got {probabilities.shape}"
            )
        if layout == "nchw":
            probabilities = probabilities.transpose(1, 2, 0)
        image = task.images[index]
        if image.ndim == 2:
            image = image[:, :, None]
        height, width = image.shape[:2]
        # output -> input size (padded tiles: padded size), then crop to the tile
        source_height, source_width = task.pad_to if task.pad_to is not None else (height, width)
        if probabilities.shape[:2] != (source_height, source_width):
            probabilities = _resize(probabilities, source_width, source_height)
        probabilities = probabilities[:height, :width]
        classes = probabilities.shape[2]
        if classes == 1:  # binary segmentation with one sigmoid channel
            probabilities = np.concatenate([1 - probabilities, probabilities], axis=2)
            classes = 2
        class_map = np.argmax(probabilities, axis=2).astype(np.uint8)
        active = np.any(image > config.activity_threshold, axis=2)
        class_counts = np.bincount(class_map.ravel(), minlength=classes).astype(np.int64)
        activity_counts = np.stack(
            [
                np.bincount(class_map[~active], minlength=classes),
                np.bincount(class_map[active], minlength=classes),
            ]
        ).astype(np.int64)

        targets = config.target_classes if config.target_classes is not None else list(range(1, classes))
        target_mask = np.isin(class_map, targets)
        if config.mask_inactive:
            target_mask &= active
        if config.threshold is not None:
            target_mask &= np.take_along_axis(probabilities, class_map[:, :, None].astype(np.intp), axis=2)[:, :, 0] >= config.threshold
        ys, xs = np.nonzero(target_mask)
        tile = task.tile(index)
        offset_x, offset_y = (tile.x, tile.y) if tile is not None else (0, 0)
        pixels = np.zeros(len(ys), _PIXEL_DTYPE)
        pixels["x"] = xs + offset_x
        pixels["y"] = ys + offset_y
        pixels["class_id"] = class_map[ys, xs]
        pixel_scores = probabilities[ys, xs]
        pixels["score"] = pixel_scores[np.arange(len(ys)), pixels["class_id"]] if len(ys) else 0

        class_id = label = score = None
        if len(ys):
            class_id = int(np.bincount(pixels["class_id"]).argmax())
            score = float(pixels["score"][pixels["class_id"] == class_id].mean())
            if config.labels and class_id < len(config.labels):
                label = config.labels[class_id]
        predictions.append(
            Prediction(
                tile=tile,
                output=probabilities,
                outputs=per_input,
                class_id=class_id,
                label=label,
                score=score,
                selected=len(ys) >= max(config.min_pixels, 1),
                class_map=class_map,
                pixels=pixels,
                pixel_scores=pixel_scores,
                class_counts=class_counts,
                activity_counts=activity_counts,
                labels=config.labels,
            )
        )
    return predictions


def as_agents(agents) -> list[Agent]:
    """List of Agent from a path, an Agent, an AgentConfig, a dict or a list of these."""
    if isinstance(agents, (str, pathlib.Path)) and not str(agents).lstrip().startswith("{"):
        return load_agents(agents)
    if isinstance(agents, (Agent, AgentConfig, dict, str)):
        agents = [agents]
    result = [agent if isinstance(agent, Agent) else Agent(agent) for agent in agents]
    names = [agent.name for agent in result]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate agent name(s): {', '.join(duplicates)}")
    return result


# ------------------------------------------------------------------ running agents on a camera or file


@dataclasses.dataclass
class AgentStep:
    """Results of all agents for one time window.

    Attributes:
        window: The time window (window.events: its events).
        frames: {frame configuration name: Frame} built for the agents.
        results: {agent name: AgentResult}.
    """

    window: typing.Any
    frames: dict[str, Frame]
    results: dict[str, AgentResult]

    def __getitem__(self, agent: str) -> AgentResult:
        return self.results[agent]


def _frame_assignments(agents: list[Agent], frame_configs: list[FrameConfig]) -> dict[str, FrameConfig]:
    by_name = {config.name: config for config in frame_configs}
    assignments = {}
    for agent in agents:
        name = agent.config.frame
        if name is None:
            if len(frame_configs) != 1:
                raise ValueError(
                    f"agent {agent.name}: set \"frame\" to one of {', '.join(by_name)} "
                    "(several frame configurations are given)"
                )
            assignments[agent.name] = frame_configs[0]
        elif name not in by_name:
            raise ValueError(
                f"agent {agent.name}: no frame configuration named {name!r} (available: {', '.join(by_name)})"
            )
        else:
            assignments[agent.name] = by_name[name]
    return assignments


def process_window(window, agents, frame_configs=None) -> AgentStep:
    """Builds the frames needed by the agents for one window and runs the agents."""
    agents = as_agents(agents)
    frame_configs = _as_frame_config_list(frame_configs)
    assignments = _frame_assignments(agents, frame_configs)
    return _process(window, agents, assignments)


def _process(window, agents: list[Agent], assignments: dict[str, FrameConfig]) -> AgentStep:
    frames: dict[str, Frame] = {}
    results: dict[str, AgentResult] = {}
    for agent in agents:
        config = assignments[agent.name]
        if config.name not in frames:
            frames[config.name] = window.frame(config)
        results[agent.name] = agent.predict(frames[config.name])
    return AgentStep(window=window, frames=frames, results=results)


def _as_frame_config_list(frame_configs) -> list[FrameConfig]:
    if frame_configs is None:
        return [FrameConfig(name="frame1")]
    if isinstance(frame_configs, (str, pathlib.Path)) and not str(frame_configs).lstrip().startswith("{"):
        from .frames import load_frame_configs

        return load_frame_configs(frame_configs)
    return as_frame_configs(frame_configs)


def run_agents(
    source, agents, frame_configs=None, duration_ms: typing.Optional[float] = None
) -> typing.Iterator[AgentStep]:
    """Runs agents on every time window of a camera or a RAW file.

    source: an open EventCamera or RawReader (anything with windows(duration_ms)).
    agents: list of Agent (load_agents("agents.json")), agent configurations, or a file path.
    frame_configs: frame configurations (load_frame_configs("frames.json")) or a file path.
        Each agent uses the configuration named in its "frame" setting; each frame is built
        once per window, also if several agents use it.
    Yields one AgentStep per window: step.results[agent name], step.frames, step.window.
    """
    agents = as_agents(agents)
    frame_configs = _as_frame_config_list(frame_configs)
    assignments = _frame_assignments(agents, frame_configs)
    for agent in agents:
        agent.load()
    for window in source.windows(duration_ms):
        yield _process(window, agents, assignments)


__all__ = [
    "TaskInput",
    "list_backends",
    "list_tasks",
    "register_backend",
    "register_task",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "Backend",
    "ModelBackend",
    "ModelInput",
    "Prediction",
    "PredictionConfig",
    "PreprocessConfig",
    "TFLiteBackend",
    "TensorFlowBackend",
    "agent_configs_from_data",
    "as_agents",
    "create_backend",
    "load_agent",
    "load_agent_config",
    "load_agent_configs",
    "load_agents",
    "process_window",
    "run_agents",
    "save_agent_configs",
]
