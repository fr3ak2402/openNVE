"""Agent configurations: AgentConfig, PreprocessConfig, PredictionConfig and agent configuration files."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import typing

from .registry import _BACKEND_SUFFIXES, _BACKENDS, _TASKS, list_tasks

PathLike = typing.Union[str, pathlib.Path]

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
    "name", "model", "backend", "device", "fallback", "gpu_precision", "delegate", "input", "frame", "batch_size",
    "threads", "call", "signature", "safe_mode", "preprocess", "prediction",
)
DEVICES = ("cpu", "gpu", "npu", "auto")
GPU_PRECISIONS = ("fp16", "fp32")


@dataclasses.dataclass
class AgentConfig:
    """Configuration of one agent (see the module documentation for the JSON format).

    model: path of the model (.tflite, .keras, .h5 or a SavedModel directory). Relative paths
        are relative to the agent configuration file (base_dir).
    backend: "auto", "tflite" or "tensorflow".
    device: where the model runs: "cpu" (default), "gpu", "npu" or "auto" (NPU, else GPU, else
        CPU). TensorFlow Lite models use the LiteRT CompiledModel API for GPU and NPU (see
        docs/accelerators.md); TensorFlow models are placed on the GPU with tf.device.
    fallback: if the device is not available or cannot run the model: true (default) runs it on
        the CPU (with a message), false raises an error.
    gpu_precision: "fp16" (default, faster) or "fp32" (GPU only).
    delegate: TensorFlow Lite delegate library for other accelerators (for instance the Coral
        Edge TPU): {"library": "libedgetpu.so.1", "options": {...}} or just the library path.
    input: "tiles" (one model input per tile) or "frame" (the whole frame).
    frame: name of the frame configuration that builds the input (see run_agents), or None.
    batch_size: inputs per model call (None: all tiles of a frame at once).
    threads: CPU threads of the TensorFlow Lite interpreter: a number, or "auto" / null (default):
        sized to the hardware (opennve.cpu_count()): all cores up to 4 for a single agent; in
        workflows the agent step splits the cores between its threads (see docs/performance.md).
    call: TensorFlow (Keras) models: "predict" (model.predict) or "call" (model(x), lower latency
        for small batches).
    signature: SavedModel signature (default "serving_default").
    """

    model: str
    name: typing.Optional[str] = None
    backend: str = "auto"
    device: typing.Optional[str] = None
    fallback: bool = True
    gpu_precision: str = "fp16"
    delegate: typing.Union[str, dict, None] = None
    input: str = "tiles"
    frame: typing.Optional[str] = None
    batch_size: typing.Optional[int] = None
    threads: typing.Union[int, str, None] = None
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
        if self.device is not None and self.device not in DEVICES:
            raise ValueError(f"agent.device must be one of {', '.join(DEVICES)} or null")
        if not isinstance(self.fallback, bool):
            raise ValueError("agent.fallback must be true or false")
        if self.gpu_precision not in GPU_PRECISIONS:
            raise ValueError(f"agent.gpu_precision must be one of {', '.join(GPU_PRECISIONS)}")
        if self.delegate is not None:
            library = self.delegate.get("library") if isinstance(self.delegate, dict) else self.delegate
            if not isinstance(library, str) or not library:
                raise ValueError("agent.delegate must be a library path or {\"library\": ..., \"options\": {...}}")
            if isinstance(self.delegate, dict) and (
                set(self.delegate) - {"library", "options"} or not isinstance(self.delegate.get("options", {}), dict)
            ):
                raise ValueError("agent.delegate supports \"library\" and \"options\" (an object)")
        if not isinstance(self.safe_mode, bool):
            raise ValueError("agent.safe_mode must be true or false")
        if self.input not in INPUTS:
            raise ValueError(f"agent.input must be one of {', '.join(INPUTS)}")
        if self.frame is not None and (not isinstance(self.frame, str) or not self.frame):
            raise ValueError("agent.frame must be the name of a frame configuration or null")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ValueError("agent.batch_size must be a positive integer or null")
        if self.threads is not None and self.threads != "auto" and (
            isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads <= 0
        ):
            raise ValueError("agent.threads must be a positive integer, \"auto\" or null")
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
            "device": self.device,
            "fallback": self.fallback,
            "gpu_precision": self.gpu_precision,
            "delegate": self.delegate,
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


