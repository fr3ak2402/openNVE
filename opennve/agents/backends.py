"""Model runtimes: TensorFlow Lite (LiteRT), TensorFlow / Keras, Python model objects."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import typing

import numpy as np

from .config import AgentConfig, PathLike
from .registry import _BACKENDS, register_backend



@dataclasses.dataclass
class ModelInput:
    """Input of a model: shape (None = any size) and dtype."""

    shape: tuple[typing.Optional[int], ...]
    dtype: np.dtype
    name: str = ""


class Backend:
    """Runs a model on a batch: predict(batch) -> list of output arrays (first dimension = batch).

    device: where the model runs ("cpu", "gpu", "npu", "gpu+cpu" / "npu+cpu" if some operations
    run on the CPU, "delegate"); device_note: why another device than the requested one is used.
    """

    input: ModelInput
    device: str = "cpu"
    device_note: typing.Optional[str] = None

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


def _load_delegate(delegate):
    """Loads a TensorFlow Lite delegate library ({"library": ..., "options": {...}} or a path)."""
    library = delegate["library"] if isinstance(delegate, dict) else delegate
    options = {str(key): str(value) for key, value in (delegate.get("options") or {}).items()} if isinstance(delegate, dict) else {}
    try:
        from ai_edge_litert.interpreter import load_delegate
    except ImportError:
        try:
            from tflite_runtime.interpreter import load_delegate  # type: ignore
        except ImportError:
            load_delegate = _import_tensorflow().lite.experimental.load_delegate
    try:
        return load_delegate(library, options)
    except (ValueError, OSError, RuntimeError) as error:
        raise RuntimeError(f"cannot load the TensorFlow Lite delegate {library!r}: {error}") from error


class TFLiteBackend(Backend):
    """TensorFlow Lite model (.tflite), run with LiteRT, tflite_runtime or tf.lite.

    Quantized models (int8 / uint8 inputs and outputs) are quantized and dequantized automatically.
    Models with a fixed batch size are called several times, models with a dynamic batch
    dimension are resized to the batch.
    """

    def __init__(self, path: PathLike, threads: typing.Union[int, str, None] = None, delegate=None):
        from ..system import is_auto, standalone_interpreter_threads

        Interpreter = _tflite_interpreter_class()
        self.threads = standalone_interpreter_threads() if is_auto(threads) else int(threads)
        kwargs = {"model_path": os.fspath(path), "num_threads": self.threads}
        if delegate is not None:
            kwargs["experimental_delegates"] = [_load_delegate(delegate)]
            self.device = "delegate"
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
    from ..networks import load_keras_model

    return load_keras_model(path, safe_mode=safe_mode)


class TensorFlowBackend(Backend):
    """TensorFlow model: Keras files (.keras, .h5) run with model.predict (or model(x) with
    call="call"), SavedModel directories run with their serving signature."""

    def __init__(
        self, path: PathLike, call: str = "predict", signature: str = "serving_default", safe_mode: bool = True,
        device: typing.Optional[str] = None, fallback: bool = True,
    ):
        path = pathlib.Path(path)
        self.call = call
        self._device_name = None
        if device is not None:
            tf = _import_tensorflow()
            gpus = tf.config.list_physical_devices("GPU")
            wanted = "gpu" if device == "auto" and gpus else ("cpu" if device == "auto" else device)
            if wanted == "npu":
                reason = "TensorFlow models cannot run on an NPU (convert the model to .tflite)"
                if not fallback:
                    raise RuntimeError(reason)
                wanted, self.device_note = "cpu", reason
            if wanted == "gpu" and not gpus:
                reason = "TensorFlow finds no GPU (CUDA on Linux, tensorflow-metal on macOS)"
                if not fallback:
                    raise RuntimeError(reason)
                wanted, self.device_note = "cpu", reason
            self.device = wanted
            self._device_name = "/GPU:0" if wanted == "gpu" else "/CPU:0"
        else:
            tf = None
            self.device = "gpu" if _tensorflow_has_gpu() else "cpu"
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
        if self._device_name is None:
            return self._predict(batch)
        with _import_tensorflow().device(self._device_name):
            return self._predict(batch)

    def _predict(self, batch: np.ndarray) -> list[np.ndarray]:
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


def _tensorflow_has_gpu() -> bool:
    try:
        return bool(_import_tensorflow().config.list_physical_devices("GPU"))
    except Exception:
        return False


def _accelerator_present(kind: str) -> bool:
    """Is there a GPU / NPU that LiteRT can use? (for "device": "auto")"""
    import platform
    import sys

    from ..system import accelerators
    from ._litert import litert_accelerators

    if kind == "npu":  # an NPU and the runtime of a vendor (the Apple Neural Engine has none)
        return any(device["type"] == "npu" for device in accelerators()) and any(
            npu["sdk_installed"] for npu in litert_accelerators()["npu"]
        )
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return True  # Metal
    return any(device["type"] == "gpu" for device in accelerators())


def create_tflite_backend(config: AgentConfig) -> Backend:
    """TensorFlow Lite backend for the device of the configuration (see AgentConfig.device)."""
    from ..system import is_auto, standalone_interpreter_threads
    from ._litert import AcceleratorUnavailable, CompiledModelBackend

    path = config.model_path
    if config.delegate is not None:
        return TFLiteBackend(path, config.threads, config.delegate)
    device = config.device or "cpu"
    if isinstance(device, list):  # the first device of the list that works
        candidates = device[: device.index("cpu")] if "cpu" in device else list(device)
        device = "list"
        if not candidates:
            return TFLiteBackend(path, config.threads)
    elif device == "cpu":
        return TFLiteBackend(path, config.threads)
    elif device == "auto":  # only devices the operating system reports
        candidates = [kind for kind in ("npu", "gpu") if _accelerator_present(kind)]
    else:
        candidates = [device]
    threads = standalone_interpreter_threads() if is_auto(config.threads) else int(config.threads)
    reasons = []
    for kind in candidates:
        batch_size = config.batch_size or (8 if kind == "gpu" else 1)
        # first everything on the accelerator; a GPU may also share the model with the CPU (not
        # with "auto"). An NPU is only used for the whole model: with the CPU in the set, LiteRT
        # cannot tell whether the NPU runs anything at all.
        attempts = [False, True] if kind == "gpu" and device != "auto" else [False]
        for cpu_fallback in attempts:
            try:
                return CompiledModelBackend(path, kind, batch_size, threads, config.gpu_precision, cpu_fallback)
            except AcceleratorUnavailable as error:
                reasons.append(str(error))
            except ImportError as error:
                reasons.append(str(error))
                break
    if device == "auto" and not candidates:
        reasons.append("no GPU or NPU found")
    if not config.fallback and device != "auto" and not (isinstance(config.device, list) and "cpu" in config.device):
        raise RuntimeError(f"agent {config.name}: " + "; ".join(dict.fromkeys(reasons)))
    backend = TFLiteBackend(path, config.threads)
    backend.device_note = "; ".join(dict.fromkeys(reasons)) or None
    return backend


register_backend("tflite", create_tflite_backend, [".tflite"])
register_backend(
    "tensorflow",
    lambda config: TensorFlowBackend(config.model_path, config.call, config.signature, config.safe_mode,
                                     config.device[0] if isinstance(config.device, list) else config.device,
                                     config.fallback),
    [".keras", ".h5", ".hdf5"],
)


def create_backend(config: AgentConfig) -> Backend:
    """Loads the model of an agent configuration with its backend."""
    path = config.model_path
    if not path.exists():
        raise FileNotFoundError(f"agent {config.name}: model not found: {path}")
    return _BACKENDS[config.resolved_backend](config)
