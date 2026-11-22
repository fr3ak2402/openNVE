"""Hailo accelerators (Raspberry Pi AI Kit / AI HAT+: Hailo-8L, Hailo-8) with HailoRT.

Runs compiled Hailo models (.hef, made with the Hailo Dataflow Compiler) through the Python API
of HailoRT (hailo_platform). HailoRT publishes that API only for some Python versions; for the
Python of openNVE build it with tools/build_hailort_python.sh (see docs/accelerators.md).
"""

from __future__ import annotations

import os
import pathlib
import threading
import typing

import numpy as np

from .backends import Backend, ModelInput

if typing.TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig

HAILO_SUFFIXES = (".hef",)
DEFAULT_BATCH_SIZE = 8
TIMEOUT_MS = 10_000

_LOCK = threading.Lock()
_VDEVICE = None  # one virtual device per process, shared by all Hailo agents (HailoRT scheduler)
_VDEVICE_PID = None

INSTALL_HELP = (
    "Hailo models need the HailoRT Python API (hailo_platform) for this Python version: on the "
    "Raspberry Pi install HailoRT (sudo apt install hailo-all) and build the Python API with "
    "tools/build_hailort_python.sh, see docs/accelerators.md#hailo"
)


def import_hailo():
    # HailoRT writes hailort.log into the current directory; HAILORT_LOGGER_PATH set by the user wins
    os.environ.setdefault("HAILORT_LOGGER_PATH", "NONE")
    try:
        import hailo_platform  # type: ignore
    except ImportError as error:
        raise ImportError(f"{INSTALL_HELP} ({error})") from error
    return hailo_platform


def hailo_info() -> dict[str, typing.Any]:
    """HailoRT Python API: installed, version and the Hailo devices it finds (for opennve system).

    Does not open the devices (they may be in use by another program)."""
    import importlib.util

    if importlib.util.find_spec("hailo_platform") is None:
        return {"installed": False}
    try:
        hailo = import_hailo()
    except ImportError as error:
        return {"installed": False, "error": str(error).splitlines()[0]}
    info: dict[str, typing.Any] = {"installed": True, "version": getattr(hailo, "__version__", None)}
    try:
        info["devices"] = [str(device) for device in hailo.Device.scan()]
    except Exception as error:  # driver missing, no device
        info["devices"] = []
        info["error"] = str(error).splitlines()[0] if str(error) else type(error).__name__
    return info


def _vdevice(hailo):
    """The shared virtual device of this process (created on first use)."""
    global _VDEVICE, _VDEVICE_PID
    if _VDEVICE is None or _VDEVICE_PID != os.getpid():
        params = hailo.VDevice.create_params()
        try:  # several models / threads on one device: HailoRT switches between them
            params.scheduling_algorithm = hailo.HailoSchedulingAlgorithm.ROUND_ROBIN
        except AttributeError:
            pass
        _VDEVICE = hailo.VDevice(params)
        _VDEVICE_PID = os.getpid()
    return _VDEVICE


def release_vdevice():
    """Releases the shared Hailo device (after all Hailo agents are closed)."""
    global _VDEVICE
    with _LOCK:
        if _VDEVICE is not None:
            try:
                _VDEVICE.release()
            except Exception:
                pass
            _VDEVICE = None


def _check_device(device) -> None:
    devices = device if isinstance(device, list) else [device]
    if any(item not in (None, "npu", "auto") for item in devices):
        raise ValueError(
            ".hef models run only on a Hailo accelerator: use \"device\": \"npu\" (or leave it out); "
            "for the CPU / GPU use the .tflite, .keras or PyTorch model the .hef was compiled from"
        )


class HailoBackend(Backend):
    """Compiled Hailo model (.hef) on a Hailo accelerator.

    The model is quantized; HailoRT converts the float32 inputs and outputs on the CPU. Inputs
    and outputs are channels last (nhwc). batch_size: frames per Hailo batch (default 8).
    Several Hailo agents and workflow threads share one device; HailoRT schedules them.
    """

    layout = "nhwc"
    device = "npu"
    device_label = "Hailo"  # shown by Agent.device_description(): "NPU Hailo (batch 8)"

    def __init__(self, path, device=None, batch_size: typing.Optional[int] = None, timeout_ms: int = TIMEOUT_MS):
        _check_device(device)
        hailo = import_hailo()
        self._hailo = hailo
        self.path = pathlib.Path(path)
        self.timeout_ms = timeout_ms
        self.batch_size = int(batch_size or DEFAULT_BATCH_SIZE)
        with _LOCK:
            try:
                vdevice = _vdevice(hailo)
            except Exception as error:
                text = str(error)
                if "OUT_OF_PHYSICAL_DEVICES" in text or "NOT_FOUND" in text:
                    raise RuntimeError(
                        "no free Hailo device: is the AI Kit / AI HAT+ connected and its driver loaded "
                        "(hailortcli scan), and no other program using it? " + text
                    ) from error
                raise
            model = vdevice.create_infer_model(os.fspath(self.path))
        if len(model.input_names) != 1:
            raise ValueError(f"{self.path.name}: Hailo models with {len(model.input_names)} inputs are not supported")
        model.set_batch_size(self.batch_size)
        model.input().set_format_type(hailo.FormatType.FLOAT32)  # quantized by HailoRT
        for output in model.outputs:
            output.set_format_type(hailo.FormatType.FLOAT32)  # dequantized by HailoRT
        self._model = model
        self._outputs = [(output.name, tuple(output.shape)) for output in model.outputs]
        self.input = ModelInput(shape=(None,) + tuple(model.input().shape), dtype=np.dtype(np.float32))
        with _LOCK:
            self._configured = model.configure()
        self.device_note = None

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        count = len(batch)
        results = [np.empty((count,) + shape, np.float32) for _, shape in self._outputs]
        for start in range(0, count, self.batch_size):
            bindings = []
            for index in range(start, min(count, start + self.batch_size)):
                binding = self._configured.create_bindings(
                    output_buffers={name: results[number][index] for number, (name, _) in enumerate(self._outputs)}
                )
                binding.input().set_buffer(batch[index])
                bindings.append(binding)
            self._configured.run(bindings, self.timeout_ms)
        # classification heads come as 1 x 1 x classes: give them as (inputs, classes)
        return [result.reshape(count, -1) if result.ndim == 4 and result.shape[1:3] == (1, 1) else result
                for result in results]

    def close(self):
        configured, self._configured = getattr(self, "_configured", None), None
        if configured is not None:
            try:
                configured.shutdown()
            except Exception:
                pass
        self._model = None


def create_hailo_backend(config: "AgentConfig") -> HailoBackend:
    return HailoBackend(config.model_path, config.device, config.batch_size)
