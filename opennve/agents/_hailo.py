"""Hailo accelerators: Hailo-8 / Hailo-8L (HailoRT 4.x; Raspberry Pi AI Kit, AI HAT+), Hailo-10H
(HailoRT 5.x; AI HAT+ 2) and Hailo-15 (HailoRT 5.x on the Hailo-15 itself).

Runs compiled Hailo models (.hef, made with the Hailo Dataflow Compiler for the chip) with
openNVE-hailo (package opennve_hailo, https://github.com/fr3ak2402/openNVE-hailo): HailoRT in
Rust with a Python module, compiled against the HailoRT installed on the computer (its
install.sh), for every Python from 3.11.
"""

from __future__ import annotations

import importlib.util
import pathlib
import typing

import numpy as np

from .backends import Backend, ModelInput

if typing.TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig

HAILO_SUFFIXES = (".hef",)
DEFAULT_BATCH_SIZE = 8
TIMEOUT_MS = 10_000

INSTALL_HELP = (
    "Hailo models need openNVE-hailo (package opennve_hailo): run opennve hailo install (sets up PCIe, "
    "driver, HailoRT and opennve_hailo on the Raspberry Pi; see docs/accelerators.md#hailo)"
)


def import_hailo():
    try:
        import opennve_hailo  # type: ignore
    except ImportError as error:
        raise ImportError(f"{INSTALL_HELP} ({error})") from error
    return opennve_hailo


def hailo_info(identify: bool = True) -> dict[str, typing.Any]:
    """openNVE-hailo and HailoRT: {"installed", "version" (HailoRT), "opennve_hailo",
    "devices": [{"id", "architecture"}]} (for opennve system).

    identify: ask each device for its chip (opens the device briefly; None if another program
    uses it)."""
    if importlib.util.find_spec("opennve_hailo") is None:
        return {"installed": False}
    try:
        hailo = import_hailo()
        info = hailo.info(identify)
    except Exception as error:  # libhailort missing or of another version than at build time
        return {"installed": False, "error": str(error).splitlines()[0] if str(error) else type(error).__name__}
    return {"installed": True, "version": info["hailort"], "opennve_hailo": info["version"], "devices": info["devices"]}


def _check_device(device) -> None:
    devices = device if isinstance(device, list) else [device]
    if any(item not in (None, "npu", "auto") for item in devices):
        raise ValueError(
            ".hef models run only on a Hailo accelerator: use \"device\": \"npu\" (or leave it out); "
            "for the CPU / GPU use the .tflite, .keras or PyTorch model the .hef was compiled from"
        )


class HailoBackend(Backend):
    """Compiled Hailo model (.hef) on a Hailo accelerator (Hailo-8, Hailo-8L, Hailo-10H, Hailo-15).

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
        self.path = pathlib.Path(path)
        self.batch_size = int(batch_size or DEFAULT_BATCH_SIZE)
        if not self.path.is_file():
            raise FileNotFoundError(f"model not found: {self.path}")
        self._model = hailo.Model(self.path, batch_size=self.batch_size, timeout_ms=timeout_ms)
        self.input = ModelInput(shape=(None,) + tuple(self._model.input.shape), dtype=np.dtype(np.float32))
        self.device_note = None

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        outputs = self._model.run(np.ascontiguousarray(batch, dtype=np.float32))
        # classification heads come as 1 x 1 x classes: give them as (inputs, classes)
        return [output.reshape(len(output), -1) if output.ndim == 4 and output.shape[1:3] == (1, 1) else output
                for output in outputs]

    def close(self):
        model, self._model = getattr(self, "_model", None), None
        if model is not None:
            try:
                model.close()
            except Exception:
                pass


def create_hailo_backend(config: "AgentConfig") -> HailoBackend:
    return HailoBackend(config.model_path, config.device, config.batch_size)
