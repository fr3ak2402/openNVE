"""PyTorch models: TorchScript (.pt), torch.export (.pt2), complete pickled models and nn.Module objects."""

from __future__ import annotations

import os
import pathlib
import threading
import typing
import warnings

import numpy as np

from .backends import Backend, ModelInput

if typing.TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig

TORCH_SUFFIXES = (".pt", ".pth", ".pt2")
_LOAD_LOCK = threading.Lock()  # torch.export.load is not thread-safe (workflows load in each thread)


def import_torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise ImportError(
            "PyTorch models need PyTorch: pip install \"opennve[pytorch]\" "
            "(or pip install torch, see https://pytorch.org/get-started for CUDA builds)"
        ) from error
    return torch


def torch_gpu() -> typing.Optional[str]:
    """The GPU device type PyTorch can use: "cuda" (NVIDIA, AMD ROCm), "mps" (Apple), "xpu"
    (Intel) or None."""
    try:
        torch = import_torch()
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return None


def torch_info() -> dict[str, typing.Any]:
    """PyTorch version and GPU (for opennve system / version)."""
    try:
        torch = import_torch()
    except ImportError:
        return {"installed": False}
    gpu = torch_gpu()
    info: dict[str, typing.Any] = {"installed": True, "version": torch.__version__, "gpu": gpu}
    if gpu == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def is_torch_module(model) -> bool:
    """Is the object a torch.nn.Module? (without importing PyTorch for other objects)"""
    if not any(cls.__module__.split(".")[0] == "torch" for cls in type(model).__mro__):
        return False
    return isinstance(model, import_torch().nn.Module)


def _load(path: pathlib.Path, safe_mode: bool):
    """Loads a model file -> (module, input shape or None, dtype or None)."""
    torch = import_torch()
    if path.suffix.lower() == ".pt2":
        program = torch.export.load(os.fspath(path))
        shape, dtype = _exported_input(program)
        return program.module(), shape, dtype
    try:  # TorchScript (torch.jit.script / trace + save; deprecated in PyTorch, still loadable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.jit.load(os.fspath(path), map_location="cpu"), None, None
    except Exception as jit_error:  # not a TorchScript archive
        jit_error_text = str(jit_error).splitlines()[0] if str(jit_error) else type(jit_error).__name__
    try:
        loaded = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    except Exception as error:
        if safe_mode:
            raise ValueError(
                f"{path.name}: not a TorchScript or torch.export model, and torch.load with weights_only "
                f"refuses it ({str(error).splitlines()[0]}). If it is a complete model saved with "
                "torch.save(model) and you trust the file, set \"safe_mode\": false (runs code from the file)"
            ) from error
        loaded = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        return loaded, None, None
    if isinstance(loaded, dict):
        raise ValueError(
            f"{path.name} only holds weights (a state_dict), not the network. Save the model with "
            "torch.export (torch.export.save(torch.export.export(model, (example,)), \"model.pt2\")) or "
            "TorchScript (torch.jit.script(model).save(\"model.pt\")), or pass the network in Python: "
            f"Agent.from_model(model) after model.load_state_dict(...) (TorchScript: {jit_error_text})"
        )
    raise ValueError(f"{path.name}: unsupported content {type(loaded).__name__}")


def _exported_input(program) -> tuple[typing.Optional[tuple], typing.Optional[np.dtype]]:
    """Input shape (None for dynamic sizes) and dtype of a torch.export program with one input."""
    try:
        names = [spec.arg.name for spec in program.graph_signature.input_specs
                 if spec.kind.name == "USER_INPUT"]
        nodes = {node.name: node for node in program.graph.nodes if node.op == "placeholder"}
        if len(names) != 1:
            return None, None
        value = nodes[names[0]].meta["val"]
        shape = tuple(size if isinstance(size, int) else None for size in value.shape)
        dtype = np.dtype(str(value.dtype).replace("torch.", ""))
        return shape, dtype
    except Exception:
        return None, None


class TorchBackend(Backend):
    """PyTorch model on the CPU or a GPU (CUDA, Apple MPS, Intel XPU).

    model: a file (.pt2 from torch.export, .pt / .pth TorchScript, or a complete model saved with
        torch.save(model): only with safe_mode=False, because unpickling runs code from the file)
        or an nn.Module.
    device: "cpu", "gpu", "auto" (GPU if PyTorch finds one) or None (CPU). PyTorch has no NPU
        support: "npu" falls back to the CPU (or raises with fallback=False).
    gpu_precision: "fp16" runs the model with autocast float16 on the GPU, "fp32" in float32.
    threads: CPU threads (torch.set_num_threads in the thread that runs the model), "auto" / None:
        all cores (in workflows the agent step splits the cores between its threads).

    PyTorch models take channels first: the agent's layout "auto" (default) is "nchw".
    """

    layout = "nchw"

    def __init__(
        self, model, threads: typing.Union[int, str, None] = None, device: typing.Union[str, list, None] = None,
        fallback: bool = True, gpu_precision: str = "fp16", safe_mode: bool = True,
        batch_size: typing.Optional[int] = None, input_shape: typing.Optional[tuple] = None,
    ):
        from ..system import cpu_count, is_auto

        torch = import_torch()
        self._torch = torch
        shape = dtype = None
        if isinstance(model, (str, os.PathLike)):
            with _LOAD_LOCK:
                self.module, shape, dtype = _load(pathlib.Path(model), safe_mode)
        elif isinstance(model, torch.nn.Module):
            self.module = model
        else:
            raise TypeError("TorchBackend needs a model file or a torch.nn.Module")
        if input_shape is not None:
            shape = tuple(input_shape)
        if hasattr(self.module, "eval"):
            try:
                self.module.eval()
            except Exception:  # exported programs do not support eval() in every version
                pass
        self.input = ModelInput(shape=shape or (None, None, None, None), dtype=dtype or np.dtype(np.float32))
        self._fixed_batch = shape[0] if shape and isinstance(shape[0], int) else None

        self.threads = max(1, cpu_count() if is_auto(threads) else int(threads))

        self._device, self.device, self.device_note = self._choose_device(device, fallback)
        if self._device != "cpu":
            self.module.to(self._device)
            self.batch_size = batch_size or 8
        # channels last in memory: convolutions about 30 % faster on the CPU (oneDNN) and on GPUs
        self._channels_last = False
        try:
            self.module.to(memory_format=torch.channels_last)
            self._channels_last = True
        except Exception:
            pass
        self._autocast = self._device != "cpu" and gpu_precision == "fp16"

    def _choose_device(self, device, fallback: bool) -> tuple[str, str, typing.Optional[str]]:
        wanted = device or "cpu"
        if isinstance(wanted, list):  # the first device of the list that works
            for candidate in wanted:
                if candidate == "cpu" or (candidate == "gpu" and torch_gpu()):
                    wanted = candidate
                    break
            else:
                wanted = wanted[0]
        if wanted == "auto":
            wanted = "gpu" if torch_gpu() else "cpu"
        if wanted == "cpu":
            return "cpu", "cpu", None
        if wanted == "npu":
            reason = "PyTorch models cannot run on an NPU (convert the model to .tflite)"
        else:
            gpu = torch_gpu()
            if gpu:
                return gpu, "gpu", None
            reason = "PyTorch finds no GPU (CUDA, ROCm, Apple MPS or Intel XPU)"
        if not fallback:
            raise RuntimeError(reason)
        return "cpu", "cpu", reason

    def _run(self, batch: np.ndarray) -> list[np.ndarray]:
        torch = self._torch
        if self._device == "cpu" and torch.get_num_threads() != self.threads:
            torch.set_num_threads(self.threads)  # per calling thread (OpenMP), workflows run several
        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=self.input.dtype))
        if self._device != "cpu":
            tensor = tensor.to(self._device)
        if self._channels_last and tensor.dim() == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            if self._autocast:
                with torch.autocast(device_type=self._device, dtype=torch.float16):
                    outputs = self.module(tensor)
            else:
                outputs = self.module(tensor)
        return _to_numpy(outputs)

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        count = len(batch)
        if count == 0:
            return []
        size = self._fixed_batch
        if not size or size == count:
            return self._run(batch)
        # exported with a fixed batch size: run in chunks, pad the last chunk
        results = []
        for start in range(0, count, size):
            chunk = batch[start : start + size]
            if len(chunk) < size:
                chunk = np.concatenate([chunk, np.zeros((size - len(chunk),) + chunk.shape[1:], chunk.dtype)])
            results.append([output[: min(size, count - start)] for output in self._run(chunk)])
        return [np.concatenate([result[index] for result in results]) for index in range(len(results[0]))]

    def close(self):
        self.module = None


def _to_numpy(outputs) -> list[np.ndarray]:
    if isinstance(outputs, dict):
        return [_tensor(outputs[name]) for name in sorted(outputs)]
    if isinstance(outputs, (list, tuple)):
        return [_tensor(output) for output in outputs]
    return [_tensor(outputs)]


def _tensor(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if value.is_floating_point():
            value = value.float()  # float16 / bfloat16 (autocast) -> float32
        return value.cpu().numpy()
    return np.asarray(value)


def create_torch_backend(config: "AgentConfig") -> TorchBackend:
    return TorchBackend(
        config.model_path, config.threads, config.device, config.fallback, config.gpu_precision,
        config.safe_mode, config.batch_size,
    )
