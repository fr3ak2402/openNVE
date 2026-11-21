"""GPU and NPU inference with the LiteRT CompiledModel API (ai-edge-litert 2.4 or newer).

The CompiledModel API runs a .tflite model on an accelerator: the GPU (WebGPU on Windows and
Linux, Metal on macOS) or an NPU through a vendor dispatch library (Intel OpenVINO, Qualcomm,
MediaTek, Google Tensor). Operations the accelerator does not support run on the CPU.

Accelerators need static tensor shapes. Models from Keras usually have a dynamic batch dimension,
and up-sampling layers (Conv2DTranspose) compute their output shape at run time. static_model()
rewrites such a model in memory: fixed batch size, and the computed shapes of TRANSPOSE_CONV,
RESHAPE, RESIZE_*, BROADCAST_TO and FILL replaced by constants; operations whose results are no
longer used are removed. The model file is not changed.
"""

from __future__ import annotations

import os
import typing

import numpy as np

from .backends import Backend, ModelInput

DEVICES = ("cpu", "gpu", "npu", "auto")


def compiled_model_available() -> bool:
    try:
        from ai_edge_litert.compiled_model import CompiledModel  # noqa: F401
        from ai_edge_litert.options import Options  # noqa: F401
    except ImportError:
        return False
    return True


def _require_compiled_model():
    if not compiled_model_available():
        raise ImportError(
            "GPU and NPU agents need LiteRT 2.4 or newer: pip install --upgrade \"ai-edge-litert>=2.4\""
        )


# ------------------------------------------------------------------ static shapes


def _builtin_codes(model) -> list[int]:
    return [max(int(code.builtinCode), int(code.deprecatedBuiltinCode)) for code in model.operatorCodes]


def _constant(model, subgraph, values: typing.Sequence[int], name: bytes) -> int:
    from ai_edge_litert import schema_py_generated as schema

    buffer = schema.BufferT()
    buffer.data = np.asarray(values, dtype=np.int32).view(np.uint8)
    model.buffers.append(buffer)
    tensor = schema.TensorT()
    tensor.shape = np.array([len(values)], dtype=np.int32)
    tensor.type = schema.TensorType.INT32
    tensor.buffer = len(model.buffers) - 1
    tensor.name = name
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def _is_constant(model, tensor) -> bool:
    data = model.buffers[tensor.buffer].data if 0 <= tensor.buffer < len(model.buffers) else None
    return data is not None and len(data) > 0


def _opcode_index(model, code: int, version: int = 1) -> int:
    from ai_edge_litert import schema_py_generated as schema

    for index, operator_code in enumerate(model.operatorCodes):
        if max(int(operator_code.builtinCode), int(operator_code.deprecatedBuiltinCode)) == code:
            operator_code.version = max(int(operator_code.version or 1), version)
            return index
    operator_code = schema.OperatorCodeT()
    operator_code.builtinCode = code
    operator_code.deprecatedBuiltinCode = min(code, 127)
    operator_code.version = version
    model.operatorCodes.append(operator_code)
    return len(model.operatorCodes) - 1


def _valid_transpose_convolutions(model, subgraph, codes):
    """TRANSPOSE_CONV with "SAME" padding -> "VALID" padding + SLICE (the same result).

    The LiteRT GPU accelerator computes "SAME" transposed convolutions wrongly for some sizes
    (for instance 5 x 5 -> 10 x 10 with a 3 x 3 kernel, as in a U-Net on 80 x 80 tiles); with
    "VALID" padding and a crop the results match the CPU."""
    from ai_edge_litert import schema_py_generated as schema

    operators = []
    for operator in subgraph.operators:
        operators.append(operator)
        options = operator.builtinOptions
        if codes[operator.opcodeIndex] != schema.BuiltinOperator.TRANSPOSE_CONV or options is None:
            continue
        if options.padding != schema.Padding.SAME:
            continue
        output = subgraph.tensors[operator.outputs[0]]
        source = subgraph.tensors[operator.inputs[2]]
        kernel = subgraph.tensors[operator.inputs[1]]
        if len(output.shape) != 4 or len(source.shape) != 4 or len(kernel.shape) != 4:
            continue
        batch, height, width, channels = (int(size) for size in output.shape)
        valid_height = (int(source.shape[1]) - 1) * int(options.strideH) + int(kernel.shape[1])
        valid_width = (int(source.shape[2]) - 1) * int(options.strideW) + int(kernel.shape[2])
        if valid_height < height or valid_width < width:
            continue
        options.padding = schema.Padding.VALID
        valid_shape = [batch, valid_height, valid_width, channels]
        inputs = np.array(operator.inputs, dtype=np.int32)
        inputs[0] = _constant(model, subgraph, valid_shape, (output.name or b"") + b"/valid_shape")
        operator.inputs = inputs
        if (valid_height, valid_width) == (height, width):
            continue
        # TensorFlow puts the smaller half of the padding at the top / left
        top, left = (valid_height - height) // 2, (valid_width - width) // 2
        tensor = schema.TensorT()
        tensor.shape = np.array(valid_shape, dtype=np.int32)
        tensor.type = output.type
        tensor.buffer = 0
        tensor.name = (output.name or b"") + b"/valid"
        tensor.quantization = output.quantization
        subgraph.tensors.append(tensor)
        valid_index = len(subgraph.tensors) - 1
        output_index = int(operator.outputs[0])
        operator.outputs = np.array([valid_index], dtype=np.int32)
        name = output.name or b""
        begin = _constant(model, subgraph, [0, top, left, 0], name + b"/crop_begin")
        size = _constant(model, subgraph, [batch, height, width, channels], name + b"/crop_size")
        crop = schema.OperatorT()
        crop.opcodeIndex = _opcode_index(model, schema.BuiltinOperator.SLICE, 2 if output.type == schema.TensorType.INT8 else 1)
        crop.inputs = np.array([valid_index, begin, size], dtype=np.int32)
        crop.outputs = np.array([output_index], dtype=np.int32)
        crop.builtinOptionsType = schema.BuiltinOptions.SliceOptions
        crop.builtinOptions = schema.SliceOptionsT()
        operators.append(crop)
    subgraph.operators = operators
    codes[:] = _builtin_codes(model)


def static_model(data: bytes, batch_size: int = 1, valid_transpose_conv: bool = True) -> bytes:
    """Returns the model (.tflite bytes) with static shapes: dynamic dimensions (-1, usually the
    batch) set to batch_size, computed shape inputs replaced by constants.

    valid_transpose_conv: replace "SAME" transposed convolutions by "VALID" ones and a crop (same
    result; works around wrong GPU results, see _valid_transpose_convolutions)."""
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema

    operators = schema.BuiltinOperator
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    codes = _builtin_codes(model)
    # operator -> (index of the shape input, which output dimensions it holds)
    shape_inputs = {
        operators.TRANSPOSE_CONV: (0, slice(None)),
        operators.RESHAPE: (1, slice(None)),
        operators.BROADCAST_TO: (1, slice(None)),
        operators.FILL: (0, slice(None)),
        operators.RESIZE_BILINEAR: (1, slice(1, 3)),
        operators.RESIZE_NEAREST_NEIGHBOR: (1, slice(1, 3)),
    }
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            signature = tensor.shapeSignature
            if signature is not None and len(signature):
                if any(int(size) < 0 for size in signature):
                    tensor.shape = np.array([batch_size if int(size) < 0 else int(size) for size in signature], dtype=np.int32)
                tensor.shapeSignature = None
        for operator in subgraph.operators:
            code = codes[operator.opcodeIndex]
            if code not in shape_inputs or operator.inputs is None:
                continue
            position, dimensions = shape_inputs[code]
            inputs = np.array(operator.inputs, dtype=np.int32)
            if position >= len(inputs) or inputs[position] < 0:
                continue
            if _is_constant(model, subgraph.tensors[inputs[position]]):
                continue
            output = subgraph.tensors[operator.outputs[0]]
            values = [int(size) for size in np.asarray(output.shape)[dimensions]]
            inputs[position] = _constant(model, subgraph, values, (output.name or b"") + b"/static_shape")
            operator.inputs = inputs
        if valid_transpose_conv:
            _valid_transpose_convolutions(model, subgraph, codes)
        # remove operations whose results are no longer used (SHAPE -> STRIDED_SLICE -> PACK ...)
        while True:
            used = {int(index) for operator in subgraph.operators for index in (operator.inputs if operator.inputs is not None else []) if index >= 0}
            used |= {int(index) for index in subgraph.outputs}
            kept = [operator for operator in subgraph.operators if any(int(index) in used for index in operator.outputs)]
            if len(kept) == len(subgraph.operators):
                break
            subgraph.operators = kept
    builder = flatbuffers.Builder(len(data) + 4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def _quantization(data: bytes) -> dict[int, tuple[float, int]]:
    """Tensor index -> (scale, zero point) of the quantized tensors of the main subgraph."""
    from ai_edge_litert import schema_py_generated as schema

    model = schema.Model.GetRootAsModel(data, 0)
    subgraph = model.Subgraphs(0)
    result = {}
    for index in range(subgraph.TensorsLength()):
        quantization = subgraph.Tensors(index).Quantization()
        if quantization is not None and quantization.ScaleLength() == 1:
            result[index] = (float(quantization.Scale(0)), int(quantization.ZeroPoint(0)) if quantization.ZeroPointLength() else 0)
    return result


# ------------------------------------------------------------------ backend


def _flush_c_streams():
    import ctypes
    import sys

    try:
        if sys.platform == "win32":
            ctypes.cdll.msvcrt.fflush(None)
        else:
            ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


class _QuietStderr:
    """Hides the log lines LiteRT writes to stderr while it creates a model (environment,
    accelerator registration). OPENNVE_LITERT_LOG=1 shows them."""

    _lock = __import__("threading").Lock()

    def __enter__(self):
        import sys

        self._fd = None
        if os.environ.get("OPENNVE_LITERT_LOG"):
            return self
        self._lock.acquire()
        try:
            sys.stderr.flush()
            self._fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            self._fd = None
            self._lock.release()
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            _flush_c_streams()  # LiteRT's log lines may still be in the C buffers
            os.dup2(self._fd, 2)
            os.close(self._fd)
            self._lock.release()
        return False


_EXIT_HOOK = []


def _quiet_exit():
    """LiteRT logs when it releases its accelerators at the end of the program: hide that (after
    Python has printed any error)."""
    if _EXIT_HOOK or os.environ.get("OPENNVE_LITERT_LOG"):
        return
    import atexit

    def hide():
        try:
            import sys

            sys.stderr.flush()
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
        except OSError:
            pass

    atexit.register(hide)
    _EXIT_HOOK.append(hide)


def _explain(device: str, model: str, error: str) -> str:
    """Short reason from a LiteRT error (which contains source locations)."""
    import re

    text = " ".join(re.sub(r"ERROR: \[[^\]]*\]|└", " ", error).split())
    name = device.upper()
    if "not accelerated" in text:
        return f"{name}: not available, or it does not support all operations of {model}"
    if "dynamic" in text.lower():
        return f"{name}: {model} has dynamic shapes the accelerator cannot handle"
    return f"{name}: not available or cannot run {model}" + (f" ({text})" if text else "")


class AcceleratorUnavailable(RuntimeError):
    """The requested accelerator cannot run the model on this machine."""


class CompiledModelBackend(Backend):
    """Runs a .tflite model with the LiteRT CompiledModel API on "gpu", "npu" or "cpu".

    batch_size: inputs per call (the model is compiled for this batch size).
    threads: CPU threads for operations that run on the CPU.
    gpu_precision: "fp16" (default, faster) or "fp32".
    cpu_fallback: False: all operations must run on the accelerator (else AcceleratorUnavailable);
        True: unsupported operations run on the CPU (device "gpu+cpu").
    verify: on a GPU / NPU, compare the results of a test input with the CPU once; if the classes
        differ (accelerator bugs), raise AcceleratorUnavailable.
    """

    def __init__(self, path, device: str, batch_size: int = 1, threads: int = 1, gpu_precision: str = "fp16",
                 cpu_fallback: bool = False, verify: bool = True, data: typing.Optional[bytes] = None):
        with _QuietStderr():
            _require_compiled_model()
            from ai_edge_litert.compiled_model import CompiledModel
            from ai_edge_litert.hardware_accelerator import HardwareAccelerator
            from ai_edge_litert.options import CpuOptions, GpuOptions, Options

        self.batch_size = max(1, int(batch_size))
        if data is None:
            with open(os.fspath(path), "rb") as file:
                data = static_model(file.read(), self.batch_size)
        accelerator = {"cpu": HardwareAccelerator.CPU, "gpu": HardwareAccelerator.GPU, "npu": HardwareAccelerator.NPU}[device]
        if cpu_fallback and device != "cpu":
            accelerator |= HardwareAccelerator.CPU
        options = Options(
            hardware_accelerators=accelerator,
            cpu_options=CpuOptions(num_threads=max(1, int(threads))) if device == "cpu" or cpu_fallback else None,
            gpu_options=GpuOptions(enforce_f32=gpu_precision == "fp32") if device == "gpu" else None,
        )
        if device == "gpu" and "tensorflow" in __import__("sys").modules:
            # TensorFlow and the LiteRT GPU accelerator crash in the same process
            raise AcceleratorUnavailable("GPU: not usable in a program that has loaded TensorFlow (use .tflite models only)")
        _quiet_exit()
        reason = None
        with _QuietStderr():  # also while the error (and the half-created model) is released
            try:
                self.model = CompiledModel.from_buffer(data, options=options)
            except RuntimeError as error:
                reason = _explain(device, os.path.basename(os.fspath(path)), str(error))
        if reason is not None:
            raise AcceleratorUnavailable(reason)
        # with the CPU in the set, is_fully_accelerated() counts the CPU too: the share of the
        # accelerator is only known for cpu_fallback=False (everything on the accelerator)
        self.fully_accelerated = bool(self.model.is_fully_accelerated()) and not (cpu_fallback and device != "cpu")
        self.device = f"{device}+cpu" if cpu_fallback and device != "cpu" else device
        self.threads = max(1, int(threads))
        signatures = self.model.get_signature_list()
        self._signature = next(iter(signatures))
        inputs = self.model.get_input_tensor_details(self._signature)
        if len(inputs) != 1:
            raise ValueError(f"{path}: only models with one input are supported, this one has {len(inputs)}")
        self._input = next(iter(inputs.values()))
        outputs = self.model.get_output_tensor_details(self._signature)
        self._outputs = sorted(outputs.values(), key=lambda detail: detail["index"])
        self._output_positions = [list(outputs.values()).index(detail) for detail in self._outputs]
        quantization = _quantization(data)
        self._input_quantization = quantization.get(int(self._input["index"]))
        self._output_quantization = [quantization.get(int(detail["index"])) for detail in self._outputs]
        self._input_buffers = self.model.create_input_buffers(0)
        self._output_buffers = self.model.create_output_buffers(0)
        shape = tuple(int(size) for size in self._input["shape"])
        self.input = ModelInput(shape=(None,) + shape[1:], dtype=np.dtype(self._input["dtype"]), name=self._input.get("name", ""))
        self._shape = shape
        with _QuietStderr():  # the first run sets up the accelerator (and logs)
            self._run(np.zeros(shape, dtype=np.float32))
        self.verification: typing.Optional[dict[str, typing.Any]] = None
        if verify and device != "cpu":
            self._verify(path, data, device)

    def _verify(self, path, data: bytes, device: str):
        test = np.random.default_rng(12345).random(self._shape).astype(np.float32)
        reference = CompiledModelBackend(path, "cpu", self.batch_size, 1, verify=False, data=data)
        try:
            expected = reference._run(test)[0].astype(np.float32)
        finally:
            reference.close()
        actual = self._run(test)[0].astype(np.float32)
        error = float(np.abs(actual - expected).max() / (np.abs(expected).max() + 1e-6))
        agreement = None
        if expected.ndim >= 2 and expected.shape[-1] > 1:
            agreement = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
        self.verification = {"relative_error": error, "agreement": agreement}
        # fp16 on a GPU: errors up to ~15 % of the largest value occur with large logits
        if error > 0.25 or (agreement is not None and agreement < 0.99):
            model = os.path.basename(os.fspath(path))
            details = f"error {error:.0%}" + (f", {agreement * 100:.1f} % equal classes" if agreement is not None else "")
            self.close()
            raise AcceleratorUnavailable(f"{device.upper()}: results of {model} differ from the CPU ({details})")

    def _run(self, batch: np.ndarray) -> list[np.ndarray]:
        dtype = self.input.dtype
        if np.issubdtype(dtype, np.integer) and self._input_quantization:
            scale, zero_point = self._input_quantization
            info = np.iinfo(dtype)
            batch = np.clip(np.round(batch / scale + zero_point), info.min, info.max)
        self._input_buffers[0].write(np.ascontiguousarray(batch, dtype=dtype).reshape(self._shape))
        self.model.run_by_index(0, self._input_buffers, self._output_buffers)
        results = []
        for detail, position, quantization in zip(self._outputs, self._output_positions, self._output_quantization):
            shape = [int(size) for size in detail["shape"]]
            output_dtype = np.dtype(detail["dtype"])
            output = self._output_buffers[position].read(int(np.prod(shape)), output_dtype).reshape(shape)
            if np.issubdtype(output_dtype, np.integer) and quantization:
                scale, zero_point = quantization
                output = (output.astype(np.float32) - zero_point) * scale
            results.append(np.array(output))
        return results

    def predict(self, batch: np.ndarray) -> list[np.ndarray]:
        count = len(batch)
        if count == 0:
            return []
        size = self.batch_size
        results: list[list[np.ndarray]] = []
        for start in range(0, count, size):
            chunk = batch[start : start + size]
            if len(chunk) < size:
                chunk = np.concatenate([chunk, np.zeros((size - len(chunk),) + chunk.shape[1:], chunk.dtype)])
            outputs = self._run(chunk)
            results.append([output[: min(size, count - start)] for output in outputs])
        return [np.concatenate([result[index] for result in results]) for index in range(len(results[0]))]

    def close(self):
        import gc

        with _QuietStderr():  # LiteRT logs when it releases the accelerator
            model, self.model = self.model, None
            for buffer in list(self._input_buffers) + list(self._output_buffers):
                try:
                    buffer.destroy()
                except Exception:
                    pass
            self._input_buffers = self._output_buffers = []
            if model is not None:
                environment = model.environment
                model.close()
                environment.close()
            del model
            gc.collect()

    def __del__(self):
        if getattr(self, "model", None) is not None:
            try:
                self.close()
            except Exception:
                pass


# ------------------------------------------------------------------ what this machine offers


def litert_accelerators() -> dict[str, typing.Any]:
    """Accelerator libraries of the installed LiteRT: {"version", "compiled_model", "gpu", "npu"}.
    gpu: "WebGPU", "Metal", "OpenCL" or None; npu: the vendor dispatch libraries of this platform (Intel
    OpenVINO, Qualcomm, MediaTek, Google Tensor): {"name", "sdk_installed", "install"} (the
    runtime SDK of the vendor is a separate package, "install" names it)."""
    import importlib.metadata
    import importlib.util
    import pathlib

    try:
        version = importlib.metadata.version("ai-edge-litert")
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "compiled_model": False, "gpu": None, "npu": []}
    spec = importlib.util.find_spec("ai_edge_litert")
    root = pathlib.Path(spec.origin).parent if spec and spec.origin else None
    gpu = None
    npu = []
    if root is not None:
        names = {path.name for path in root.iterdir()}
        for library, name in (("Metal", "Metal"), ("WebGpu", "WebGPU"), ("GpuAccelerator", "OpenCL")):
            if any(f"LiteRt{library}" in item for item in names):
                gpu = name
                break
        vendors = root / "vendors"
        sdks = {"intel_openvino": ("Intel OpenVINO", "ai_edge_litert_sdk_intel", "opennve[npu-intel]"),
                "qualcomm": ("Qualcomm", "ai_edge_litert_sdk_qualcomm", "opennve[npu-qualcomm]"),
                "mediatek": ("MediaTek", "ai_edge_litert_sdk_mediatek", "ai-edge-litert-sdk-mediatek"),
                "google_tensor": ("Google Tensor", None, None)}
        for folder, (name, module, install) in sdks.items():
            dispatch = vendors / folder / "dispatch"
            if dispatch.is_dir() and any(dispatch.iterdir()):
                installed = module is None or importlib.util.find_spec(module) is not None
                npu.append({"name": name, "sdk_installed": installed, "install": install})
    return {"version": version, "compiled_model": compiled_model_available(), "gpu": gpu, "npu": npu}
