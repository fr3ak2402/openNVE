"""int8 quantization: converts an agent's TensorFlow / Keras model to a TensorFlow Lite model with
8 bit weights and activations, calibrated with frames from recordings.

The calibration data are the inputs the agent really gets: the recordings (HDF5 event files with
the training data, or .raw files) are cut into time windows, the agent's frame configuration
builds the frames and tiles, and the agent's preprocessing prepares them. A part of the inputs is
kept back to compare the quantized model with the original one (agreement of the classes).

    report = opennve.quantize_agent(config, ["data/train.h5"], frame_config, "models/unet_int8.tflite")
    print(opennve.quantize_report(report))

On the command line: opennve quantize agents.json --agent prototyp --data data/*.h5 --frames frames.json

Needs TensorFlow (pip install "opennve[tensorflow]"). The quantized .tflite runs with LiteRT like
any other .tflite (CPU, GPU, NPU); the inputs and outputs stay float32 unless int8_io is set (the
model then quantizes the inputs itself, which saves a conversion on accelerators that only take
int8).
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
import typing

import numpy as np

from .agent import Agent
from .config import AgentConfig

PathLike = typing.Union[str, os.PathLike]

DEFAULT_SAMPLES = 500
DEFAULT_VALIDATION = 200
DEFAULT_STRIDE = 5
DEFAULT_WINDOW_MS = 40.0
DEFAULT_EPOCHS = 5
DEFAULT_QAT_SAMPLES = 20000
DEFAULT_QAT_VALIDATION = 1000


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "quantization needs TensorFlow: pip install \"opennve[tensorflow]\" (Python 3.14: tf-nightly)"
        ) from error
    return tf


def _images(config: AgentConfig, frame) -> list[np.ndarray]:
    if config.input == "tiles":
        return [tile.data for tile in frame.tiles]
    return [frame.data]


PROGRESS_SECONDS = 10.0


def _share_read(recording) -> float | None:
    """How much of a recording has been read (0..1), if the reader knows it."""
    try:
        total = len(recording.source)
        return min(1.0, recording._position / total) if total else None
    except (AttributeError, TypeError):
        return None


def _windows(config: AgentConfig, recordings, frame_config, window_ms: float, stride: int,
             max_windows: int | None, stats: dict,
             progress: typing.Callable[[str], None] | None = None) -> typing.Iterator[list[np.ndarray]]:
    """The model inputs (tiles or frame) of every stride-th time window of the recordings. Reports
    every PROGRESS_SECONDS how far it is."""
    from ..recordings import open_recording

    used = 0
    for number, path in enumerate(recordings, 1):
        stats["files"] += 1
        name = os.path.basename(os.fspath(path))
        if progress:
            progress(f"opening {name} ({number}/{len(recordings)}; without width / height in the file its size is "
                     "measured first, one pass over the events)")
        last = time.perf_counter()
        with open_recording(path, window={"duration_ms": window_ms}) as recording:
            for index, window in enumerate(recording.windows()):
                stats["windows"] += 1
                if progress and time.perf_counter() - last >= PROGRESS_SECONDS:
                    last = time.perf_counter()
                    share = _share_read(recording)
                    progress(f"  {name}: {window.end_t / 1e6:,.0f} s of the recording"
                             + (f" ({share * 100:.0f} %)" if share is not None else "")
                             + f", {stats['windows']} windows, {stats['candidates']} inputs found"
                             + (f", {used} of --max-windows {max_windows} used" if max_windows else ""))
                if index % max(1, stride):
                    continue
                if max_windows is not None and used >= max_windows:
                    return
                used += 1
                images = _images(config, window.frame(frame_config))
                stats["candidates"] += len(images)
                if images:
                    yield images


PER_WINDOW = 16  # at most this many tiles of one random window (more different windows)


def _is_hdf5(path) -> bool:
    from ..recordings.hdf5 import HDF5_SUFFIXES, is_hdf5

    path = pathlib.Path(path)
    return path.suffix.lower() in HDF5_SUFFIXES or (path.suffix.lower() != ".raw" and is_hdf5(path))


def _random_windows(config: AgentConfig, recordings, frame_config, window_ms: float, max_windows: int | None,
                    stats: dict, seed: int, width: int | None = None, height: int | None = None,
                    progress: typing.Callable[[str], None] | None = None) -> typing.Iterator[list[np.ndarray]]:
    """Model inputs of time windows at random places of HDF5 recordings, read directly there
    (random access: a 300 GB file needs only the windows that are used, not a pass over all of it).
    The files are chosen in proportion to their events; of one window at most PER_WINDOW tiles are
    used. Runs until the caller stops (or max_windows)."""
    from ..recordings.hdf5 import open_hdf5

    rng = np.random.default_rng(seed)
    readers = []
    try:
        for path in recordings:
            name = os.path.basename(os.fspath(path))
            if progress:
                progress(f"opening {name}")
            reader = open_hdf5(path, window={"duration_ms": window_ms}, width=width, height=height, estimate_size=True)
            readers.append(reader)
            stats["files"] += 1
            if progress:
                size = f"{reader.width} x {reader.height}"
                note = (" (estimated from samples: not in the file; give --width / --height if it is wrong)"
                        if reader.size_estimated else "")
                progress(f"  {name}: {reader.event_count:,} events, {size}{note}")
        counts = np.array([reader.event_count for reader in readers], np.float64)
        if not counts.sum():
            return
        weights = counts / counts.sum()
        last = time.perf_counter()
        empty = 0
        while max_windows is None or stats["windows"] < max_windows:
            reader = readers[int(rng.choice(len(readers), p=weights))]
            window = reader.window_at(int(rng.integers(0, reader.event_count)), window_ms)
            stats["windows"] += 1
            images = _images(config, window.frame(frame_config))
            stats["candidates"] += len(images)
            if len(images) > PER_WINDOW:
                images = [images[i] for i in rng.choice(len(images), PER_WINDOW, replace=False)]
            if progress and time.perf_counter() - last >= PROGRESS_SECONDS:
                last = time.perf_counter()
                progress(f"  {stats['windows']} random windows read")
            if not images:
                empty += 1
                if empty >= 1000 and empty == stats["windows"]:
                    return  # no inputs at all (min_events?)
                continue
            yield images
    finally:
        for reader in readers:
            reader.close()


def collect_inputs(
    config: AgentConfig,
    recordings: typing.Sequence[PathLike],
    frame_config,
    count: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    stride: int = DEFAULT_STRIDE,
    seed: int = 0,
    max_windows: int | None = None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Up to `count` model inputs (tiles or frames, not yet prepared) from the recordings, chosen
    at random from every `stride`-th time window of all files (reservoir sampling: the memory
    stays small for long recordings). Tiles without events are skipped (the frame configuration's
    min_events decides). Returns (images, {"files", "windows", "candidates"})."""
    stats = {"files": 0, "windows": 0, "candidates": 0}
    if count <= 0:
        return [], stats
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    seen = 0
    for images in _windows(config, recordings, frame_config, window_ms, stride, max_windows, stats):
        for image in images:
            seen += 1
            if len(chosen) < count:
                chosen.append(np.asarray(image, dtype=np.float32))
            else:
                slot = int(rng.integers(0, seen))
                if slot < count:
                    chosen[slot] = np.asarray(image, dtype=np.float32)
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order], stats


def collect_prepared(
    agent: Agent,
    recordings: typing.Sequence[PathLike],
    frame_config,
    count: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    stride: int = DEFAULT_STRIDE,
    seed: int = 0,
    max_windows: int | None = None,
    progress: typing.Callable[[str], None] | None = None,
    sampling: str = "auto",
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Like collect_inputs, but prepared for the model (the agent's preprocessing) and stored as
    float16 in one array (count x height x width x channels): for training with many inputs.

    sampling "random" (default for HDF5 files): time windows at random places, read directly
    (fast for very large files); "sequential" (default for .raw): every stride-th window of the
    whole files, inputs chosen by reservoir sampling. width / height: sensor size if the HDF5 file
    does not store it (else it is estimated from samples)."""
    config = agent.config
    stats = {"files": 0, "windows": 0, "candidates": 0}
    rng = np.random.default_rng(seed)
    data: np.ndarray | None = None
    filled = seen = 0
    pad_to = None
    if sampling == "auto":
        sampling = "random" if all(_is_hdf5(path) for path in recordings) else "sequential"
    if sampling == "random":
        source = _random_windows(config, recordings, frame_config, window_ms, max_windows, stats, seed, width, height,
                                 progress)
    elif sampling == "sequential":
        source = _windows(config, recordings, frame_config, window_ms, stride, max_windows, stats, progress)
    else:
        raise ValueError("sampling must be auto, random or sequential")
    for images in source:
        if sampling == "random" and filled >= count:
            break
        if pad_to is None:
            pad_to = _pad_size(config, frame_config, images)
        prepared = agent.prepare(images, pad_to)
        for sample in prepared:
            seen += 1
            if data is None:
                data = np.empty((count, *sample.shape), np.float16)
            if filled < count:
                data[filled] = sample
                filled += 1
            else:
                slot = int(rng.integers(0, seen))
                if slot < count:
                    data[slot] = sample
    if data is None:
        return np.zeros((0,), np.float16), stats
    data = data[:filled]
    return data[rng.permutation(filled)], stats


def _pad_size(config: AgentConfig, frame_config, images: typing.Sequence[np.ndarray]) -> tuple[int, int] | None:
    if config.input != "tiles" or not images:
        return None
    tiles = getattr(frame_config, "tiles", None)
    size = getattr(tiles, "size", None) if tiles is not None else None
    if not size:
        size = max(max(image.shape[0], image.shape[1]) for image in images)
    return (int(size), int(size))


def _keras_or_saved_model(config: AgentConfig, tf, workdir: pathlib.Path):
    """A converter for the agent's model: Keras (.keras / .h5) or a SavedModel directory."""
    path = config.model_path
    if path.is_dir():
        return tf.lite.TFLiteConverter.from_saved_model(os.fspath(path)), path
    suffix = path.suffix.lower()
    if suffix not in (".keras", ".h5", ".hdf5"):
        raise ValueError(
            f"agent {config.name}: {path.name} can not be quantized here; quantize the Keras model (.keras / .h5) or "
            "SavedModel it was made from (a .tflite is already converted, PyTorch models: export them with "
            "ai-edge-torch or to ONNX and use the quantizer of the target)"
        )
    from .backends import _load_keras_model

    model = _load_keras_model(os.fspath(path), config.safe_mode)
    saved = workdir / "saved_model"
    if hasattr(model, "export"):  # Keras 3: a SavedModel with the serving function
        model.export(os.fspath(saved), format="tf_saved_model", verbose=False)
    else:  # pragma: no cover - tf.keras 2
        tf.saved_model.save(model, os.fspath(saved))
    return tf.lite.TFLiteConverter.from_saved_model(os.fspath(saved)), saved


def _classes(output: np.ndarray, layout: str) -> np.ndarray:
    """Arg-max classes of a segmentation / classification output (class axis by the layout)."""
    if output.ndim == 4:
        axis = 1 if layout == "nchw" else 3
        if output.shape[axis] == 1:
            return (output > 0).squeeze(axis)
        return output.argmax(axis=axis)
    if output.ndim == 2 and output.shape[1] > 1:
        return output.argmax(axis=1)
    return output > 0


def compare_outputs(reference: list[np.ndarray], quantized: list[np.ndarray], layout: str,
                    target_classes: typing.Sequence[int] = ()) -> dict[str, typing.Any]:
    """Agreement of the classes of two models on the same inputs: share of equal classes, and for
    every target class the overlap (IoU) of its pixels and how many of them the quantized model
    finds (recall)."""
    a = _classes(np.asarray(reference[0]), layout)
    b = _classes(np.asarray(quantized[0]), layout)
    result: dict[str, typing.Any] = {
        "agreement": float(np.mean(a == b)) if a.size else None,
        "max_abs_difference": float(np.max(np.abs(np.asarray(reference[0], np.float32) - np.asarray(quantized[0], np.float32))))
        if a.size else None,
        "classes": {},
    }
    for target in target_classes:
        in_a, in_b = a == target, b == target
        union = np.logical_or(in_a, in_b).sum()
        result["classes"][int(target)] = {
            "pixels": int(in_a.sum()),
            "iou": float(np.logical_and(in_a, in_b).sum() / union) if union else None,
            "recall": float(np.logical_and(in_a, in_b).sum() / in_a.sum()) if in_a.sum() else None,
        }
    return result


def quantize_agent(
    config: AgentConfig,
    recordings: typing.Sequence[PathLike],
    frame_config,
    output: PathLike | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
    validation: int | None = None,
    window_ms: float = DEFAULT_WINDOW_MS,
    stride: int = DEFAULT_STRIDE,
    max_windows: int | None = None,
    int8_io: bool = False,
    qat: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    target_weight: float = 10.0,
    seed: int = 0,
    sampling: str = "auto",
    width: int | None = None,
    height: int | None = None,
    progress: typing.Callable[[str], None] | None = None,
) -> dict[str, typing.Any]:
    """Quantizes the agent's model to int8 and writes it as .tflite (default: <model>_int8.tflite
    next to the model). recordings: the training data (HDF5 event files, .raw); frame_config: the
    agent's frame configuration (FrameConfig).

    qat=False (post-training quantization): samples inputs calibrate the ranges, the float model is
    rounded. qat=True (quantization aware training, for opennve U-Nets): the network is trained
    further with the 8 bit rounding, `epochs` times over `samples` inputs (default 20000), to give
    the same classes as the float network (distillation, no labels needed); see
    opennve.networks.qat. validation other inputs (default 200 / QAT 1000) compare the int8 model
    with the original one. The inputs come from time windows at random places of HDF5 files (fast
    also for very large files) or from every stride-th window of .raw files (see collect_prepared).

    Returns a report: output, sizes, inputs, calibration / validation counts, agreement (share of
    equal classes, 0..1), classes (IoU / recall of the target classes), history (QAT: per epoch),
    seconds."""
    tf = _tensorflow()
    say = progress or (lambda message: None)
    started = time.perf_counter()
    if not recordings:
        raise ValueError("quantization needs recordings with typical data (the training data)")
    if validation is None:
        validation = DEFAULT_QAT_VALIDATION if qat else DEFAULT_VALIDATION
    if qat and samples == DEFAULT_SAMPLES:
        samples = DEFAULT_QAT_SAMPLES
    source = config.model_path
    output_path = pathlib.Path(output) if output else source.with_name(f"{source.stem}_int8.tflite")
    if output_path.suffix.lower() != ".tflite":
        raise ValueError("the quantized model is a .tflite file")

    float_agent = Agent(config.copy(device="cpu", fallback=True, delegate=None, batch_size=None))
    say(f"reading inputs from {len(recordings)} recording(s)")
    inputs, stats = collect_prepared(float_agent, recordings, frame_config, samples + validation, window_ms=window_ms,
                                     stride=stride, seed=seed, max_windows=max_windows, progress=say,
                                     sampling=sampling, width=width, height=height)
    say(f"{len(inputs)} inputs from {stats['windows']} windows ({stats['candidates']} candidates)")
    if not len(inputs):
        raise ValueError("the recordings gave no inputs (no events in the windows, or tiles below min_events)")
    if len(inputs) > validation:
        calibration, held_back = inputs[: len(inputs) - validation], inputs[len(inputs) - validation:]
    else:
        calibration, held_back = inputs, inputs[:0]
    target_classes = list(config.prediction.target_classes or ())
    history: list[dict] = []

    if qat:
        data, history = _qat(config, tf, calibration, held_back, target_classes, epochs=epochs, batch_size=batch_size,
                             learning_rate=learning_rate, target_weight=target_weight, int8_io=int8_io, seed=seed,
                             say=say)
    else:
        with tempfile.TemporaryDirectory() as temp:
            say(f"converting {source.name} (calibration with {len(calibration)} inputs)")
            converter, _ = _keras_or_saved_model(config, tf, pathlib.Path(temp))

            def representative():
                for sample in calibration:
                    yield [sample[np.newaxis].astype(np.float32)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8 if int8_io else tf.float32
            converter.inference_output_type = tf.int8 if int8_io else tf.float32
            data = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    report: dict[str, typing.Any] = {
        "agent": config.name,
        "model": os.fspath(source),
        "output": os.fspath(output_path),
        "method": "qat" if qat else "post-training",
        "model_bytes": _size(source),
        "output_bytes": len(data),
        "int8_io": int8_io,
        "input_shape": list(calibration.shape[1:]),
        "recordings": stats["files"],
        "windows": stats["windows"],
        "candidates": stats["candidates"],
        "calibration": len(calibration),
        "validation": len(held_back),
        "agreement": None,
        "max_abs_difference": None,
        "classes": {},
        "history": history,
    }
    if len(held_back):
        say(f"comparing the int8 .tflite with the original on {len(held_back)} other inputs")
        held_back = held_back.astype(np.float32)
        reference = float_agent.predict_batch(held_back)
        quantized_agent = Agent(config.copy(model=os.fspath(output_path.resolve()), backend="tflite", device="cpu",
                                            fallback=True, delegate=None, batch_size=None))
        try:
            quantized = quantized_agent.predict_batch(held_back)
        finally:
            quantized_agent.close()
        report.update(compare_outputs(reference, quantized, float_agent.layout, target_classes))
    float_agent.close()
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def _find_unet(model):
    """The opennve U-Net in a loaded Keras model (the model itself or one of its layers)."""
    from ..networks.unet import UNet

    if isinstance(model, UNet):
        return model
    for layer in getattr(model, "layers", []):
        if isinstance(layer, UNet):
            return layer
    return None


def _is_functional_unet(model) -> bool:
    names = {layer.name for layer in getattr(model, "layers", [])}
    return {"enc_conv0-1", "enc_convbtleneck-1", "dec_output"} <= names


def _unet_from_functional(model, inputs: np.ndarray):
    """The opennve U-Net with the weights of a functional U-Net with named layers (enc_conv{i}-1,
    enc_bn{i}-1, ..., dec_output; for instance the HSNR U-Net of embedded-unet). Checks that both
    give the same outputs."""
    from ..networks import Network, NetworkParameters

    names = {layer.name for layer in model.layers}
    depth = 0
    while f"enc_conv{depth}-1" in names:
        depth += 1
    first = model.get_layer("enc_conv0-1")
    head = model.get_layer("dec_output")
    activation = getattr(head.activation, "__name__", "linear")
    height, width, channels = model.input_shape[1:4]
    parameters = NetworkParameters(number_of_classes=head.filters, tile_height=height or inputs.shape[1],
                                   tile_width=width or inputs.shape[2], channels=channels)
    batch_norm = "enc_bn0-1" in names
    network = Network.build("UNet", parameters, layer_depth=depth, filters_orig=first.filters,
                            kernel_size=first.kernel_size[0], batch_norm=batch_norm,
                            final_activation=activation)
    network.transfer_weights_from(model, method="hsnr_unet", layer_depth=depth, batch_norm=batch_norm)
    expected = np.asarray(model.predict(inputs, verbose=0), np.float32)
    actual = np.asarray(network.model.predict(inputs, verbose=0), np.float32)
    scale = max(float(np.abs(expected).max()), 1.0)
    if expected.shape != actual.shape or float(np.abs(expected - actual).max()) > 1e-3 * scale:
        raise ValueError("the functional U-Net does not match the opennve U-Net (other layers or activations): "
                         "quantization aware training is not possible for it")
    return network.model


def _output_activation(teacher, unet, inputs: np.ndarray) -> str | None:
    """What the saved model does after the U-Net layer: nothing (None), "softmax" or "sigmoid"
    (for instance a model that wraps the U-Net and adds a softmax layer). Anything else can not be
    reproduced by the quantized network."""
    if teacher is unet:
        return None
    expected = np.asarray(teacher.predict(inputs, verbose=0), np.float32)
    logits = np.asarray(unet(inputs, training=False), np.float32)
    if expected.shape != logits.shape:
        raise ValueError("the model has layers around the U-Net that change the output shape: quantization aware "
                         "training needs the U-Net alone")

    def close(values):
        return float(np.abs(values - expected).max()) <= 1e-3 * max(1.0, float(np.abs(expected).max()))

    if close(logits):
        return None
    if unet.output_layer.activation.__name__ == "linear":
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        if close(shifted / shifted.sum(axis=-1, keepdims=True)):
            return "softmax"
        if close(1.0 / (1.0 + np.exp(-logits))):
            return "sigmoid"
    raise ValueError("the model has layers around the U-Net (for instance preprocessing) that quantization aware "
                     "training can not reproduce: save the U-Net alone")


def _qat(config: AgentConfig, tf, train: np.ndarray, validation: np.ndarray, target_classes, *, epochs, batch_size,
         learning_rate, target_weight, int8_io, seed, say) -> tuple[bytes, list[dict]]:
    from ..networks.qat import QuantizedUNet, train_qat, unet_logits
    from .backends import _load_keras_model

    path = config.model_path
    if path.suffix.lower() not in (".keras", ".h5", ".hdf5"):
        raise ValueError(f"agent {config.name}: quantization aware training needs the Keras model (.keras / .h5) "
                         f"of the U-Net, not {path.name}")
    teacher = _load_keras_model(os.fspath(path), config.safe_mode)
    unet = _find_unet(teacher)
    if unet is None and _is_functional_unet(teacher):
        say(f"{path.name}: functional U-Net (enc_conv0-1 ... dec_output), copied into the opennve U-Net")
        teacher = _unet_from_functional(teacher, train[: min(len(train), 64)].astype(np.float32))
        unet = _find_unet(teacher)
    if unet is None:
        raise ValueError(f"agent {config.name}: quantization aware training supports the opennve U-Net "
                         "(opennve.networks.UNet, or a functional U-Net with the layers enc_conv0-1 ... dec_output); "
                         "use post-training quantization (without --qat) for other models")
    activation = config.prediction.activation or "none"
    shape = tuple(train.shape[1:])
    output_activation = _output_activation(teacher, unet, train[: min(len(train), 16)].astype(np.float32))
    if output_activation:
        say(f"{path.name}: the model applies {output_activation} after the U-Net; the int8 model does too")
    student = QuantizedUNet.from_unet(unet, shape, output_activation=output_activation)
    calibration = train[: min(len(train), 2048)]
    student.calibrate(calibration[i:i + batch_size].astype(np.float32) for i in range(0, len(calibration), batch_size))
    say(f"quantization aware training: {len(train)} inputs, {epochs} epochs, batch {batch_size}, "
        f"learning rate {learning_rate:g}")
    history = train_qat(student, teacher, train, validation, activation=activation, target_classes=target_classes,
                        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                        target_weight=target_weight, seed=seed, progress=say, teacher_logits=unet_logits(unet))
    say("converting the trained network to int8 TensorFlow Lite")
    return student.to_tflite(shape, int8_io=int8_io), history


def _size(path: pathlib.Path) -> int:
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def quantize_report(report: dict[str, typing.Any]) -> str:
    """Text for the command line."""
    def megabytes(value):
        return f"{value / 1e6:.2f} MB"

    qat = report.get("method") == "qat"
    lines = [
        f"agent {report['agent']}: {pathlib.Path(report['model']).name} ({megabytes(report['model_bytes'])})"
        f" -> {report['output']} ({megabytes(report['output_bytes'])}, int8"
        f"{', int8 inputs / outputs' if report['int8_io'] else ', float32 inputs / outputs'})",
        f"inputs        {report['calibration']} for {'training' if qat else 'calibration'}, {report['validation']} for the comparison "
        f"(chosen from {report['candidates']} in {report['windows']} windows of {report['recordings']} recording(s))",
    ]
    if qat and report.get("history"):
        lines.append("training      epoch: loss (difference of the logits, relative), same class as the float network "
                     "(in the training, with simulated int8)")
        for entry in report["history"]:
            loss = "-" if entry["loss"] is None else f"{entry['loss']:.5f}"
            agreement = "-" if entry["agreement"] is None else f"{entry['agreement'] * 100:.3f} %"
            ious = ", ".join(f"class {k} IoU {v['iou'] * 100:.1f} %" for k, v in entry["classes"].items() if v["iou"] is not None)
            lines.append(f"  {entry['epoch']:>3}: {loss}, {agreement}{', ' + ious if ious else ''}")
    if report["agreement"] is not None:
        lines.append("int8 .tflite  (LiteRT, as openNVE runs it) compared with the original on other inputs:")
        lines.append(f"agreement     {report['agreement'] * 100:.2f} % of the pixels / inputs have the same class as with the original model")
        for target, values in report["classes"].items():
            if values["pixels"]:
                lines.append(f"class {target:<7} IoU {values['iou'] * 100:.1f} %, found {values['recall'] * 100:.1f} % "
                             f"of {values['pixels']} pixels of the original")
            else:
                lines.append(f"class {target:<7} not in the comparison inputs (more --validation or other recordings)")
        ious = [v["iou"] for v in report["classes"].values() if v["iou"] is not None]
        if report["agreement"] < 0.995 or (ious and min(ious) < 0.9):
            lines.append("warning       the int8 model differs clearly from the original"
                         + ("" if qat else ": train it with the int8 rounding (--qat)")
                         + ("; more training data or epochs help" if qat else ""))
    lines.append(f"time          {report['seconds']} s")
    return "\n".join(lines)
