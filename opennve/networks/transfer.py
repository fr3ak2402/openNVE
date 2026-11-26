"""Weight transfer between networks of the same structure, written in different ways.

The weights of a trained network often come without the code that built it, or the network is
rebuilt with other classes (a functional HSNR U-Net -> opennve UNet, a subclassed model -> a
functional one). transfer_weights copies the weights layer by layer into any network with the same
layers and checks the result:

    report = opennve.transfer_weights("unet_tiled.weights.h5", "UNet", reference="ORGIN_NET.keras",
                                      output="models/unet.keras")
    print(report.text())

Sources: a Keras model or Network, a .keras file, a Keras 3 .weights.h5 file or a Keras 2 .h5
file (model or weights). Destinations: a Keras model or Network, a .keras file (its structure; the
weights are replaced) or the name of a registered network (for "UNet" the options are read from
the weights: depth, filters, kernel size, classes, input channels, batch normalization).

Layers are paired by name (when the source has the names of the destination layers) or else in
order within each layer type (the 3rd Conv2D of the source -> the 3rd Conv2D of the destination,
the 2nd BatchNormalization -> the 2nd ...). Every pair must have the same weight shapes. Afterwards
the destination is compared with a reference model (the source if it is a model; else
reference=): same outputs on random sparse inputs (like event frames). A source with only initial
values (batch normalization 0 / 1, all biases 0) is reported: it was saved before the weights were
loaded or trained.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import tempfile
import typing

import numpy as np

PathLike = typing.Union[str, pathlib.Path]
MATCHES = ("auto", "name", "order")


@dataclasses.dataclass
class WeightGroup:
    """The weights of one layer: name, type (snake case class name such as "conv2d",
    "batch_normalization"; None if unknown) and arrays (in the layer's weight order)."""

    name: str
    kind: typing.Optional[str]
    arrays: list
    layer: typing.Any = None

    @property
    def shapes(self) -> list[tuple]:
        return [tuple(a.shape) for a in self.arrays]

    @property
    def signature(self) -> tuple:
        return tuple(len(shape) for shape in self.shapes)


@dataclasses.dataclass
class WeightSource:
    """Weights read by read_weights: groups, the model (if the source is a model) and whether the
    names are layer names (they can be matched by name)."""

    groups: list[WeightGroup]
    model: typing.Any = None
    layer_names: bool = False
    description: str = ""
    replaced_lambdas: tuple = ()


@dataclasses.dataclass
class TransferReport:
    """Result of transfer_weights."""

    source: str
    destination: str
    match: str
    layers: int
    arrays: int
    output: typing.Optional[str] = None
    warnings: list = dataclasses.field(default_factory=list)
    reference: typing.Optional[str] = None
    max_difference: typing.Optional[float] = None
    output_scale: typing.Optional[float] = None
    same_class: typing.Optional[float] = None
    replaced_lambdas: tuple = ()
    model: typing.Any = dataclasses.field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """No warnings and (if compared) the same outputs as the reference."""
        return not self.warnings and (self.max_difference is None or self.max_difference <= 1e-3 * max(1.0, self.output_scale or 0.0))

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data.pop("model")
        data["replaced_lambdas"] = list(self.replaced_lambdas)
        data["ok"] = self.ok
        return data

    def text(self) -> str:
        lines = [f"source        {self.source}", f"destination   {self.destination}",
                 f"transferred   {self.layers} layers ({self.arrays} weight arrays), paired {self.match}"]
        if self.replaced_lambdas:
            lines.append(f"lambda layers {', '.join(self.replaced_lambdas)} loaded as portable layers")
        if self.reference:
            if self.max_difference is None:
                lines.append(f"reference     {self.reference}: not compared")
            else:
                same = f", {self.same_class * 100:.3f} % same class" if self.same_class is not None else ""
                verdict = "same outputs" if self.max_difference <= 1e-3 * max(1.0, self.output_scale or 0.0) else "DIFFERENT OUTPUTS"
                lines.append(f"reference     {self.reference}: {verdict} (largest difference {self.max_difference:.3g}, "
                             f"outputs up to {self.output_scale:.3g}{same})")
        for warning in self.warnings:
            lines.append(f"warning       {warning}")
        if self.output:
            lines.append(f"saved         {self.output}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- reading
def _snake(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()


def _kind_of_name(name: str) -> tuple[str, int]:
    """("conv2d_transpose", 3) for "conv2d_transpose_3"; ("conv2d", 0) for "conv2d"."""
    match = re.fullmatch(r"(.+?)_(\d+)", name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def model_weights(model) -> list[WeightGroup]:
    """The weight groups of a Keras model: every layer with weights and no sublayers, depth first
    in the order Keras tracks them (the order they were created in; keras' own _flatten_layers
    visits the sublayers of nested layers in reverse order)."""
    model = getattr(model, "model", model)
    groups, seen = [], set()

    def visit(layer):
        for child in layer._layers:
            if id(child) in seen:
                continue
            seen.add(id(child))
            if child._layers:
                visit(child)
                continue
            variables = _variables(child)
            if variables:
                groups.append(WeightGroup(child.name, _snake(child.__class__.__name__),
                                          [np.asarray(v.numpy()) for v in variables], child))

    visit(model)
    return groups


def _variables(layer) -> list:
    """The layer's weights without random generator states (Dropout)."""
    return [v for v in layer.weights if "seed_generator" not in getattr(v, "path", "")]


def _set(layer, arrays) -> None:
    variables = _variables(layer)
    for variable, value in zip(variables, arrays):
        variable.assign(value)


def read_weights(source, safe_mode: bool = True) -> WeightSource:
    """Reads the weights of a model, a .keras file, a Keras 3 .weights.h5 or a Keras 2 .h5 file."""
    if not isinstance(source, (str, os.PathLike)):
        model = getattr(source, "model", source)
        return WeightSource(model_weights(model), model, True, f"model {getattr(model, 'name', '')}".strip())
    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    name = path.name.lower()
    if name.endswith(".keras") or path.is_dir():
        from .network import load_keras_model

        model = load_keras_model(path, safe_mode=safe_mode)
        return WeightSource(model_weights(model), model, True, path.name, tuple(getattr(model, "opennve_replaced_lambdas", ())))
    import h5py

    with h5py.File(path, "r") as file:
        if "layer_names" in file.attrs or ("model_weights" in file and "layer_names" in file["model_weights"].attrs):
            return _read_legacy(file, path)
        if "layers" in file:
            return _read_keras3(file, path)
    raise ValueError(f"{path.name}: not a Keras weights file (no 'layers' group and no 'layer_names')")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_legacy(file, path: pathlib.Path) -> WeightSource:
    """Keras 2 .h5 (save_weights or a whole model): layer names in order, weights by weight_names."""
    root = file["model_weights"] if "model_weights" in file else file
    groups = []
    for layer_name in (_decode(n) for n in root.attrs["layer_names"]):
        group = root[layer_name]
        names = [_decode(n) for n in group.attrs.get("weight_names", [])]
        if not names:
            continue
        arrays = [np.asarray(group[n][()]) for n in names]
        kind, _ = _kind_of_name(layer_name)
        groups.append(WeightGroup(layer_name, kind, arrays))
    return WeightSource(groups, None, True, path.name)


def _read_keras3(file, path: pathlib.Path) -> WeightSource:
    """Keras 3 .weights.h5. A functional or sequential model stores its layers flat as
    layers/<type>_<n>/vars: type and order per type come from the names. A subclassed model stores
    them nested by attribute name, without type and order: it can only be read into the same
    structure (see transfer_weights)."""
    layers = file["layers"]
    # layers without weights (Dropout: only its random generator state) are left out
    names = [name for name in layers if not set(layers[name]) <= {"seed_generator"}]
    flat = all("vars" in layers[name] and set(layers[name]) <= {"vars", "seed_generator"} for name in names)
    if not flat:
        raise _NestedWeights(path)
    groups = []
    for name in names:
        variables = layers[name]["vars"]
        arrays = [np.asarray(variables[str(i)][()]) for i in range(len(variables))]
        if not arrays:
            continue
        kind, index = _kind_of_name(name)
        groups.append((kind, index, WeightGroup(name, kind, arrays)))
    groups.sort(key=lambda item: (item[0], item[1]))
    # names here are type names, not the layer names: no matching by name
    return WeightSource([g for _, _, g in groups], None, False, path.name)


class _NestedWeights(ValueError):
    def __init__(self, path: pathlib.Path):
        super().__init__(
            f"{path.name}: the weights of a subclassed model (nested by attribute names, without layer types and "
            "order): they load only into a model of the same classes; give the .keras file of the model they come "
            "from as the source, or a destination built with the same classes")
        self.path = path


# ---------------------------------------------------------------------------- pairing
def pair_layers(source: WeightSource, destination: list[WeightGroup], match: str = "auto") -> tuple[list, str]:
    """Pairs of (source group, destination group) and the way they were paired ("by name",
    "in order per layer type", "in order per weight shape")."""
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    if match in ("auto", "name") and source.layer_names:
        by_name = {g.name: g for g in source.groups}
        if all(d.name in by_name and by_name[d.name].shapes == d.shapes for d in destination):
            return [(by_name[d.name], d) for d in destination], "by name"
        if match == "name":
            missing = [d.name for d in destination if d.name not in by_name]
            different = [d.name for d in destination if d.name in by_name and by_name[d.name].shapes != d.shapes]
            raise ValueError("pairing by name failed: " + (f"not in the source: {', '.join(missing[:8])}" if missing else
                             f"other shapes: {', '.join(different[:8])}"))
    elif match == "name":
        raise ValueError("pairing by name needs a source with layer names (a model, .keras or Keras 2 .h5 file)")
    known = all(g.kind for g in source.groups)
    key = (lambda g: g.kind) if known else (lambda g: g.signature)
    how = "in order per layer type" if known else "in order per weight shape"
    sources, destinations = _by_key(source.groups, key), _by_key(destination, key)
    problems = []
    for kind in sorted(set(sources) | set(destinations), key=str):
        a, b = sources.get(kind, []), destinations.get(kind, [])
        if len(a) != len(b):
            problems.append(f"{_label(kind)}: {len(a)} in the source, {len(b)} in the destination")
    if problems:
        raise ValueError("the networks have different layers: " + "; ".join(problems))
    pairs = []
    for kind, targets in destinations.items():
        for a, b in zip(sources[kind], targets):
            if a.shapes != b.shapes:
                raise ValueError(f"{_label(kind)} {a.name} -> {b.name}: different shapes {a.shapes} / {b.shapes} "
                                 "(other filters, kernel size or depth)")
            pairs.append((a, b))
    order = {id(d): i for i, d in enumerate(destination)}
    pairs.sort(key=lambda pair: order[id(pair[1])])
    return pairs, how


def _by_key(groups, key) -> dict:
    result: dict = {}
    for group in groups:
        result.setdefault(key(group), []).append(group)
    return result


def _label(kind) -> str:
    return kind if isinstance(kind, str) else f"layers with weights of ranks {kind}"


def initial_values(groups: list[WeightGroup]) -> typing.Optional[str]:
    """A note if the weights look like freshly created layers (never trained or never loaded):
    every batch normalization at 1 / 0 / 0 / 1 and every bias 0."""
    norms = [g for g in groups if g.kind == "batch_normalization" or (len(g.arrays) == 4 and g.signature == (1, 1, 1, 1))]
    biases = [g.arrays[1] for g in groups if len(g.arrays) == 2 and g.signature[1] == 1 and g.arrays[0].ndim >= 2]
    norms_initial = norms and all(np.all(g.arrays[0] == 1) and np.all(g.arrays[1] == 0) and np.all(g.arrays[2] == 0)
                                  and np.all(g.arrays[3] == 1) for g in norms)
    biases_zero = len(biases) >= 3 and all(not np.any(b) for b in biases)
    if norms_initial and (biases_zero or not biases):
        return ("the weights look untrained: every batch normalization has its initial values (1 / 0 / 0 / 1) and "
                "every bias is 0; they were probably saved before the trained weights were loaded")
    if biases_zero and not norms:
        return "every bias is 0: check that the weights are trained (they may be the initial values)"
    return None


# ---------------------------------------------------------------------------- transfer
def unet_options(groups: list[WeightGroup]) -> dict:
    """Options of the opennve UNet that has these weights (by layer types): layer_depth (number of
    transposed convolutions), filters_orig, kernel_size, channels, classes, batch_norm."""
    convs = [g for g in groups if g.kind == "conv2d"]
    deconvs = [g for g in groups if g.kind == "conv2d_transpose"]
    if not convs or not deconvs:
        raise ValueError("the UNet options can not be read from these weights (no Conv2D / Conv2DTranspose layers "
                         "with known types); give them with options=")
    first, last = convs[0].arrays[0], convs[-1].arrays[0]
    return {"layer_depth": len(deconvs), "filters_orig": int(first.shape[3]), "kernel_size": int(first.shape[0]),
            "channels": int(first.shape[2]), "classes": int(last.shape[3]),
            "batch_norm": any(g.kind == "batch_normalization" for g in groups)}


def _destination_model(destination, source: WeightSource, reference_model, options: dict, input_size, safe_mode):
    """(model, description) of the destination."""
    if not isinstance(destination, (str, os.PathLike)):
        model = getattr(destination, "model", destination)
        return model, f"model {getattr(model, 'name', '')}".strip()
    path = pathlib.Path(destination)
    if path.suffix.lower() in (".keras", ".h5") or path.is_dir():
        from .network import load_keras_model

        return load_keras_model(path, safe_mode=safe_mode), path.name
    from .network import Network
    from .registry import NetworkParameters

    name = str(destination)
    options = dict(options)
    shape = _input_shape(reference_model) or _input_shape(source.model)
    if input_size:
        shape = (int(input_size[0]), int(input_size[1]), shape[2] if shape else None)
    if name == "UNet":
        found = unet_options(source.groups)
        channels, classes = found.pop("channels"), found.pop("classes")
        found["final_activation"] = _final_activation(reference_model or source.model) or "linear"
        found.update(options)
        options = found
    else:
        channels, classes = options.pop("channels", None), options.pop("classes", None)
    channels = options.pop("channels", channels) or (shape[2] if shape else None)
    classes = options.pop("classes", classes)
    if not shape or not shape[0] or not shape[1]:
        raise ValueError("the input size of the destination network is not known: give input_size=(height, width)")
    if not channels or not classes:
        raise ValueError("give channels= and classes= for the destination network")
    parameters = NetworkParameters(int(classes), int(shape[0]), int(shape[1]), int(channels))
    network = Network.build(name, parameters, **options)
    text = ", ".join(f"{k}={v}" for k, v in options.items())
    return network.model, f"{name} ({shape[0]} x {shape[1]} x {channels} -> {classes} classes{', ' + text if text else ''})"


def _input_shape(model) -> typing.Optional[tuple]:
    if model is None:
        return None
    shape = model.input_shape
    shape = shape[0] if isinstance(shape, list) else shape
    return tuple(shape[1:4]) if len(shape) == 4 else None


def _final_activation(model) -> typing.Optional[str]:
    if model is None:
        return None
    groups = model_weights(model)
    if not groups:
        return None
    activation = getattr(groups[-1].layer, "activation", None)
    return getattr(activation, "__name__", None)


def sample_inputs(shape: tuple, count: int = 32, seed: int = 0) -> np.ndarray:
    """Random sparse inputs like event frames (about 10 % of the pixels set, values 0..1)."""
    rng = np.random.default_rng(seed)
    shape = (count,) + tuple(s or 64 for s in shape)
    return ((rng.random(shape) < 0.1) * rng.random(shape)).astype(np.float32)


def transfer_weights(
    source,
    destination,
    *,
    output: typing.Optional[PathLike] = None,
    reference=None,
    match: str = "auto",
    options: typing.Optional[dict] = None,
    input_size: typing.Optional[tuple] = None,
    inputs: typing.Optional[np.ndarray] = None,
    safe_mode: bool = True,
) -> TransferReport:
    """Copies the weights of source into destination (see the module documentation) and checks
    them. output: saves the destination model (.keras). reference: a model or file to compare the
    outputs with (default: the source if it is a model). options: options of a destination network
    given by name (for "UNet" they are read from the weights; options= overrides). input_size:
    (height, width) of a destination built by name if no reference has it. inputs: the comparison
    inputs (default: random sparse inputs)."""
    from .network import load_keras_model

    reference_model = None
    if reference is not None:
        reference_model = load_keras_model(reference, safe_mode=safe_mode) if isinstance(reference, (str, os.PathLike)) \
            else getattr(reference, "model", reference)
    try:
        weights = read_weights(source, safe_mode=safe_mode)
    except _NestedWeights:
        return _transfer_same_structure(source, destination, output, reference_model, reference, inputs, safe_mode)
    if reference_model is None and weights.model is not None:
        reference_model, reference = weights.model, weights.description
    model, description = _destination_model(destination, weights, reference_model, options or {}, input_size, safe_mode)
    destination_groups = model_weights(model)
    pairs, how = pair_layers(weights, destination_groups, match)
    for a, b in pairs:
        _set(b.layer, a.arrays)
    report = TransferReport(weights.description, description, how, len(pairs), sum(len(a.arrays) for a, _ in pairs),
                            replaced_lambdas=weights.replaced_lambdas + tuple(getattr(reference_model, "opennve_replaced_lambdas", ())
                                                                             if reference_model is not weights.model else ()),
                            model=model)
    note = initial_values(weights.groups)
    if note:
        report.warnings.append(note)
    _verify(pairs, report)
    if reference_model is not None:
        _compare(model, reference_model, reference, inputs, report)
    if output:
        output = pathlib.Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(os.fspath(output))
        report.output = os.fspath(output)
    return report


def transfer_into(source_model, destination_model, match: str = "auto", **options) -> None:
    """Weight transfer method "auto" (Network.transfer_weights_from(source, method="auto"))."""
    pairs, _ = pair_layers(read_weights(source_model), model_weights(destination_model), match)
    for a, b in pairs:
        _set(b.layer, a.arrays)


def _verify(pairs, report: TransferReport) -> None:
    for a, b in pairs:
        now = [np.asarray(v.numpy()) for v in _variables(b.layer)]
        if any(not np.array_equal(x, y) for x, y in zip(a.arrays, now)):
            report.warnings.append(f"layer {b.name}: the weights were not set")


def _compare(model, reference_model, reference_name, inputs, report: TransferReport) -> None:
    report.reference = reference_name if isinstance(reference_name, str) else os.fspath(reference_name) \
        if isinstance(reference_name, os.PathLike) else f"model {getattr(reference_model, 'name', '')}".strip()
    if inputs is None:
        shape = _input_shape(reference_model) or _input_shape(model)
        if shape is None:
            return
        inputs = sample_inputs(shape)
    inputs = np.asarray(inputs, np.float32)
    expected = np.asarray(reference_model.predict(inputs, verbose=0), np.float32)
    actual = np.asarray(model.predict(inputs, verbose=0), np.float32)
    if expected.shape != actual.shape:
        report.warnings.append(f"the reference gives outputs of shape {expected.shape}, the destination {actual.shape}")
        return
    report.max_difference = float(np.abs(expected - actual).max())
    report.output_scale = float(np.abs(expected).max())
    if expected.shape[-1] > 1:
        report.same_class = float(np.mean(expected.argmax(-1) == actual.argmax(-1)))
    if report.max_difference > 1e-3 * max(1.0, report.output_scale):
        report.warnings.append("the destination gives other outputs than the reference: other layers or settings "
                               "(activations, final activation, layer order) than the source")


def _transfer_same_structure(source, destination, output, reference_model, reference, inputs, safe_mode) -> TransferReport:
    """Nested Keras 3 weights (subclassed model): Keras loads them into a destination of the same
    classes; checked by comparing every array with the file."""
    import h5py

    path = pathlib.Path(source)
    if isinstance(destination, (str, os.PathLike)) and pathlib.Path(destination).suffix.lower() not in (".keras", ".h5"):
        raise _NestedWeights(path)
    model, description = _destination_model(destination, WeightSource([]), None, {}, None, safe_mode)
    model.load_weights(os.fspath(path))
    with tempfile.TemporaryDirectory() as temp:
        saved = pathlib.Path(temp) / "check.weights.h5"
        model.save_weights(os.fspath(saved))
        with h5py.File(path, "r") as a, h5py.File(saved, "r") as b:
            expected, actual = _datasets(a["layers"]), _datasets(b["layers"])
    if expected.keys() != actual.keys() or any(not np.array_equal(expected[k], actual[k]) for k in expected):
        raise ValueError(f"{path.name} does not fit {description}: a subclassed model's weights load only into the same classes")
    report = TransferReport(path.name, description, "by structure (same classes)", len(model_weights(model)),
                            len(expected), model=model)
    note = initial_values(model_weights(model))
    if note:
        report.warnings.append(note)
    if reference_model is not None:
        _compare(model, reference_model, reference, inputs, report)
    if output:
        output = pathlib.Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(os.fspath(output))
        report.output = os.fspath(output)
    return report


def _datasets(group, prefix="") -> dict:
    import h5py

    result = {}
    for name, item in group.items():
        if isinstance(item, h5py.Dataset):
            result[prefix + name] = np.asarray(item[()])
        else:
            result.update(_datasets(item, f"{prefix}{name}/"))
    return result
