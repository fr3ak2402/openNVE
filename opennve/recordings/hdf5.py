"""Reading and writing HDF5 event files (.h5, .hdf5) with the columns x, y, p, t and optionally l (label).

Supported layouts (found automatically, or selected with dataset="..."):

    columns:   a group with one dataset per column, for instance /events/x, /events/y, /events/p,
               /events/t, /events/l (also DSEC files: t relative to /t_offset)
    compound:  one dataset with named fields, for instance /events with fields x, y, p, t, l
    matrix:    one 2D dataset (events x columns); the column order comes from its "columns"
               attribute or columns=["t", "x", "y", "p"]
    pandas:    DataFrames saved with DataFrame.to_hdf (format "fixed"; format "table" needs pandas
               and PyTables)

Column names are case-insensitive; aliases: p = polarity / pol / on, t = ts / timestamp(s) /
time, l = label / labels / class / class_id. Polarity values > 0 are ON (1, True), all others OFF
(0, -1, False).

    with opennve.open_hdf5("recording.h5", config_path="cameras.json", name="left") as recording:
        for window in recording.windows():               # like a live camera
            labels = window.polarity_events["l"]          # labels of the events (if the file has them)

    opennve.write_hdf5("labeled.h5", events, width=1280, height=720)
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np

from ..cameras.config import CameraConfig, PathLike, load_config
from ..events.filters import FilterConfig
from ..events.output import OutputConfig
from ..events.window import WindowConfig
from .reader import POLARITY_EVENTS_DTYPE, RecordingReader, _empty_trigger

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
HDF5_SUFFIXES = (".h5", ".hdf5", ".hdf", ".he5", ".h")
LABELED_EVENTS_DTYPE = np.dtype(POLARITY_EVENTS_DTYPE.descr + [("l", "<i4")])

ALIASES = {
    "x": ("x",),
    "y": ("y",),
    "p": ("p", "polarity", "pol", "on"),
    "t": ("t", "ts", "timestamp", "timestamps", "time"),
    "l": ("l", "label", "labels", "class", "class_id"),
}
TIME_UNITS = {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0, "ns": 0.001}


def _h5py():
    try:
        import h5py  # type: ignore
    except ImportError as error:
        raise ImportError("HDF5 files need h5py: pip install h5py") from error
    try:  # compression filters of other tools (Blosc, LZ4, Zstd, ...), if installed
        import hdf5plugin  # type: ignore  # noqa: F401
    except ImportError:
        pass
    return h5py


def is_hdf5(path: PathLike) -> bool:
    """True if the file starts with the HDF5 signature."""
    try:
        with open(path, "rb") as file:
            return file.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def _match_columns(names: typing.Iterable[str]) -> dict[str, str]:
    """{x, y, p, t, l: name in the file} for the names that match an alias (case-insensitive)."""
    lowered = {name.lower(): name for name in names}
    result = {}
    for column, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                result[column] = lowered[alias]
                break
    return result


def _decode_names(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in np.asarray(values).ravel()]


# ------------------------------------------------------------------ layouts


class _Source:
    """Reads row ranges of one layout as {column: array}."""

    layout = "?"
    path = ""
    columns: dict[str, str]

    def __len__(self) -> int:
        raise NotImplementedError

    def read(self, start: int, stop: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def t_dtype(self) -> np.dtype:
        return self.read(0, min(1, len(self)))["t"].dtype

    def attrs(self) -> dict:
        return {}


class _ColumnSource(_Source):
    layout = "columns"

    def __init__(self, group, columns: dict[str, str]):
        self.group, self.columns, self.path = group, columns, group.name
        self._length = len(group[columns["t"]])

    def __len__(self):
        return self._length

    def read(self, start, stop):
        return {column: self.group[name][start:stop] for column, name in self.columns.items()}

    def t_dtype(self):
        return self.group[self.columns["t"]].dtype

    def attrs(self):
        merged = dict(self.group.attrs)
        merged.update({f"t.{key}": value for key, value in self.group[self.columns["t"]].attrs.items()})
        return merged


class _CompoundSource(_Source):
    layout = "compound"

    def __init__(self, dataset, columns: dict[str, str]):
        self.dataset, self.columns, self.path = dataset, columns, dataset.name

    def __len__(self):
        return len(self.dataset)

    def read(self, start, stop):
        data = self.dataset[start:stop]
        return {column: data[name] for column, name in self.columns.items()}

    def t_dtype(self):
        return self.dataset.dtype[self.columns["t"]]

    def attrs(self):
        return dict(self.dataset.attrs)


class _MatrixSource(_Source):
    layout = "matrix"

    def __init__(self, dataset, order: list[str]):
        self.dataset, self.path = dataset, dataset.name
        matched = _match_columns(order)
        self.columns = {column: str(order.index(name)) for column, name in matched.items()}

    def __len__(self):
        return self.dataset.shape[0]

    def read(self, start, stop):
        data = self.dataset[start:stop]
        return {column: data[:, int(index)] for column, index in self.columns.items()}

    def t_dtype(self):
        return self.dataset.dtype

    def attrs(self):
        return dict(self.dataset.attrs)


class _PandasFixedSource(_Source):
    """DataFrame.to_hdf(..., format="fixed"): block<i>_items (column names), block<i>_values (rows x items)."""

    layout = "pandas"

    def __init__(self, group):
        self.group, self.path = group, group.name
        self._where: dict[str, tuple[str, int]] = {}
        index = 0
        while f"block{index}_items" in group:
            for position, name in enumerate(_decode_names(group[f"block{index}_items"][()])):
                self._where[name] = (f"block{index}_values", position)
            index += 1
        self.columns = _match_columns(self._where)
        self._length = len(group["axis1"])

    def __len__(self):
        return self._length

    def read(self, start, stop):
        result = {}
        for column, name in self.columns.items():
            dataset, position = self._where[name]
            result[column] = self.group[dataset][start:stop, position]
        return result

    def t_dtype(self):
        dataset, _ = self._where[self.columns["t"]]
        return self.group[dataset].dtype


class _PandasTableSource(_Source):
    """DataFrame.to_hdf(..., format="table"): read with pandas (needs PyTables)."""

    layout = "pandas_table"

    def __init__(self, file_path: pathlib.Path, key: str, length: int):
        try:
            import pandas  # type: ignore
        except ImportError as error:
            raise ImportError("HDF5 files written by pandas in table format need pandas and PyTables: pip install pandas tables") from error
        self.pandas, self.file_path, self.key, self.path, self._length = pandas, file_path, key, key, length
        head = pandas.read_hdf(file_path, key, start=0, stop=1)
        self.columns = _match_columns(head.columns.astype(str))
        self._names = {column: head.columns[list(head.columns.astype(str)).index(name)] for column, name in self.columns.items()}

    def __len__(self):
        return self._length

    def read(self, start, stop):
        frame = self.pandas.read_hdf(self.file_path, self.key, start=start, stop=stop)
        return {column: frame[name].to_numpy() for column, name in self._names.items()}


def _candidates(file, file_path: pathlib.Path, columns: typing.Optional[list[str]]) -> list[_Source]:
    h5py = _h5py()
    found: list[_Source] = []

    def consider(name, item):
        if isinstance(item, h5py.Group):
            if item.attrs.get("pandas_type") in (b"frame", "frame"):
                source = _PandasFixedSource(item)
                found.append(source)
                return
            if item.attrs.get("pandas_type") in (b"frame_table", "frame_table") and "table" in item:
                found.append(_PandasTableSource(file_path, item.name, len(item["table"])))
                return
            datasets = {key: value for key, value in item.items() if isinstance(value, h5py.Dataset) and value.ndim == 1}
            matched = _match_columns(datasets)
            if {"x", "y", "t"} <= set(matched):
                lengths = {len(datasets[name]) for name in matched.values()}
                if len(lengths) == 1:
                    found.append(_ColumnSource(item, matched))
        elif isinstance(item, h5py.Dataset):
            if item.dtype.names:
                matched = _match_columns(item.dtype.names)
                if {"x", "y", "t"} <= set(matched) and item.ndim == 1:
                    found.append(_CompoundSource(item, matched))
            elif item.ndim == 2 and item.shape[1] in (3, 4, 5):
                order = columns
                if order is None:
                    for key in ("columns", "fields", "names", "column_names"):
                        if key in item.attrs:
                            order = _decode_names(item.attrs[key])
                            break
                if order is not None and len(order) == item.shape[1] and {"x", "y", "t"} <= set(_match_columns(order)):
                    found.append(_MatrixSource(item, list(order)))

    consider("/", file)
    file.visititems(consider)
    # a group inside a pandas group is not a separate candidate
    pandas_paths = [s.path for s in found if s.layout.startswith("pandas")]
    return [s for s in found if not any(s.path != p and s.path.startswith(p + "/") for p in pandas_paths)]


# ------------------------------------------------------------------ reader


class Hdf5Reader(RecordingReader):
    """Reads an HDF5 event file and delivers the same packets and windows as a live camera.

    Use opennve.open_hdf5() (or open_recording()) to create it. Events with labels have the
    additional field "l" (int32) in packet.polarity_events / window.polarity_events, and the
    output field "l" ("output": {"fields": ["x", "y", "p", "t", "l"]}) contains them.
    """

    def __init__(
        self,
        path: PathLike,
        config: typing.Optional[CameraConfig] = None,
        *,
        name: typing.Optional[str] = None,
        filter: typing.Union[FilterConfig, dict, None] = None,
        output: typing.Union[OutputConfig, dict, None] = None,
        window: typing.Union[WindowConfig, dict, None, str] = "config",
        dataset: typing.Optional[str] = None,
        columns: typing.Optional[list[str]] = None,
        time_unit: str = "auto",
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        labels: bool = True,
        chunk_events: int = 1_000_000,
        sort: bool = False,
        speed: typing.Optional[float] = None,
        zero_time: bool = False,
    ):
        h5py = _h5py()
        path = pathlib.Path(path)
        if time_unit != "auto" and time_unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be auto, {', '.join(TIME_UNITS)}")
        if chunk_events <= 0:
            raise ValueError("chunk_events must be positive")
        self._h5 = h5py.File(path, "r")
        try:
            self.source = self._select(path, dataset, columns)
            missing = [column for column in ("x", "y", "p", "t") if column not in self.source.columns]
            if missing:
                raise ValueError(
                    f"{path}: {self.source.path} has no column(s) {', '.join(missing)} "
                    f"(found: {', '.join(self.source.columns.values())})"
                )
            self.has_labels = labels and "l" in self.source.columns
            self.time_unit = self._time_unit(time_unit)
            self._t_offset = self._find_t_offset()
            self._width, self._height = self._size(width, height)
            self._chunk_events = chunk_events
            self._position = 0
            self._last_t: typing.Optional[int] = None
            self._sorted_events: typing.Optional[np.ndarray] = None
            if sort:
                self._sorted_events = self._convert(self.source.read(0, len(self.source)))
                self._sorted_events = self._sorted_events[np.argsort(self._sorted_events["t"], kind="stable")]
            super().__init__(
                path, config, name=name, filter=filter, output=output, window=window, speed=speed, zero_time=zero_time
            )
        except BaseException:
            self._h5.close()
            raise

    def _select(self, path, dataset, columns) -> _Source:
        h5py = _h5py()
        if dataset is not None:
            if dataset not in self._h5:
                raise ValueError(f"{path}: no dataset or group {dataset!r}")
            item = self._h5[dataset]
            sources = [s for s in _candidates(self._h5, path, columns) if s.path == item.name]
            if not sources and isinstance(item, h5py.Dataset) and item.ndim == 2 and columns is None:
                raise ValueError(f"{path}: {dataset} is a matrix without column names, pass columns=[...]")
            if not sources:
                raise ValueError(f"{path}: {dataset} does not contain the columns x, y, p, t")
            return sources[0]
        sources = _candidates(self._h5, path, columns)
        if not sources:
            raise ValueError(
                f"{path}: no events with the columns x, y, p, t found (a group with one dataset per column, "
                "a dataset with named fields, a DataFrame written by pandas, or a 2D dataset with columns=[...])"
            )
        if len(sources) > 1:
            preferred = [s for s in sources if s.path.strip("/").lower() in ("events", "cd/events", "", "data")]
            if len(preferred) == 1:
                return preferred[0]
            if len(preferred) == 0 and len({s.path for s in sources}) > 1:
                raise ValueError(
                    f"{path}: several event datasets ({', '.join(s.path for s in sources)}), choose one with dataset=..."
                )
        return sources[0]

    def _time_unit(self, value: str) -> str:
        if value != "auto":
            return value
        attrs = self.source.attrs()
        for key in ("t.units", "t.unit", "time_unit", "t_unit", "units"):
            if key in attrs:
                unit = _decode_names(attrs[key])[0].lower().replace("µ", "u").replace("micro", "u")
                for name in TIME_UNITS:
                    if unit in (name, name + "s", {"us": "useconds", "ms": "mseconds", "s": "seconds", "ns": "nseconds"}[name]):
                        return name
        return "s" if np.issubdtype(self.source.t_dtype(), np.floating) else "us"

    def _find_t_offset(self) -> int:
        for location in ("t_offset", f"{self.source.path.rstrip('/')}/t_offset"):
            if location in self._h5 and self._h5[location].shape in ((), (1,)):
                return int(np.asarray(self._h5[location][()]).ravel()[0])
        return 0

    def _size(self, width, height) -> tuple[int, int]:
        attrs = {**dict(self._h5.attrs), **self.source.attrs()}
        for w_key, h_key in (("width", "height"), ("sensor_width", "sensor_height"), ("x_size", "y_size")):
            if width is None and w_key in attrs:
                width = int(np.asarray(attrs[w_key]).ravel()[0])
            if height is None and h_key in attrs:
                height = int(np.asarray(attrs[h_key]).ravel()[0])
        if "geometry" in attrs and (width is None or height is None):
            text = _decode_names(attrs["geometry"])[0]
            if "x" in text:
                width, height = width or int(text.split("x")[0]), height or int(text.split("x")[1])
        if width is None or height is None:  # one pass over x and y
            max_x = max_y = -1
            for start in range(0, len(self.source), 4_000_000):
                part = self.source.read(start, min(len(self.source), start + 4_000_000))
                if len(part["x"]):
                    max_x, max_y = max(max_x, int(np.max(part["x"]))), max(max_y, int(np.max(part["y"])))
            width = width if width is not None else max(1, max_x + 1)
            height = height if height is not None else max(1, max_y + 1)
        return int(width), int(height)

    def _convert(self, data: dict[str, np.ndarray]) -> np.ndarray:
        count = len(data["t"])
        events = np.empty(count, dtype=LABELED_EVENTS_DTYPE if self.has_labels else POLARITY_EVENTS_DTYPE)
        t = np.asarray(data["t"])
        if self.time_unit == "us" and np.issubdtype(t.dtype, np.integer):
            t = t.astype(np.int64)
        else:
            t = np.round(t.astype(np.float64) * TIME_UNITS[self.time_unit]).astype(np.int64)
        t = t + self._t_offset
        if count and t.min() < 0:
            raise ValueError(f"{self.path}: negative time stamps (use time_unit / zero_time)")
        events["t"] = t
        events["x"] = data["x"]
        events["y"] = data["y"]
        p = np.asarray(data["p"])
        events["on"] = p if p.dtype == np.bool_ else p > 0
        if self.has_labels:
            events["l"] = data["l"]
        return events

    # Information

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    @property
    def event_count(self) -> int:
        return len(self.source)

    def info(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "format": "HDF5",
            "layout": self.source.layout,
            "dataset": self.source.path,
            "columns": dict(self.source.columns),
            "labels": self.has_labels,
            "time_unit": self.time_unit,
            "t_offset": self._t_offset,
            "events": len(self.source),
            "resolution": {"width": self.width, "height": self.height},
            "size_bytes": self.path.stat().st_size,
            "filter": self.filter_config.to_dict(),
            "output": self.output_config.to_dict(),
            "window": None if self.window_config is None else self.window_config.to_dict(),
        }

    # Reading

    def _next_chunk(self):
        total = len(self.source)
        if self._position >= total:
            return None
        stop = min(total, self._position + self._chunk_events)
        if self._sorted_events is not None:
            events = self._sorted_events[self._position:stop].copy()
        else:
            events = self._convert(self.source.read(self._position, stop))
            t = events["t"]
            if len(t) and ((self._last_t is not None and t[0] < self._last_t) or np.any(t[1:] < t[:-1])):
                raise ValueError(f"{self.path}: the events are not sorted by time, open the file with sort=True")
        self._position = stop
        if len(events):
            self._last_t = int(events["t"][-1])
        current_t = self._last_t
        return events, _empty_trigger(), current_t

    def _close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def open_hdf5(
    path: PathLike,
    config: typing.Union[CameraConfig, dict, str, None] = None,
    *,
    config_path: typing.Optional[PathLike] = None,
    name: typing.Optional[str] = None,
    **kwargs,
) -> Hdf5Reader:
    """Opens an HDF5 event file (.h5 / .hdf5) with the columns x, y, p, t (and optionally l).

    The optional camera configuration (CameraConfig, dict, JSON string, or config_path with name)
    provides the name, filter, output format and window duration.

    Keyword arguments: dataset (group or dataset with the events), columns (column order of a 2D
    dataset), time_unit ("auto", "us", "ms", "s", "ns"), width / height (else from the attributes or
    the largest x / y), labels (read the l column, default True), chunk_events, sort (sort by time,
    reads the whole file), speed (1.0 = real time), zero_time, filter, output, window.
    """
    if config_path is not None:
        config = load_config(config_path, name)
        name = None
    if isinstance(config, str):
        config = CameraConfig.from_json(config)
    elif isinstance(config, dict):
        config = CameraConfig.from_dict(config)
    return Hdf5Reader(path, config, name=name, **kwargs)


def read_hdf5(path: PathLike, **kwargs) -> np.ndarray:
    """Reads all events of an HDF5 file at once (fields t, x, y, on and l if the file has labels)."""
    with open_hdf5(path, **kwargs) as recording:
        return recording.read_all()[0]


# ------------------------------------------------------------------ writing


def _columns_of(events: np.ndarray) -> dict[str, np.ndarray]:
    names = events.dtype.names or ()
    if not {"x", "y", "t"} <= set(names) or not ({"p", "on"} & set(names)):
        raise ValueError("events need the fields x, y, t and p (or on)")
    columns = {
        "x": events["x"].astype(np.uint16),
        "y": events["y"].astype(np.uint16),
        "p": (events["p"] if "p" in names else events["on"]).astype(np.uint8),
        "t": events["t"].astype(np.int64),
    }
    if "l" in names:
        columns["l"] = events["l"].astype(np.int32)
    return columns


class Hdf5Writer:
    """Appends events to an HDF5 file: one dataset per column in a group (default /events),
    resizable and compressed, with the attributes width, height and time_unit "us".

        with opennve.Hdf5Writer("out.h5", width=1280, height=720) as writer:
            writer.write(events)        # structured array with x, y, t, p (or on), optionally l
    """

    def __init__(
        self,
        path: PathLike,
        *,
        group: str = "events",
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        labels: typing.Optional[bool] = None,
        compression: typing.Optional[str] = "gzip",
        mode: str = "a",
        attrs: typing.Optional[dict] = None,
    ):
        h5py = _h5py()
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, mode)
        self._group = self._file.require_group(group)
        self._labels = labels
        self._compression = compression
        for key, value in (("width", width), ("height", height), ("time_unit", "us")):
            if value is not None:
                self._group.attrs[key] = value
        for key, value in (attrs or {}).items():
            self._group.attrs[key] = value
        self.count = len(self._group["t"]) if "t" in self._group else 0

    def write(self, events: np.ndarray):
        columns = _columns_of(events)
        if self._labels is None:
            self._labels = "l" in columns or "l" in self._group
        if self._labels and "l" not in columns:
            columns["l"] = np.zeros(len(events), np.int32)
        if not self._labels:
            columns.pop("l", None)
        for name, values in columns.items():
            if name not in self._group:
                self._group.create_dataset(
                    name, shape=(self.count,), maxshape=(None,), dtype=values.dtype, chunks=(65536,),
                    compression=self._compression,
                )
            dataset = self._group[name]
            dataset.resize((self.count + len(values),))
            dataset[self.count:] = values
        self.count += len(events)

    def flush(self):
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "Hdf5Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def write_hdf5(path: PathLike, events: np.ndarray, *, mode: str = "w", **kwargs) -> pathlib.Path:
    """Writes events (structured array with x, y, t, p or on, optionally l) to an HDF5 file.
    Keyword arguments as for Hdf5Writer (group, width, height, compression, attrs)."""
    with Hdf5Writer(path, mode=mode, **kwargs) as writer:
        writer.write(events)
    return pathlib.Path(path)


def open_recording(path: PathLike, config=None, **kwargs) -> RecordingReader:
    """Opens a recording by its type: HDF5 (.h5, .hdf5, .h, or the HDF5 signature) or .raw."""
    from .raw import open_raw

    path = pathlib.Path(path)
    if path.suffix.lower() in HDF5_SUFFIXES or (path.suffix.lower() != ".raw" and is_hdf5(path)):
        return open_hdf5(path, config, **kwargs)
    return open_raw(path, config, **kwargs)


__all__ = [
    "Hdf5Reader",
    "Hdf5Writer",
    "is_hdf5",
    "open_hdf5",
    "open_recording",
    "read_hdf5",
    "write_hdf5",
]
