"""Reading .raw files (Prophesee / Metavision RAW format, EVT 2.0, EVT 2.1 and EVT 3.0).

RAW files are written by Metavision Studio, the Metavision SDK / OpenEB and the SENSING
software. They start with a text header ("% key value" lines, for instance
"% format EVT3;height=720;width=1280"), followed by the binary event stream of the camera.

    import opennve

    with opennve.open_raw("recording.raw") as raw:
        print(raw.info())
        for packet in raw.events():      # same packets as a live camera
            print(len(packet.events))

The decoders follow OpenEB (hal/cpp/include/metavision/hal/decoders), vectorized with numpy.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np

from ..cameras.config import CameraConfig, PathLike, load_config
from ..events.filters import FilterConfig
from ..events.output import OutputConfig
from ..events.window import WindowConfig
from .reader import (
    POLARITY_EVENTS_DTYPE,
    TRIGGER_EVENTS_DTYPE,
    RecordingReader,
    _empty_polarity,
    _empty_trigger,
)


FORMATS = ("EVT3", "EVT2", "EVT21")
_WORD_TYPES = {"EVT3": np.dtype("<u2"), "EVT2": np.dtype("<u4"), "EVT21": np.dtype("<u8")}
_MAXIMUM_HEADER_BYTES = 1 << 20

# Sensor size of the formats, if the header has no geometry
_DEFAULT_GEOMETRY = (1280, 720)


# ------------------------------------------------------------------ header


@dataclasses.dataclass
class RawHeader:
    """Header of a RAW file.

    Attributes:
        fields: All "% key value" lines, for instance {"format": "EVT3;height=720;width=1280", ...}.
        format: Event format: "EVT3", "EVT2" or "EVT21".
        width, height: Sensor size in pixels.
        endianness: EVT 2.1 only, "little" or "legacy".
        data_offset: Position of the first event byte in the file.
    """

    fields: dict[str, str]
    format: str
    width: int
    height: int
    endianness: str
    data_offset: int

    @property
    def serial(self) -> typing.Optional[str]:
        return self.fields.get("serial_number")


def _normalize_format(name: str) -> str:
    key = name.strip().upper().replace(".", "").replace(" ", "")
    aliases = {"EVT3": "EVT3", "EVT30": "EVT3", "3": "EVT3", "30": "EVT3",
               "EVT2": "EVT2", "EVT20": "EVT2", "2": "EVT2", "20": "EVT2",
               "EVT21": "EVT21", "21": "EVT21"}
    if key not in aliases:
        raise ValueError(f"unsupported RAW event format {name!r} (supported: EVT 2.0, EVT 2.1, EVT 3.0)")
    return aliases[key]


def read_header(
    path: PathLike,
    format: typing.Optional[str] = None,
    width: typing.Optional[int] = None,
    height: typing.Optional[int] = None,
) -> RawHeader:
    """Reads the header of a RAW file. format, width and height override the header
    (needed for files without header, for instance raw camera data)."""
    fields: dict[str, str] = {}
    offset = 0
    with open(path, "rb") as file:
        while offset < _MAXIMUM_HEADER_BYTES:
            start = file.tell()
            if file.read(2) != b"% ":
                file.seek(start)
                break
            line = file.readline()
            offset = file.tell()
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            key, _, value = text.partition(" ")
            if key == "end":
                break
            fields[key] = value.strip()
        offset = file.tell()

    format_parameters: dict[str, str] = {}
    format_name = None
    if "format" in fields:
        parts = fields["format"].split(";")
        format_name = parts[0]
        for part in parts[1:]:
            key, _, value = part.partition("=")
            format_parameters[key.strip()] = value.strip()
    elif "evt" in fields:
        format_name = fields["evt"]
    if format is not None:
        format_name = format
    if format_name is None:
        raise ValueError(
            f"{path}: the event format is unknown (no header), pass format=\"EVT3\", \"EVT2\" or \"EVT21\""
        )
    normalized = _normalize_format(format_name)

    header_width = header_height = None
    if "width" in format_parameters and "height" in format_parameters:
        header_width, header_height = int(format_parameters["width"]), int(format_parameters["height"])
    elif "geometry" in fields:
        geometry_width, _, geometry_height = fields["geometry"].partition("x")
        header_width, header_height = int(geometry_width), int(geometry_height)
    result_width = width if width is not None else header_width
    result_height = height if height is not None else header_height
    if result_width is None or result_height is None:
        result_width, result_height = _DEFAULT_GEOMETRY

    endianness = format_parameters.get("endianness", fields.get("endianness", "little")).lower()
    if endianness not in ("little", "legacy"):
        raise ValueError(f"{path}: unsupported EVT 2.1 endianness {endianness!r}")

    return RawHeader(
        fields=fields,
        format=normalized,
        width=int(result_width),
        height=int(result_height),
        endianness=endianness,
        data_offset=offset,
    )


# ------------------------------------------------------------------ decoders


def _forward_fill(mask: np.ndarray, values: np.ndarray, initial):
    """For each position, the value at the latest position <= i where mask is True (initial before)."""
    index = np.where(mask, np.arange(len(mask)), -1)
    np.maximum.accumulate(index, out=index)
    result = np.empty(len(mask), dtype=values.dtype)
    has_value = index >= 0
    result[has_value] = values[index[has_value]]
    result[~has_value] = initial
    return result


class _Evt3Decoder:
    """EVT 3.0: 16-bit words, see OpenEB evt3_decoder.h."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.started = False  # events before the first time high are ignored (as in OpenEB)
        self.y = 0
        self.is_cd = False
        self.high = 0
        self.low = 0
        self.loop = 0
        self.base_x = 0
        self.base_on = False

    def decode(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray, typing.Optional[int]]:
        if not self.started:
            first_high = np.flatnonzero((words >> 12) == 0x8)
            if len(first_high) == 0:
                return _empty_polarity(), _empty_trigger(), None
            high = int(words[first_high[0]] & 0xFFF)
            self.high = max(high - 1, 0)
            self.started = True
            words = words[first_high[0]:]
        count = len(words)
        if count == 0:
            return _empty_polarity(), _empty_trigger(), None
        types = words >> 12
        content = (words & 0xFFF).astype(np.int64)

        # Time: loop, high (12 bits) and low (12 bits)
        high_positions = np.flatnonzero(types == 0x8)
        highs = content[high_positions]
        previous_highs = np.concatenate(([self.high], highs[:-1]))
        loops = self.loop + np.cumsum(previous_highs >= highs + 2048)
        # time low: set by EVT_TIME_LOW, reset to 0 by a time high with a new value
        low_positions = np.flatnonzero(types == 0x6)
        reset_positions = high_positions[highs != previous_highs]
        low_set_positions = np.concatenate((low_positions, reset_positions))
        low_set_values = np.concatenate((content[low_positions], np.zeros(len(reset_positions), np.int64)))
        order = np.argsort(low_set_positions, kind="stable")
        low_set_positions, low_set_values = low_set_positions[order], low_set_values[order]

        # Time high including loops, in one value: loop << 12 | high
        high_full = (loops << 12) | highs
        initial_high_full = (self.loop << 12) | self.high
        # Row: EVT_ADDR_Y (CD events) sets y, an EM row (type 1) makes the following events invalid
        row_positions = np.flatnonzero(types <= 0x1)
        row_values = np.where(types[row_positions] == 0x0, content[row_positions] & 0x7FF, _INVALID_ROW)
        initial_row = self.y if self.is_cd else _INVALID_ROW
        # Vectors: base x and polarity, x increments of VECT_12 (12) and VECT_8 (8)
        base_positions = np.flatnonzero(types == 0x3)
        increments = np.zeros(count + 1, dtype=np.int64)
        increments[1:] = np.cumsum(np.where(types == 0x4, 12, np.where(types == 0x5, 8, 0)))

        def state_at(positions: np.ndarray):
            """Time, row and validity at sorted word positions (computed once per distinct word)."""
            if len(positions) == 0:
                empty = np.empty(0, dtype=np.int64)
                return empty, empty, np.empty(0, dtype=bool)
            change = np.empty(len(positions), dtype=bool)
            change[0] = True
            np.not_equal(positions[1:], positions[:-1], out=change[1:])
            distinct = positions[change]
            inverse = np.cumsum(change) - 1
            t = (_fill_at(high_positions, high_full, distinct, initial_high_full) << 12) | _fill_at(
                low_set_positions, low_set_values, distinct, self.low
            )
            row = _fill_at(row_positions, row_values, distinct, initial_row)
            return t[inverse], row[inverse], (row < self.height)[inverse]

        # Single events (EVT_ADDR_X)
        single = np.flatnonzero(types == 0x2)
        x_parts = [content[single] & 0x7FF]
        word_parts = [single]
        bit_parts = [np.zeros(len(single), dtype=np.int64)]
        on_parts = [((content[single] >> 11) & 1).astype(bool)]
        for vector_type, bits in ((0x4, 12), (0x5, 8)):
            positions = np.flatnonzero(types == vector_type)
            if len(positions) == 0:
                continue
            base_index = np.searchsorted(base_positions, positions, side="right") - 1
            has_base = base_index >= 0
            safe_base = np.where(has_base, base_index, 0)
            base_word = base_positions[safe_base] if len(base_positions) > 0 else np.zeros(len(positions), np.int64)
            base_x = np.where(
                has_base,
                (content[base_word] & 0x7FF) + increments[positions] - increments[base_word],
                self.base_x + increments[positions],
            )
            on = np.where(has_base, ((content[base_word] >> 11) & 1).astype(bool), self.base_on)
            bit_set = ((content[positions][:, None] >> np.arange(bits)) & 1).astype(bool)
            rows, columns = np.nonzero(bit_set)
            x_parts.append(base_x[rows] + columns)
            word_parts.append(positions[rows])
            bit_parts.append(columns.astype(np.int64))
            on_parts.append(on[rows])

        word_positions = np.concatenate(word_parts)
        order = np.argsort(word_positions * 32 + np.concatenate(bit_parts), kind="stable")
        word_positions = word_positions[order]
        x = np.concatenate(x_parts)[order]
        on = np.concatenate(on_parts)[order]
        t, y, valid = state_at(word_positions)
        keep = valid & (x >= 0) & (x < self.width)
        events = np.empty(int(np.count_nonzero(keep)), dtype=POLARITY_EVENTS_DTYPE)
        events["t"] = t[keep]
        events["x"] = x[keep]
        events["y"] = y[keep]
        events["on"] = on[keep]

        trigger_positions = np.flatnonzero(types == 0xA)
        trigger_t, _, _ = state_at(trigger_positions)
        triggers = np.empty(len(trigger_positions), dtype=TRIGGER_EVENTS_DTYPE)
        triggers["t"] = trigger_t
        triggers["id"] = (content[trigger_positions] >> 8) & 0xF
        triggers["rising"] = (content[trigger_positions] & 1).astype(bool)

        # Carry the state to the next chunk
        last = np.array([count - 1])
        last_t, last_row, _ = state_at(last)
        last_high_full = int(_fill_at(high_positions, high_full, last, initial_high_full)[0])
        self.loop, self.high = last_high_full >> 12, last_high_full & 0xFFF
        self.low = int(_fill_at(low_set_positions, low_set_values, last, self.low)[0])
        self.is_cd = int(last_row[0]) != _INVALID_ROW
        if self.is_cd:
            self.y = int(last_row[0])
        if len(base_positions) > 0:
            last_base = base_positions[-1]
            self.base_x = int((content[last_base] & 0x7FF) + increments[count] - increments[last_base])
            self.base_on = bool((content[last_base] >> 11) & 1)
        else:
            self.base_x += int(increments[count])
        return events, triggers, int(last_t[0])


_INVALID_ROW = 1 << 16


def _fill_at(set_positions: np.ndarray, values: np.ndarray, positions: np.ndarray, initial) -> np.ndarray:
    """Value of the latest set position <= each query position (initial if there is none).
    set_positions must be sorted."""
    index = np.searchsorted(set_positions, positions, side="right") - 1
    values = np.asarray(values)
    if len(values) == 0:
        return np.full(len(positions), initial, dtype=values.dtype if values.dtype != object else np.int64)
    result = values[np.maximum(index, 0)]
    return np.where(index >= 0, result, initial).astype(values.dtype)


class _Evt2Decoder:
    """EVT 2.0 (32-bit words) and EVT 2.1 (64-bit words, vectorized), see OpenEB evt2/evt21 decoders."""

    LOOP = 1 << 34

    def __init__(self, width: int, height: int, vectorized: bool, legacy: bool = False):
        self.width = width
        self.height = height
        self.vectorized = vectorized
        self.legacy = legacy
        self.started = False
        self.time_high = 0  # time high << 6, including loops

    def _fields(self, words: np.ndarray):
        if not self.vectorized:
            words = words.astype(np.uint64)
            return (
                (words >> 28) & 0xF,  # type
                words & 0xFFFFFFF,  # time high payload
                (words >> 22) & 0x3F,  # timestamp LSB
                (words >> 11) & 0x7FF,  # x
                words & 0x7FF,  # y
                None,  # valid mask
                words & 1,  # trigger value
                (words >> 8) & 0x1F,  # trigger id
            )
        if self.legacy:
            low = words & 0xFFFFFFFF
            return ((low >> 28) & 0xF, low & 0xFFFFFFF, (low >> 22) & 0x3F, (low >> 11) & 0x7FF, low & 0x7FF,
                    words >> 32, low & 1, (low >> 8) & 0x1F)
        high = words >> 32
        return ((high >> 28) & 0xF, high & 0xFFFFFFF, (high >> 22) & 0x3F, (high >> 11) & 0x7FF, high & 0x7FF,
                words & 0xFFFFFFFF, high & 1, (high >> 8) & 0x1F)

    def decode(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray, typing.Optional[int]]:
        types, high_payload, lsb, x, y, valid, trigger_value, trigger_id = self._fields(words)
        types = types.astype(np.uint8)
        is_high = types == 0x8
        if not self.started:
            first_high = np.flatnonzero(is_high)
            if len(first_high) == 0:
                return _empty_polarity(), _empty_trigger(), None
            start = first_high[0]
            self.time_high = int(high_payload[start]) << 6
            self.started = True
            types, high_payload, lsb, x, y = types[start:], high_payload[start:], lsb[start:], x[start:], y[start:]
            trigger_value, trigger_id = trigger_value[start:], trigger_id[start:]
            valid = None if valid is None else valid[start:]
            is_high = is_high[start:]

        highs = (high_payload[is_high].astype(np.int64)) << 6
        previous = np.concatenate(([self.time_high % self.LOOP], highs[:-1]))
        loops = np.cumsum(highs < previous - (self.LOOP >> 1))
        base = self.time_high - self.time_high % self.LOOP
        high_values = np.zeros(len(types), dtype=np.int64)
        high_values[is_high] = base + highs + loops * self.LOOP
        high_per_word = _forward_fill(is_high, high_values, self.time_high)
        t = high_per_word + lsb.astype(np.int64)

        is_cd = (types == 0x0) | (types == 0x1)
        positions = np.flatnonzero(is_cd)
        if valid is None:
            events_x = x[positions].astype(np.int64)
            events_positions = positions
        else:
            masks = valid[positions].astype(np.uint32)
            bits = np.unpackbits(masks.view(np.uint8).reshape(-1, 4), axis=1, bitorder="little").astype(bool)
            rows, columns = np.nonzero(bits)
            events_positions = positions[rows]
            events_x = x[events_positions].astype(np.int64) + columns
        events_y = y[events_positions].astype(np.int64)
        keep = (events_x < self.width) & (events_y < self.height)
        events = np.empty(int(np.count_nonzero(keep)), dtype=POLARITY_EVENTS_DTYPE)
        events["t"] = t[events_positions[keep]]
        events["x"] = events_x[keep]
        events["y"] = events_y[keep]
        events["on"] = types[events_positions[keep]] == 0x1

        trigger_positions = np.flatnonzero(types == 0xA)
        triggers = np.empty(len(trigger_positions), dtype=TRIGGER_EVENTS_DTYPE)
        triggers["t"] = t[trigger_positions]
        triggers["id"] = trigger_id[trigger_positions]
        triggers["rising"] = trigger_value[trigger_positions].astype(bool)

        if len(types) > 0:
            self.time_high = int(high_per_word[-1])
        current_t = int(t.max()) if len(t) > 0 else None
        return events, triggers, current_t


def _make_decoder(header: RawHeader):
    if header.format == "EVT3":
        return _Evt3Decoder(header.width, header.height)
    if header.format == "EVT2":
        return _Evt2Decoder(header.width, header.height, vectorized=False)
    return _Evt2Decoder(header.width, header.height, vectorized=True, legacy=header.endianness == "legacy")


# ------------------------------------------------------------------ reader


class RawReader(RecordingReader):
    """Reads a RAW file and delivers the same packets and windows as a live camera.

    Use opennve.open_raw() to create it. The camera configuration (name, filter,
    output, window) is applied as for a live camera; hardware settings are ignored.
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
        format: typing.Optional[str] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        chunk_bytes: int = 1 << 20,
        speed: typing.Optional[float] = None,
        zero_time: bool = False,
    ):
        path = pathlib.Path(path)
        self.header = read_header(path, format=format, width=width, height=height)
        word_size = _WORD_TYPES[self.header.format].itemsize
        self._chunk_bytes = max(word_size, chunk_bytes - chunk_bytes % word_size)
        self._file: typing.Optional[typing.BinaryIO] = open(path, "rb")
        self._file.seek(self.header.data_offset)
        self._decoder = _make_decoder(self.header)
        self._remainder = b""
        super().__init__(
            path, config, name=name, filter=filter, output=output, window=window, speed=speed, zero_time=zero_time
        )

    # Information

    @property
    def width(self) -> int:
        return self.header.width

    @property
    def height(self) -> int:
        return self.header.height

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def info(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "format": self.header.format,
            "resolution": {"width": self.header.width, "height": self.header.height},
            "serial": self.header.serial,
            "size_bytes": self.path.stat().st_size,
            "header": dict(self.header.fields),
            "filter": self.filter_config.to_dict(),
            "output": self.output_config.to_dict(),
            "window": None if self.window_config is None else self.window_config.to_dict(),
        }

    # Reading

    def _next_chunk(self) -> typing.Optional[tuple[np.ndarray, np.ndarray, typing.Optional[int]]]:
        if self._file is None:
            return None
        data = self._remainder + self._file.read(self._chunk_bytes)
        if not data:
            return None
        word_type = _WORD_TYPES[self.header.format]
        usable = len(data) - len(data) % word_type.itemsize
        self._remainder = data[usable:]
        words = np.frombuffer(data[:usable], dtype=word_type)
        return self._decoder.decode(words)

    def _close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def open_raw(
    path: PathLike,
    config: typing.Union[CameraConfig, dict, str, None] = None,
    *,
    config_path: typing.Optional[PathLike] = None,
    name: typing.Optional[str] = None,
    **kwargs,
) -> RawReader:
    """Opens a RAW file (EVT 2.0, 2.1 or 3.0).

    The optional camera configuration (CameraConfig, dict, JSON string, or config_path
    with name) provides the name, filter, output format and window duration, so that the
    file is processed exactly like the live camera.

    Keyword arguments: filter, output, window, format/width/height (for files without
    header), chunk_bytes, speed (1.0 = real-time playback, None = as fast as possible),
    zero_time (timestamps start at 0).

    Examples:
        open_raw("recording.raw")
        open_raw("recording.raw", config_path="cameras.json", name="left")
        open_raw("recording.raw", filter={"background_activity": {"enabled": True}}, speed=1.0)
    """
    if config_path is not None:
        config = load_config(config_path, name)
        name = None
    if isinstance(config, str):
        config = CameraConfig.from_json(config)
    elif isinstance(config, dict):
        config = CameraConfig.from_dict(config)
    return RawReader(path, config, name=name, **kwargs)


def read_raw(path: PathLike, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Reads a whole RAW file: (polarity events with fields t, x, y, on, trigger events).
    Keyword arguments as for open_raw. For large files, prefer open_raw(...).events()."""
    with open_raw(path, **kwargs) as raw:
        return raw.read_all()
