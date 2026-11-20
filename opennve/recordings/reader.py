"""RecordingReader: common base of the recording readers (.raw, HDF5).

A reader delivers the packets, time windows and frames of a recording exactly like a live
camera (CameraPacket, EventWindow, Frame), with the filter, output format and window of a
camera configuration, optionally paced in real time (speed).
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
import typing

import numpy as np

from ..cameras.camera import CameraPacket
from ..cameras.config import CameraConfig, PathLike
from ..events.filters import EventFilter, FilterConfig
from ..events.output import OutputConfig
from ..events.window import EventWindow, WindowAccumulator, WindowConfig
from ..frames import Frame, FrameConfig, FrameEngine, as_frame_config, as_frame_configs

POLARITY_EVENTS_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
TRIGGER_EVENTS_DTYPE = np.dtype([("t", "<u8"), ("id", "u1"), ("rising", "?")])


def _empty_polarity() -> np.ndarray:
    return np.empty(0, dtype=POLARITY_EVENTS_DTYPE)


def _empty_trigger() -> np.ndarray:
    return np.empty(0, dtype=TRIGGER_EVENTS_DTYPE)


class RecordingReader:
    """Base class of the recording readers (RawReader, Hdf5Reader): delivers the same packets,
    windows and frames as a live camera.

    Subclasses implement width, height, info(), _next_chunk() (events, trigger events,
    current time; None at the end) and _close().
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
        speed: typing.Optional[float] = None,
        zero_time: bool = False,
    ):
        self.path = pathlib.Path(path)
        self.name = name or (config.name if config is not None else None) or self.path.stem
        self.filter_config = _as_filter(filter if filter is not None else (config.filter if config else None))
        self.output_config = _as_output(output if output is not None else (config.output if config else None))
        if isinstance(window, str) and window == "config":
            self.window_config = config.window if config is not None else WindowConfig()
        else:
            self.window_config = WindowConfig.from_dict(window) if isinstance(window, dict) else window
        self._speed: typing.Optional[float] = None
        self._playback_start: typing.Optional[tuple[float, int]] = None
        self.speed = speed
        self.zero_time = zero_time
        self._filter = (
            EventFilter(self.filter_config, self.width, self.height) if self.filter_config.is_active else None
        )
        self._accumulator: typing.Optional[WindowAccumulator] = None
        self._time_origin: typing.Optional[int] = None
        self.finished = False

    @property
    def speed(self) -> typing.Optional[float]:
        """Playback speed: 1.0 = real time, 2.0 = twice as fast, None = as fast as possible.
        Can be changed while reading."""
        return self._speed

    @speed.setter
    def speed(self, value: typing.Optional[float]):
        if value is not None and value <= 0:
            raise ValueError("speed must be positive (1.0 = real time) or None (as fast as possible)")
        self._speed = value
        self._playback_start = None

    # Information (implemented by the subclasses)

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    def info(self) -> dict[str, typing.Any]:
        raise NotImplementedError

    def _next_chunk(self) -> typing.Optional[tuple[np.ndarray, np.ndarray, typing.Optional[int]]]:
        """Next decoded chunk: (polarity events, trigger events, current time or None); None at the end."""
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    # Reading

    def _read_chunk(self) -> typing.Optional[tuple[np.ndarray, np.ndarray, typing.Optional[int]]]:
        if not self.is_open or self.finished:
            return None
        chunk = self._next_chunk()
        if chunk is None:
            self.finished = True
            return None
        events, triggers, current_t = chunk
        if self.zero_time:
            if self._time_origin is None:
                candidates = [int(array["t"][0]) for array in (events, triggers) if len(array) > 0]
                if current_t is not None or candidates:
                    self._time_origin = min(candidates) if candidates else int(current_t)
            if self._time_origin is not None:
                events["t"] -= np.uint64(self._time_origin)
                triggers["t"] -= np.uint64(self._time_origin)
                if current_t is not None:
                    current_t -= self._time_origin
        return events, triggers, current_t

    def _pace(self, current_t: typing.Optional[int]):
        if self.speed is None or current_t is None:
            return
        if self._playback_start is None:
            self._playback_start = (time.monotonic(), current_t)
            return
        wall_start, t_start = self._playback_start
        delay = wall_start + (current_t - t_start) / 1e6 / self.speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def read(self) -> typing.Optional[CameraPacket]:
        """Reads the next chunk of the file. Returns None at the end of the file."""
        while True:
            chunk = self._read_chunk()
            if chunk is None:
                return None
            events, triggers, current_t = chunk
            if current_t is None and len(events) == 0:
                continue  # data before the first time reference
            self._pace(current_t)
            raw_event_count = len(events)
            if self._filter is not None:
                events = self._filter(events)
            return CameraPacket(
                camera=self.name,
                status=_RawStatus(current_t),
                polarity_events=events,
                trigger_events=triggers if len(triggers) > 0 else None,
                raw_event_count=raw_event_count,
                packet=None,
                output_config=self.output_config,
            )

    def events(self) -> typing.Iterator[CameraPacket]:
        """Yields the packets of the file until its end (filtered, in the configured output format)."""
        while True:
            packet = self.read()
            if packet is None:
                return
            yield packet

    def read_windows(self, duration_ms: typing.Optional[float] = None) -> typing.Optional[list[EventWindow]]:
        """Reads the next chunk and returns the windows it completed (often []).
        Returns None at the end of the file (after the last, possibly shorter, window)."""
        accumulator = self._window_accumulator(duration_ms)
        packet = self.read()
        if packet is None:
            windows = accumulator.flush()
            return windows if windows else None
        return accumulator.push(packet.polarity_events, packet.status.ring.current_t)

    def windows(self, duration_ms: typing.Optional[float] = None) -> typing.Iterator[EventWindow]:
        """Yields the time windows of the file. The last window is returned as well, also if
        the file ends before its end time."""
        while True:
            windows = self.read_windows(duration_ms)
            if windows is None:
                return
            yield from windows

    def _window_accumulator(self, duration_ms: typing.Optional[float]) -> WindowAccumulator:
        if duration_ms is None:
            if self.window_config is None:
                raise ValueError("no window duration: pass duration_ms or a \"window\" configuration")
            duration_us = self.window_config.duration_us
        else:
            window = WindowConfig(duration_ms=duration_ms)
            window.validate()
            duration_us = window.duration_us
        if self._accumulator is None or self._accumulator.duration_us != duration_us:
            self._accumulator = WindowAccumulator(
                duration_us, self.name, self.width, self.height, self.output_config
            )
        return self._accumulator

    def read_frames(
        self,
        config: typing.Union[FrameConfig, dict, str, None] = None,
        duration_ms: typing.Optional[float] = None,
    ) -> typing.Optional[list[Frame]]:
        """Frames of the windows completed by the next chunk (often []), None at the end of the file.
        config: frame configuration (see frames())."""
        engine = self._frame_engine(config)
        windows = self.read_windows(duration_ms)
        if windows is None:
            return None
        return [engine.build_window(window) for window in windows]

    def frames(
        self,
        config: typing.Union[FrameConfig, dict, str, None] = None,
        duration_ms: typing.Optional[float] = None,
    ) -> typing.Iterator[Frame]:
        """Yields one frame per time window of the file (frame.data, frame.tiles, frame.events).

        config: frame configuration, for instance opennve.load_frame_config("frames.json", "tiles80"),
            a FrameConfig, a dict or a JSON string. Default: sensor size, linear time surface, no tiles.
        """
        engine = self._frame_engine(config)
        for window in self.windows(duration_ms):
            yield engine.build_window(window)

    def read_frame_sets(
        self, configs, duration_ms: typing.Optional[float] = None
    ) -> typing.Optional[list[dict[str, Frame]]]:
        """Like read_frames with several frame configurations: one {name: Frame} per completed window."""
        engines = self._frame_engines(configs)
        windows = self.read_windows(duration_ms)
        if windows is None:
            return None
        return [{name: engine.build_window(window) for name, engine in engines.items()} for window in windows]

    def frame_sets(self, configs, duration_ms: typing.Optional[float] = None) -> typing.Iterator[dict[str, Frame]]:
        """Yields one frame per frame configuration and time window: {name: Frame}.
        configs: list of frame configurations (for instance load_frame_configs("frames.json"))."""
        engines = self._frame_engines(configs)
        for window in self.windows(duration_ms):
            yield {name: engine.build_window(window) for name, engine in engines.items()}

    def _frame_engine(self, config) -> FrameEngine:
        return FrameEngine.from_config(as_frame_config(config), self.width, self.height)

    def _frame_engines(self, configs) -> dict[str, FrameEngine]:
        return {
            config.name: FrameEngine.from_config(config, self.width, self.height)
            for config in as_frame_configs(configs)
        }

    def read_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Reads the rest of the file at once: (polarity events, trigger events), filtered.
        For large files, prefer events() or windows()."""
        polarity, triggers = [], [_empty_trigger()]
        for packet in self.events():
            polarity.append(packet.polarity_events)
            if packet.trigger_events is not None:
                triggers.append(packet.trigger_events)
        return (np.concatenate(polarity) if polarity else _empty_polarity()), np.concatenate(triggers)

    # Lifecycle

    def close(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


@dataclasses.dataclass
class _RingStatus:
    current_t: typing.Optional[int]
    backlog: int = 0


class _RawStatus:
    """Mimics the neuromorphic_drivers status (status.ring.current_t) for RAW files."""

    def __init__(self, current_t: typing.Optional[int]):
        self.ring = _RingStatus(current_t)


def _as_filter(value) -> FilterConfig:
    if value is None:
        return FilterConfig()
    if isinstance(value, dict):
        return FilterConfig.from_dict(value)
    value.validate()
    return value


def _as_output(value) -> OutputConfig:
    if value is None:
        return OutputConfig()
    if isinstance(value, dict):
        return OutputConfig.from_dict(value)
    value.validate()
    return value
