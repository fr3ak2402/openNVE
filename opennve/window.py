"""Time windows: collect the events of a fixed duration, for instance to build frames.

JSON example (camera configuration):

    "window": {"duration_ms": 40}

Default: 40 ms (also if the "window" section is missing). "window": null disables windows.

Windows are aligned to the camera clock: with 40 ms, windows are [0, 40) ms, [40, 80) ms, ...
of the camera timestamps, so synchronized cameras produce windows with the same boundaries.
A window is complete as soon as the camera time passes its end, also if there are no events
(empty windows are produced for periods without events).
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from .frames import Frame, FrameConfig, FrameEngine
from .output import OutputConfig, to_csv, to_rows

FRAME_MODES = ("count", "polarity", "binary", "color")

DEFAULT_DURATION_MS = 40.0

# Periods without events longer than this many windows are skipped instead of producing empty windows
MAXIMUM_EMPTY_WINDOWS = 10_000


@dataclasses.dataclass
class WindowConfig:
    """Duration of the time windows, in milliseconds (0.01 ms to 60 s)."""

    duration_ms: float = DEFAULT_DURATION_MS

    def validate(self):
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise ValueError("window.duration_ms must be a number")
        if not 0.01 <= self.duration_ms <= 60_000:
            raise ValueError("window.duration_ms must be in [0.01, 60000]")

    @property
    def duration_us(self) -> int:
        return max(1, round(self.duration_ms * 1000))

    def to_dict(self) -> dict[str, typing.Any]:
        return {"duration_ms": self.duration_ms}

    @classmethod
    def from_dict(cls, data: typing.Optional[dict[str, typing.Any]]) -> typing.Optional["WindowConfig"]:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("window must be an object, for instance {\"duration_ms\": 40}, or null")
        unknown = set(data) - {"duration_ms"}
        if unknown:
            raise ValueError(f"unknown window setting(s): {', '.join(sorted(unknown))}")
        result = cls(**data)
        result.validate()
        return result


@dataclasses.dataclass
class EventWindow:
    """The events of one time window of one camera.

    Attributes:
        camera: Camera name.
        start_t: Window start (camera time, µs, included).
        end_t: Window end (camera time, µs, excluded).
        width, height: Sensor size in pixels.
        polarity_events: Filtered events of the window (fields t, x, y, on).
        output_config: Output format of the camera configuration.
    """

    camera: str
    start_t: int
    end_t: int
    width: int
    height: int
    polarity_events: np.ndarray
    output_config: OutputConfig = dataclasses.field(default_factory=OutputConfig, repr=False)
    frame_config: FrameConfig = dataclasses.field(default_factory=FrameConfig, repr=False)

    def frame(
        self,
        config: typing.Union[FrameConfig, dict, None] = None,
        mode: typing.Optional[str] = None,
        decay: typing.Optional[str] = None,
    ) -> Frame:
        """Builds the frame (and tiles) of this window with the frame engine.

        config: FrameConfig or dict, default: the "frame" section of the camera configuration.
        mode: overrides the mode ("time_surface", "binary" or "count").
        decay: overrides the time surface decay ("linear" or "exponential").
        Returns a Frame: frame.data (height x width x 2, float32), frame.tiles, frame.events.
        """
        if config is None:
            config = self.frame_config
        elif isinstance(config, dict):
            config = FrameConfig.from_dict(config)
        return FrameEngine.from_config(config, self.width, self.height).build_window(self, mode=mode, decay=decay)

    @property
    def duration_ms(self) -> float:
        return (self.end_t - self.start_t) / 1000

    @property
    def events(self) -> np.ndarray:
        """Events in the configured output format (for instance fields x, y, p, t)."""
        return self.output_config.format(self.polarity_events)

    def __len__(self) -> int:
        return len(self.polarity_events)

    def rows(self) -> list[tuple]:
        return to_rows(self.events)

    def to_csv(self, header: bool = False, delimiter: str = ",", camera: bool = False) -> str:
        return to_csv(self.events, header=header, delimiter=delimiter, camera=self.camera if camera else None)

    def to_frame(self, mode: str = "count") -> np.ndarray:
        """Builds an image (height x width) from the events of the window.

        Modes:
            count:    number of events per pixel (uint16)
            polarity: ON events minus OFF events per pixel (int16)
            binary:   255 where at least one event occurred, else 0 (uint8)
            color:    BGR image (uint8), ON = green, OFF = red (latest event wins)
        """
        events = self.polarity_events
        x = events["x"].astype(np.intp)
        y = events["y"].astype(np.intp)
        size = self.width * self.height
        if mode == "count":
            counts = np.bincount(y * self.width + x, minlength=size)
            return np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16).reshape(self.height, self.width)
        if mode == "polarity":
            sums = np.bincount(y * self.width + x, weights=np.where(events["on"], 1, -1), minlength=size)
            limits = np.iinfo(np.int16)
            return np.clip(sums, limits.min, limits.max).astype(np.int16).reshape(self.height, self.width)
        if mode == "binary":
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
            frame[y, x] = 255
            return frame
        if mode == "color":
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            on = events["on"]
            frame[y[on], x[on]] = (0, 255, 0)
            frame[y[~on], x[~on]] = (0, 0, 255)
            return frame
        raise ValueError(f"unknown frame mode {mode!r}, supported: {', '.join(FRAME_MODES)}")


class WindowAccumulator:
    """Splits a stream of polarity events into consecutive time windows."""

    def __init__(
        self,
        duration_us: int,
        camera: str,
        width: int,
        height: int,
        output_config: typing.Optional[OutputConfig] = None,
        frame_config: typing.Optional[FrameConfig] = None,
    ):
        self.duration_us = duration_us
        self.frame_config = frame_config or FrameConfig()
        self.camera = camera
        self.width = width
        self.height = height
        self.output_config = output_config or OutputConfig()
        self._start_t: typing.Optional[int] = None
        self._chunks: list[np.ndarray] = []
        self._empty: typing.Optional[np.ndarray] = None

    def reset(self):
        self._start_t = None
        self._chunks = []

    def _close(self) -> EventWindow:
        assert self._start_t is not None
        if self._chunks:
            events = self._chunks[0] if len(self._chunks) == 1 else np.concatenate(self._chunks)
        else:
            events = self._empty
        window = EventWindow(
            camera=self.camera,
            start_t=self._start_t,
            end_t=self._start_t + self.duration_us,
            width=self.width,
            height=self.height,
            polarity_events=events,
            output_config=self.output_config,
            frame_config=self.frame_config,
        )
        self._chunks = []
        self._start_t += self.duration_us
        return window

    def _align(self, t: int) -> int:
        return (t // self.duration_us) * self.duration_us

    def flush(self) -> list[EventWindow]:
        """Returns the current window if it has events (for instance at the end of a file)."""
        if self._start_t is None or not self._chunks:
            return []
        return [self._close()]

    def push(
        self, events: typing.Optional[np.ndarray], current_t: typing.Optional[int] = None
    ) -> list[EventWindow]:
        """Adds events (sorted by t) and the current camera time, returns the completed windows."""
        windows: list[EventWindow] = []
        if events is not None:
            if self._empty is None:
                self._empty = events[:0]
            if len(events) > 0:
                first_t = int(events["t"][0])
                if self._start_t is None or first_t < self._start_t:
                    # First events, or the camera clock restarted
                    self._chunks = []
                    self._start_t = self._align(first_t)
                # Skip long periods without events
                if (
                    not self._chunks
                    and first_t >= self._start_t + MAXIMUM_EMPTY_WINDOWS * self.duration_us
                ):
                    self._start_t = self._align(first_t)
                t = events["t"]
                index = 0
                while index < len(events):
                    end_t = self._start_t + self.duration_us
                    split = int(np.searchsorted(t, end_t, side="left"))
                    if split > index:
                        self._chunks.append(events[index:split])
                        index = split
                    if index < len(events):
                        windows.append(self._close())
                        # Skip long periods without events
                        next_t = int(t[index])
                        if next_t >= self._start_t + MAXIMUM_EMPTY_WINDOWS * self.duration_us:
                            self._start_t = self._align(next_t)
        if current_t is not None and self._start_t is not None and self._empty is not None:
            if current_t >= self._start_t + MAXIMUM_EMPTY_WINDOWS * self.duration_us:
                if self._chunks:
                    windows.append(self._close())
                self._start_t = self._align(current_t)
            while current_t >= self._start_t + self.duration_us:
                windows.append(self._close())
        return windows
