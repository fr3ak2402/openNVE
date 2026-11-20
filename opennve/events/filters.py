"""Software event filters, configured with the "filter" section of a camera configuration.

JSON example:

    "filter": {
        "polarity": "both",
        "background_activity": {"enabled": true, "time_us": 5000, "min_neighbors": 1}
    }
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

POLARITIES = ("both", "on", "off")


@dataclasses.dataclass
class BackgroundActivityConfig:
    """Removes isolated events (noise, hot pixels).

    An event is kept only if at least min_neighbors of its 8 neighbor pixels also had
    an event within the last time_us microseconds. Real edges trigger neighboring pixels
    at almost the same time, noise does not. Smaller time_us / larger min_neighbors
    filter more.
    """

    enabled: bool = False
    time_us: int = 5_000
    min_neighbors: int = 1

    def validate(self):
        if not 1 <= self.time_us <= 1_000_000:
            raise ValueError("filter.background_activity.time_us must be in [1, 1000000]")
        if not 1 <= self.min_neighbors <= 8:
            raise ValueError("filter.background_activity.min_neighbors must be in [1, 8]")


@dataclasses.dataclass
class FilterConfig:
    """Software filter settings of a camera."""

    polarity: str = "both"
    background_activity: BackgroundActivityConfig = dataclasses.field(
        default_factory=BackgroundActivityConfig
    )

    def validate(self):
        if self.polarity not in POLARITIES:
            raise ValueError(f"filter.polarity must be one of {', '.join(POLARITIES)}")
        self.background_activity.validate()

    @property
    def is_active(self) -> bool:
        return self.polarity != "both" or self.background_activity.enabled

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: typing.Optional[dict[str, typing.Any]]) -> "FilterConfig":
        data = dict(data or {})
        unknown = set(data) - {"polarity", "background_activity"}
        if unknown:
            raise ValueError(f"unknown filter settings: {', '.join(sorted(unknown))}")
        background_activity = data.get("background_activity") or {}
        unknown = set(background_activity) - {f.name for f in dataclasses.fields(BackgroundActivityConfig)}
        if unknown:
            raise ValueError(
                f"unknown filter.background_activity settings: {', '.join(sorted(unknown))}"
            )
        result = cls(
            polarity=str(data.get("polarity", "both")).lower(),
            background_activity=BackgroundActivityConfig(**background_activity),
        )
        result.validate()
        return result


class BackgroundActivityFilter:
    """Vectorized background activity filter for polarity events (fields t, x, y, on)."""

    _NEIGHBORS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]

    def __init__(self, width: int, height: int, time_us: int, min_neighbors: int = 1):
        self.time_us = time_us
        self.min_neighbors = min_neighbors
        # Latest event timestamp per pixel, with a 1-pixel border to avoid bounds checks
        self.last_t = np.full((height + 2, width + 2), -(2**62), dtype=np.int64)

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if len(events) == 0:
            return events
        t = events["t"].astype(np.int64)
        x = events["x"].astype(np.intp) + 1
        y = events["y"].astype(np.intp) + 1
        # Store this packet first so that neighbors within the same packet support each other
        # (events are sorted by time, the last write per pixel wins)
        self.last_t[y, x] = t
        threshold = t - self.time_us
        support = np.zeros(len(events), dtype=np.uint8)
        for dx, dy in self._NEIGHBORS:
            support += self.last_t[y + dy, x + dx] >= threshold
        return events[support >= self.min_neighbors]


class EventFilter:
    """Applies a FilterConfig to polarity events."""

    def __init__(self, config: FilterConfig, width: int, height: int):
        self.config = config
        self._background_activity = (
            BackgroundActivityFilter(
                width,
                height,
                config.background_activity.time_us,
                config.background_activity.min_neighbors,
            )
            if config.background_activity.enabled
            else None
        )

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if self._background_activity is not None:
            events = self._background_activity(events)
        if self.config.polarity == "on":
            events = events[events["on"]]
        elif self.config.polarity == "off":
            events = events[~events["on"]]
        return events
