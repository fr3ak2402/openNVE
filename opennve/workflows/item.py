"""WorkItem: the data that travels through the steps of a workflow."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ..agents import AgentResult
from ..frames import Frame
from ..window import EventWindow


@dataclasses.dataclass
class WorkItem:
    """One block of events and everything the steps derived from it.

    Attributes:
        index: Sequence number (assigned by the source, 0, 1, 2, ... over all cameras).
        camera: Camera name.
        block: The events of the block (EventWindow, for instance 1000 ms).
        windows: Time windows of the block (step "windows"; default: the block itself).
        frames: One {frame configuration name: Frame} per window (step "frames").
        results: One {agent name: AgentResult} per window (step "agent").
        events: Output events (structured array; step "label_events" adds the label field).
        data: Free storage for custom steps.
        timings: Seconds spent in each step.
    """

    index: int
    camera: str
    block: EventWindow
    windows: list[EventWindow] = dataclasses.field(default_factory=list)
    frames: list[dict[str, Frame]] = dataclasses.field(default_factory=list)
    results: list[dict[str, AgentResult]] = dataclasses.field(default_factory=list)
    events: typing.Optional[np.ndarray] = None
    data: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    timings: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.windows:
            self.windows = [self.block]

    @property
    def start_t(self) -> int:
        return self.block.start_t

    @property
    def end_t(self) -> int:
        return self.block.end_t

    @property
    def event_count(self) -> int:
        return len(self.block)

    def positive_windows(self, agent: str) -> list[int]:
        """Indices of the windows in which the agent selected at least one prediction."""
        return [
            index for index, results in enumerate(self.results) if agent in results and results[agent].selected
        ]

    def summary(self) -> dict[str, typing.Any]:
        """JSON-friendly summary: block, windows, selected predictions per agent and window."""
        return {
            "index": self.index,
            "camera": self.camera,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "events": self.event_count,
            "windows": [
                {
                    "start_t": window.start_t,
                    "end_t": window.end_t,
                    "events": len(window),
                    "agents": {
                        name: result.to_dict(selected_only=True)
                        for name, result in (self.results[index] if index < len(self.results) else {}).items()
                    },
                }
                for index, window in enumerate(self.windows)
            ],
            "output_events": None if self.events is None else int(len(self.events)),
            "timings": {name: round(seconds, 6) for name, seconds in self.timings.items()},
        }
