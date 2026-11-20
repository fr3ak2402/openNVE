"""Workflow steps: sources (camera, raw), processing steps and sinks (write, collect).

Every step is a class registered with register_step("type"). A workflow configuration lists the
steps in order: {"type": "frames", "frame": "tiles80", "threads": 2}. Steps with "threads" > 1
run in several threads; each thread has its own step instance (for instance its own model).

Own steps:

    @opennve.register_step("my_step")
    class MyStep(opennve.Step):
        PARAMETERS = {"factor": 1.0}          # allowed parameters and their defaults

        def setup(self):                      # once per thread, for instance loading a model
            ...

        def process(self, item: opennve.WorkItem):
            item.data["my_value"] = len(item.block) * self.params["factor"]
            return item                       # None drops the item

Modules: base (Step, StepContext, register_step), sources (camera, raw), processing (windows,
frames, tiles, agent, require, label_events, log, callback), sinks (write, collect).
"""

from .base import COMMON_PARAMETERS, Step, StepContext, get_step_class, list_steps, register_step
from .processing import (
    AgentStep,
    CallbackStep,
    FramesStep,
    LabelEventsStep,
    LogStep,
    RequireStep,
    TilesStep,
    WindowsStep,
    sub_window,
    window_bounds,
)
from .sinks import CollectStep, WriteStep
from .sources import CameraSource, RawSource

__all__ = [
    "COMMON_PARAMETERS",
    "AgentStep",
    "CallbackStep",
    "CameraSource",
    "CollectStep",
    "FramesStep",
    "LabelEventsStep",
    "LogStep",
    "RawSource",
    "RequireStep",
    "Step",
    "StepContext",
    "TilesStep",
    "WindowsStep",
    "WriteStep",
    "get_step_class",
    "list_steps",
    "register_step",
    "sub_window",
    "window_bounds",
]
