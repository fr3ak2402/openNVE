"""Step base class, step registry and StepContext."""

from __future__ import annotations

import datetime
import os
import pathlib
import threading
import typing

from ...frames import FrameConfig, as_frame_config
from ..item import WorkItem


_STEPS: dict[str, type["Step"]] = {}

COMMON_PARAMETERS = ("type", "name", "threads")


def register_step(name: str, step: typing.Optional[type] = None):
    """Registers a step class under a type name (usable as a decorator)."""

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, Step)):
            raise TypeError("a step must be a subclass of opennve.Step")
        cls.TYPE = name
        _STEPS[name] = cls
        return cls

    return decorator if step is None else decorator(step)


def list_steps() -> list[str]:
    return sorted(_STEPS)


def get_step_class(name: str) -> type["Step"]:
    if name not in _STEPS:
        raise ValueError(f"unknown step type {name!r} (registered: {', '.join(list_steps())})")
    return _STEPS[name]


class StepContext:
    """What steps can use: the workflow name, the configuration files (loaded on first use),
    the start time and a lock for printing."""

    def __init__(
        self,
        workflow: str = "workflow",
        base_dir: typing.Optional[pathlib.Path] = None,
        cameras=None,
        frames=None,
        agents=None,
        debug: bool = False,
    ):
        self.workflow = workflow
        self.base_dir = base_dir or pathlib.Path.cwd()
        self._cameras = cameras
        self._frames = frames
        self._agents = agents
        self.debug = debug
        self.started = datetime.datetime.now()
        self.print_lock = threading.Lock()

    def path(self, value: typing.Union[str, os.PathLike]) -> pathlib.Path:
        path = pathlib.Path(value).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    def log(self, message: str):
        with self.print_lock:
            print(message, flush=True)

    # configurations

    def camera_configs(self):
        from ...cameras.config import CameraConfig, load_configs

        if self._cameras is None:
            raise ValueError("this step needs camera configurations: set \"cameras\" in the workflow")
        if isinstance(self._cameras, (str, os.PathLike)):
            self._cameras = load_configs(self.path(self._cameras))
        return [c if isinstance(c, CameraConfig) else CameraConfig.from_dict(c) for c in self._cameras]

    def frame_configs(self) -> dict[str, FrameConfig]:
        from ...frames import as_frame_configs, load_frame_configs

        if self._frames is None:
            return {}
        if isinstance(self._frames, (str, os.PathLike)):
            self._frames = load_frame_configs(self.path(self._frames))
        return {config.name: config for config in as_frame_configs(self._frames)}

    def frame_config(self, value) -> FrameConfig:
        """A frame configuration by name (from "frames") or inline (dict)."""
        if isinstance(value, str):
            configs = self.frame_configs()
            if value not in configs:
                raise ValueError(f"no frame configuration named {value!r} (available: {', '.join(configs) or 'none'})")
            return configs[value]
        config = as_frame_config(value)
        if config.name is None:
            config = config.copy(name="frame")
        return config

    def agent_config(self, value):
        """An agent configuration by name (from "agents") or inline (dict)."""
        from ...agents import AgentConfig, agent_configs_from_data, load_agent_configs

        if isinstance(value, dict):
            return AgentConfig.from_dict(value, self.base_dir)
        if self._agents is None:
            raise ValueError("this step needs agent configurations: set \"agents\" in the workflow")
        if isinstance(self._agents, (str, os.PathLike)):
            self._agents = load_agent_configs(self.path(self._agents))
        elif isinstance(self._agents, (list, dict)) and not all(isinstance(a, AgentConfig) for a in self._agents):
            self._agents = agent_configs_from_data(self._agents, "agents", self.base_dir)
        for config in self._agents:
            if config.name == value:
                return config
        names = ", ".join(config.name for config in self._agents)
        raise ValueError(f"no agent named {value!r} (available: {names})")


class Step:
    """Base class of all steps.

    kind: "source" (creates items: items()), "process" (process(item) -> item or None) or
        "sink" (process(item), at the end of the workflow).
    PARAMETERS: allowed parameters with their defaults (plus "type", "name", "threads").
    DEFAULT_THREADS: threads without "threads" in the step configuration: a number or "auto".
    HEAVY: steps that keep the CPU cores busy (agents); the workflow splits the cores between them.

    Automatic threads ("threads": "auto"): auto_threads(cpus) gives the number of threads,
    thread_plan(threads, cpus) further settings (for instance the interpreter threads of the
    agents), which every instance gets through configure_threads(plan) before setup().
    """

    TYPE = "step"
    kind = "process"
    PARAMETERS: dict[str, typing.Any] = {}
    MAX_THREADS: typing.Optional[int] = None
    ORDERED = False  # True: gets the items in their original order (like the sinks) if the workflow is "ordered"
    DEFAULT_THREADS: typing.Union[int, str] = 1
    HEAVY = False
    thread_index = 0  # set by the workflow: which thread of the step this instance runs in

    def __init__(self, context: StepContext, **params):
        unknown = set(params) - set(self.PARAMETERS) - set(COMMON_PARAMETERS)
        if unknown:
            raise ValueError(
                f"step {self.TYPE}: unknown parameter(s) {', '.join(sorted(unknown))} "
                f"(supported: {', '.join(sorted(self.PARAMETERS)) or 'none'})"
            )
        self.context = context
        self.params = {**self.PARAMETERS, **{k: v for k, v in params.items() if k in self.PARAMETERS}}
        self.name = params.get("name") or self.TYPE
        self.validate()

    def validate(self):
        """Checks the parameters (called when the workflow is created)."""

    def auto_threads(self, cpus: int) -> int:
        """Threads for "threads": "auto" (cpus: cores available to this step)."""
        return 1

    def thread_plan(self, threads: int, cpus: int) -> dict[str, typing.Any]:
        """Further settings for the given number of threads (shown by Workflow.thread_plan())."""
        return {}

    def configure_threads(self, plan: dict[str, typing.Any]):
        """Applies thread_plan() to this instance (called in each thread before setup();
        self.thread_index tells which thread)."""

    def statistics_details(self) -> dict[str, typing.Any]:
        """Extra numbers for the workflow statistics (added up over the threads of the step)."""
        return {}

    def setup(self):
        """Called once in each thread before the first item (load models here)."""

    def process(self, item: WorkItem):
        return item

    def items(self, stop: threading.Event) -> typing.Iterator[WorkItem]:  # sources
        raise NotImplementedError

    def close(self):
        """Called once in each thread after the last item."""

    def describe(self) -> str:
        values = ", ".join(f"{key}={value!r}" for key, value in self.params.items() if value is not None)
        return f"{self.TYPE}({values})"


def _positive_number(step: Step, name: str, allow_none: bool = True, integer: bool = False):
    value = step.params[name]
    if value is None and allow_none:
        return
    valid_type = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, valid_type) or value <= 0:
        kind = "a positive integer" if integer else "a positive number"
        raise ValueError(f"step {step.TYPE}: {name} must be {kind}" + (" or null" if allow_none else ""))
