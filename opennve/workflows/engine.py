"""Workflows: freely configurable chains of steps, run in threads.

A workflow configuration (workflows.json) lists named workflows. Each one names the camera,
frame and agent configuration files it uses and its steps, in order:

    {
        "workflows": [
            {
                "name": "insects",
                "default": true,
                "cameras": "cameras.json",
                "frames": "frames.json",
                "agents": "agents.json",
                "ordered": true,
                "steps": [
                    {"type": "camera", "cameras": ["left"], "block_ms": 1000},
                    {"type": "windows", "count": 3, "overlap_ms": 200},
                    {"type": "frames", "frame": "tiles80", "min_events_per_ms": 0.15},
                    {"type": "agent", "agent": "insects"},
                    {"type": "require", "agent": "insects", "min_windows": 1},
                    {"type": "label_events", "agent": "insects", "label": 2},
                    {"type": "write", "format": "csv", "path": "output/{workflow}_{camera}_{date}_{time}.csv"}
                ]
            }
        ]
    }

Every step runs in its own thread(s) ("threads"), connected by queues: while the agent
evaluates block 1, the source already reads block 2 and the writer saves block 0. With
"ordered": true the sink receives the items in their original order.

Threads: "threads": "auto" (the default of the frames, tiles and agent steps) sizes a step to the
CPU cores of the machine (opennve.cpu_count(), or the workflow's "cpus"): the agent steps share
all cores (one TensorFlow Lite interpreter per thread), frames and tiles get 1-2 threads. A number
fixes the threads. workflow.thread_report() shows the result.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import queue
import threading
import time
import typing

from ..system import cpu_count
from .item import WorkItem
from .steps import COMMON_PARAMETERS, CollectStep, Step, StepContext, get_step_class

PathLike = typing.Union[str, pathlib.Path]


class _Stop:
    """End marker passed through the queues."""


class _Skip:
    """Marker for an item that a step dropped (keeps the order for "ordered")."""

    def __init__(self, index: int):
        self.index = index


_STOP = _Stop()


# ------------------------------------------------------------------ configuration


@dataclasses.dataclass
class WorkflowConfig:
    """Configuration of one workflow (see the module documentation)."""

    name: str
    steps: list[dict[str, typing.Any]]
    default: bool = False
    description: str = ""
    cameras: typing.Any = None
    frames: typing.Any = None
    agents: typing.Any = None
    queue_size: typing.Optional[int] = None
    ordered: bool = True
    debug: bool = False
    cpus: typing.Union[int, str, None] = None
    base_dir: typing.Optional[pathlib.Path] = dataclasses.field(default=None, compare=False, repr=False)

    FIELDS = (
        "name", "steps", "default", "description", "cameras", "frames", "agents", "queue_size", "ordered", "debug",
        "cpus",
    )

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workflow.name must be a non-empty string")
        if not isinstance(self.steps, list) or not self.steps:
            raise ValueError(f"workflow {self.name}: steps must be a non-empty list")
        for index, step in enumerate(self.steps):
            if not isinstance(step, dict) or not isinstance(step.get("type"), str):
                raise ValueError(f"workflow {self.name}: step {index} must be an object with \"type\"")
        if self.queue_size is not None and (
            isinstance(self.queue_size, bool) or not isinstance(self.queue_size, int) or self.queue_size <= 0
        ):
            raise ValueError(f"workflow {self.name}: queue_size must be a positive integer or null (automatic)")
        if self.cpus is not None and self.cpus != "auto" and (
            isinstance(self.cpus, bool) or not isinstance(self.cpus, int) or self.cpus <= 0
        ):
            raise ValueError(f"workflow {self.name}: cpus must be a positive integer, \"auto\" or null")
        for key in ("default", "ordered", "debug"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"workflow {self.name}: {key} must be true or false")

    def to_dict(self) -> dict[str, typing.Any]:
        return {key: getattr(self, key) for key in self.FIELDS if key != "steps"} | {"steps": self.steps}

    @classmethod
    def from_dict(cls, data: dict, base_dir: typing.Optional[PathLike] = None) -> "WorkflowConfig":
        if not isinstance(data, dict):
            raise ValueError("a workflow must be an object")
        unknown = set(data) - set(cls.FIELDS)
        if unknown:
            raise ValueError(f"unknown workflow setting(s): {', '.join(sorted(unknown))} (supported: {', '.join(cls.FIELDS)})")
        if "name" not in data or "steps" not in data:
            raise ValueError("a workflow needs \"name\" and \"steps\"")
        config = cls(**data, base_dir=None if base_dir is None else pathlib.Path(base_dir))
        config.validate()
        return config


def workflow_configs_from_data(data, source: str = "workflow configuration", base_dir=None) -> list[WorkflowConfig]:
    if isinstance(data, dict) and "workflows" in data:
        data = data["workflows"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected a workflow object, a list or {{\"workflows\": [...]}}")
    configs = []
    for index, item in enumerate(data):
        try:
            configs.append(WorkflowConfig.from_dict(item, base_dir))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{source}: workflow {index}: {error}") from error
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate workflow name(s): {', '.join(duplicates)}")
    if sum(config.default for config in configs) > 1:
        raise ValueError(f"{source}: more than one workflow has \"default\": true")
    return configs


def load_workflow_configs(path: PathLike) -> list[WorkflowConfig]:
    """Workflow configurations of a file. Relative paths (configuration files, outputs) are
    relative to the file."""
    path = pathlib.Path(path)
    return workflow_configs_from_data(json.loads(path.read_text(encoding="utf-8")), str(path), path.resolve().parent)


def load_workflow_config(path: PathLike, name: typing.Optional[str] = None) -> WorkflowConfig:
    """One workflow by name; without a name the default workflow (or the only one)."""
    configs = load_workflow_configs(path)
    if name is None:
        defaults = [config for config in configs if config.default]
        if defaults:
            return defaults[0]
        if len(configs) == 1:
            return configs[0]
        raise ValueError(f"{path} contains {len(configs)} workflows and none is the default, specify a name")
    for config in configs:
        if config.name == name:
            return config
    raise ValueError(f"{path}: no workflow named {name!r} (available: {', '.join(c.name for c in configs)})")


def load_workflow(path: PathLike, name: typing.Optional[str] = None) -> "Workflow":
    """The workflow (by name, else the default) of a workflow configuration file."""
    return Workflow(load_workflow_config(path, name))



# ------------------------------------------------------------------ statistics


@dataclasses.dataclass
class StepStatistics:
    name: str
    threads: int
    items: int = 0
    dropped: int = 0
    seconds: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.seconds / self.items * 1000 if self.items else 0.0


# ------------------------------------------------------------------ workflow


class _Stage:
    def __init__(self, factory: typing.Callable[[], Step], config: dict, requested, kind: str, name: str, prototype: Step):
        self._factory = factory
        self.config = config
        self.requested = requested  # number or "auto"
        self.threads = requested if isinstance(requested, int) else 1
        self.plan: dict[str, typing.Any] = {}
        self.prototype = prototype
        self.kind = kind
        self.name = name
        self.instances: list[Step] = []
        self.ordered_input = kind == "sink"
        self.statistics = StepStatistics(name, self.threads)
        self.lock = threading.Lock()
        self.finished = 0
        self._created = 0

    def factory(self) -> Step:
        step = self._factory()
        with self.lock:
            step.thread_index = self._created  # which thread of the step (0, 1, ...)
            self._created += 1
        step.configure_threads(self.plan)
        return step

    def details(self) -> dict[str, typing.Any]:
        """Step.statistics_details() of all threads, numbers added up."""
        merged: dict[str, typing.Any] = {}
        for instance in list(self.instances):
            for key, value in instance.statistics_details().items():
                if isinstance(value, dict):
                    target = merged.setdefault(key, {})
                    for name, number in value.items():
                        target[name] = target.get(name, 0) + number
                else:
                    merged[key] = merged.get(key, 0) + value
        return merged


class Workflow:
    """A configured chain of steps.

        workflow = opennve.load_workflow("workflows.json", "insects")
        workflow.run()                 # until the source ends or Ctrl+C / workflow.stop()

        workflow.start(); ...; workflow.stop(); workflow.join()   # in the background

    In code: Workflow.from_steps("name", [step dicts or functions], cameras=..., frames=..., agents=...).
    """

    def __init__(self, config: typing.Union[WorkflowConfig, dict], **overrides):
        if isinstance(config, dict):
            config = WorkflowConfig.from_dict(config)
        if overrides:
            config = dataclasses.replace(config, **overrides)
        config.validate()
        self.config = config
        self.context = StepContext(
            workflow=config.name,
            base_dir=config.base_dir,
            cameras=config.cameras,
            frames=config.frames,
            agents=config.agents,
            debug=config.debug,
        )
        self._stages: list[_Stage] = []
        for index, step in enumerate(config.steps):
            self._stages.append(self._make_stage(index, step))
        self._check_structure()
        self._plan_threads()
        self._threads: list[threading.Thread] = []
        self._queues: list[queue.Queue] = []
        self._stop = threading.Event()
        self._errors: list[BaseException] = []
        self._started: typing.Optional[float] = None
        self._finished: typing.Optional[float] = None

    @classmethod
    def from_steps(cls, name: str, steps: list, **settings) -> "Workflow":
        """Workflow from a list of step configurations (dicts) or functions (callback steps)."""
        steps = [step if isinstance(step, dict) else {"type": "callback", "function": step} for step in steps]
        return cls(WorkflowConfig(name=name, steps=steps, **settings))

    def _make_stage(self, index: int, step: dict) -> _Stage:
        params = {key: value for key, value in step.items() if key != "type"}
        cls = get_step_class(step["type"])
        threads = params.get("threads", cls.DEFAULT_THREADS)
        if threads != "auto" and (isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0):
            raise ValueError(f"step {index} ({step['type']}): threads must be a positive integer or \"auto\"")
        if cls.MAX_THREADS is not None and threads == "auto":
            threads = cls.MAX_THREADS
        if cls.MAX_THREADS is not None and threads > cls.MAX_THREADS:
            raise ValueError(f"step {index} ({step['type']}): at most {cls.MAX_THREADS} thread(s)")
        try:
            first = cls(self.context, **params)  # validates the parameters now
        except (ValueError, TypeError) as error:
            raise ValueError(f"workflow {self.config.name}, step {index} ({step['type']}): {error}") from error
        instances = [first]

        def factory():
            if instances:
                return instances.pop()
            return cls(self.context, **params)

        name = params.get("name") or step["type"]
        stage = _Stage(factory, step, threads, cls.kind, name, first)
        stage.ordered_input = cls.kind == "sink" or bool(getattr(cls, "ORDERED", False))
        return stage

    def _check_structure(self):
        kinds = [stage.kind for stage in self._stages]
        if kinds[0] != "source":
            raise ValueError(f"workflow {self.config.name}: the first step must be a source (camera, raw)")
        if "source" in kinds[1:]:
            raise ValueError(f"workflow {self.config.name}: only the first step can be a source")
        names = [stage.name for stage in self._stages]
        for name in set(names):
            if names.count(name) > 1:
                # make statistics names unique
                count = 0
                for stage in self._stages:
                    if stage.name == name:
                        count += 1
                        stage.name = f"{name}#{count}"
                        stage.statistics.name = stage.name

    def add_step(self, step, position: typing.Optional[int] = None, **params) -> "Workflow":
        """Adds a step before the workflow starts: a step type (str) with params, a step dict,
        or a function (callback step: function(item) -> item or None)."""
        if self._threads:
            raise RuntimeError("steps cannot be added while the workflow runs")
        if callable(step) and not isinstance(step, type):
            step = {"type": "callback", "function": step, **params}
        elif isinstance(step, str):
            step = {"type": step, **params}
        self.config.steps.insert(len(self.config.steps) if position is None else position, step)
        index = self.config.steps.index(step)
        self._stages.insert(index, self._make_stage(index, step))
        self._check_structure()
        self._plan_threads()
        return self

    # threads

    @property
    def cpus(self) -> int:
        """CPU cores the automatic threads are sized for ("cpus" of the workflow, else
        opennve.cpu_count())."""
        cpus = self.config.cpus
        return cpu_count() if cpus is None or cpus == "auto" else cpus

    def _plan_threads(self):
        """Sizes the steps with "threads": "auto" to the CPU cores: light steps (frames, tiles)
        1-2 threads, the agent steps share all cores (several agent steps: in equal parts)."""
        cpus = self.cpus
        heavy = [stage for stage in self._stages if stage.prototype.HEAVY]
        budget = max(1, cpus // max(1, len(heavy)))
        for stage in self._stages:
            available = budget if stage.prototype.HEAVY else cpus
            if stage.requested == "auto":
                stage.threads = max(1, int(stage.prototype.auto_threads(available)))
                if stage.prototype.MAX_THREADS is not None:
                    stage.threads = min(stage.threads, stage.prototype.MAX_THREADS)
            stage.statistics.threads = stage.threads
            stage.plan = stage.prototype.thread_plan(stage.threads, available)

    @property
    def queue_size(self) -> int:
        """Items per queue between two steps ("queue_size", else at least 8 and twice the
        threads of the busiest step, so that all threads get work)."""
        if self.config.queue_size is not None:
            return self.config.queue_size
        return max(8, 2 * max(stage.threads for stage in self._stages))

    def thread_plan(self) -> dict[str, typing.Any]:
        """The threads of every step for this hardware:
        {"cpus": 8, "steps": [{"name": "agent", "threads": 8, "auto": True,
        "interpreter_threads": {"prototyp": 1}}, ...]}."""
        return {
            "cpus": self.cpus,
            "queue_size": self.queue_size,
            "steps": [
                {"name": stage.name, "threads": stage.threads, "auto": stage.requested == "auto", **stage.plan}
                for stage in self._stages
            ],
        }

    def thread_report(self) -> str:
        plan = self.thread_plan()
        source = "" if self.config.cpus not in (None, "auto") else " (detected)"
        lines = [f"threads for {plan['cpus']} CPU cores{source}:"]
        for step in plan["steps"]:
            extra = ""
            if step.get("interpreter_threads"):
                extra = ", interpreter threads " + ", ".join(f"{k} {v}" for k, v in step["interpreter_threads"].items())
            if step.get("device_threads"):
                for name, assignment in step["device_threads"].items():
                    counts = {device: assignment.count(device) for device in dict.fromkeys(assignment)}
                    extra += f", {name}: " + " + ".join(f"{device} x{count}" for device, count in counts.items())
            elif step.get("devices"):
                extra += ", device " + ", ".join(f"{k} {v}" for k, v in step["devices"].items())
            lines.append(f"  {step['name']:<16} x{step['threads']}{' (auto)' if step['auto'] else ''}{extra}")
        return "\n".join(lines)

    # running

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    @property
    def results(self) -> list[WorkItem]:
        """Items kept by "collect" steps."""
        items = []
        for stage in self._stages:
            for instance in stage.instances:
                if isinstance(instance, CollectStep):
                    items.extend(instance.items_collected)
        return items

    def start(self) -> "Workflow":
        """Starts all threads and returns immediately."""
        if self._threads:
            raise RuntimeError("the workflow was already started (create a new Workflow to run it again)")
        self._started = time.perf_counter()
        self._queues = [queue.Queue(maxsize=self.queue_size) for _ in self._stages[1:]]
        source = self._stages[0]
        self._threads.append(threading.Thread(target=self._run_source, args=(source,), name=f"{self.name}:{source.name}", daemon=True))
        for position, stage in enumerate(self._stages[1:], start=1):
            ordered = self.config.ordered and stage.ordered_input
            for thread_index in range(stage.threads):
                self._threads.append(
                    threading.Thread(
                        target=self._run_stage,
                        args=(position, stage, ordered),
                        name=f"{self.name}:{stage.name}:{thread_index}",
                        daemon=True,
                    )
                )
        for thread in self._threads:
            thread.start()
        if self.config.debug:
            self.context.log(f"[{self.name}] started: " + " -> ".join(f"{s.name}x{s.threads}" for s in self._stages))
            self.context.log(self.thread_report())
        return self

    def stop(self):
        """Stops the source; the items already read are still processed and written."""
        self._stop.set()

    def join(self, timeout: typing.Optional[float] = None) -> bool:
        """Waits for the workflow to finish. Raises the first error of a step. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self._threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                return False
        if self._finished is None:
            self._finished = time.perf_counter()
            if self.config.debug:
                self.context.log(self.statistics_report())
        if self._errors:
            raise self._errors[0]
        return True

    def run(self, duration_s: typing.Optional[float] = None) -> "Workflow":
        """Runs until the source ends, Ctrl+C, workflow.stop() or duration_s seconds."""
        self.start()
        started = time.monotonic()
        try:
            while self.running:
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    self.stop()
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop()
        self.join()
        return self

    # threads

    def _put(self, position: int, value):
        """Puts into the queue after stage position (blocking, but gives up if a step failed)."""
        target = self._queues[position]
        while True:
            try:
                target.put(value, timeout=0.1)
                return
            except queue.Full:
                if self._errors and not isinstance(value, _Stop):
                    return

    def _finish_stage(self, position: int, stage: _Stage):
        with stage.lock:
            stage.finished += 1
            last = stage.finished == stage.threads
        if last and position < len(self._stages) - 1:
            for _ in range(self._stages[position + 1].threads):
                self._put(position, _STOP)

    def _fail(self, error: BaseException, stage: _Stage):
        if self.config.debug:
            self.context.log(f"[{self.name}] step {stage.name} failed: {error!r}")
        self._errors.append(error)
        self._stop.set()

    def _run_source(self, stage: _Stage):
        step = None
        try:
            step = stage.factory()
            stage.instances.append(step)
            step.setup()
            iterator = step.items(self._stop)
            while True:
                start = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                elapsed = time.perf_counter() - start
                item.timings[stage.name] = elapsed
                stage.statistics.items += 1
                stage.statistics.seconds += elapsed
                if len(self._stages) > 1:
                    self._put(0, item)
                if self._stop.is_set() and self._errors:
                    break
        except BaseException as error:  # noqa: BLE001 - reported by join()
            self._fail(error, stage)
        finally:
            if step is not None:
                try:
                    step.close()
                except BaseException as error:  # noqa: BLE001
                    self._fail(error, stage)
            self._finish_stage(0, stage)

    def _run_stage(self, position: int, stage: _Stage, ordered: bool):
        source = self._queues[position - 1]
        step = None
        pending: dict[int, typing.Any] = {}
        next_index = 0
        try:
            step = stage.factory()
            with stage.lock:
                stage.instances.append(step)
            step.setup()
        except BaseException as error:  # noqa: BLE001
            self._fail(error, stage)

        def handle(value):
            if isinstance(value, _Skip):
                if position < len(self._stages) - 1:
                    self._put(position, value)
                return
            if self._errors:  # drain after a failure
                return
            start = time.perf_counter()
            result = step.process(value)
            elapsed = time.perf_counter() - start
            with stage.lock:
                stage.statistics.items += 1
                stage.statistics.seconds += elapsed
                if result is None:
                    stage.statistics.dropped += 1
            if result is not None:
                result.timings[stage.name] = result.timings.get(stage.name, 0.0) + elapsed
            if position < len(self._stages) - 1:
                self._put(position, result if result is not None else _Skip(value.index))

        while True:
            value = source.get()
            if isinstance(value, _Stop):
                break
            try:
                if ordered:
                    pending[value.index] = value
                    while next_index in pending:
                        handle(pending.pop(next_index))
                        next_index += 1
                else:
                    handle(value)
            except BaseException as error:  # noqa: BLE001
                self._fail(error, stage)
        try:
            for index in sorted(pending):  # items after a gap (only if a source skipped indices)
                handle(pending[index])
        except BaseException as error:  # noqa: BLE001
            self._fail(error, stage)
        if step is not None:
            try:
                step.close()
            except BaseException as error:  # noqa: BLE001
                self._fail(error, stage)
        self._finish_stage(position, stage)

    # statistics

    def statistics(self) -> dict[str, typing.Any]:
        """Items, dropped items and time per step; total run time."""
        end = self._finished or time.perf_counter()
        return {
            "workflow": self.name,
            "cpus": self.cpus,
            "seconds": None if self._started is None else end - self._started,
            "steps": [
                {
                    "name": stage.name,
                    "threads": stage.threads,
                    "items": stage.statistics.items,
                    "dropped": stage.statistics.dropped,
                    "seconds": stage.statistics.seconds,
                    "mean_ms": stage.statistics.mean_ms,
                    **stage.details(),
                }
                for stage in self._stages
            ],
        }

    def statistics_report(self) -> str:
        stats = self.statistics()
        seconds = stats["seconds"] or 0.0
        lines = [f"[{self.name}] {seconds:.1f} s"]
        for step in stats["steps"]:
            dropped = f", {step['dropped']} dropped" if step["dropped"] else ""
            devices = ""
            if step.get("inputs_per_device") and len(step["inputs_per_device"]) > 1:
                devices = " (" + ", ".join(f"{name} {count} inputs" for name, count in step["inputs_per_device"].items()) + ")"
            lines.append(
                f"  {step['name']:<16} x{step['threads']}  {step['items']} items{dropped}, "
                f"{step['mean_ms']:.1f} ms per item{devices}"
            )
        return "\n".join(lines)

    def describe(self) -> str:
        """The steps as text."""
        lines = [f"workflow {self.name}" + (" (default)" if self.config.default else "")]
        for stage in self._stages:
            params = {k: v for k, v in stage.config.items() if k not in COMMON_PARAMETERS and not callable(v)}
            threads = f" x{stage.threads}" if stage.threads > 1 else ""
            lines.append(f"  {stage.config['type']}{threads} {json.dumps(params) if params else ''}".rstrip())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Workflow(name={self.name!r}, steps={[stage.name for stage in self._stages]})"


def save_workflow_configs(path: PathLike, configs: typing.Union[WorkflowConfig, typing.Sequence[WorkflowConfig]]):
    if isinstance(configs, WorkflowConfig):
        configs = [configs]
    data = {"workflows": [config.to_dict() for config in configs]}
    pathlib.Path(path).write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
