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
"""

from __future__ import annotations

import datetime
import glob
import json
import math
import os
import pathlib
import threading
import time
import typing

import numpy as np

from ..frames import FrameConfig, as_frame_config
from ..window import EventWindow
from .item import WorkItem

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
        from ..config import CameraConfig, load_configs

        if self._cameras is None:
            raise ValueError("this step needs camera configurations: set \"cameras\" in the workflow")
        if isinstance(self._cameras, (str, os.PathLike)):
            self._cameras = load_configs(self.path(self._cameras))
        return [c if isinstance(c, CameraConfig) else CameraConfig.from_dict(c) for c in self._cameras]

    def frame_configs(self) -> dict[str, FrameConfig]:
        from ..frames import as_frame_configs, load_frame_configs

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
        from ..agents import AgentConfig, agent_configs_from_data, load_agent_configs

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
    """

    TYPE = "step"
    kind = "process"
    PARAMETERS: dict[str, typing.Any] = {}
    MAX_THREADS: typing.Optional[int] = None

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


# ------------------------------------------------------------------ sources


class _Source(Step):
    kind = "source"
    MAX_THREADS = 1

    def _limits(self):
        return self.params.get("max_blocks"), self.params.get("duration_s")


@register_step("camera")
class CameraSource(_Source):
    """Live cameras: one item per block of block_ms (default: the camera "window" duration).

    cameras: camera names of the camera configuration (null: all cameras of the file).
    max_blocks / duration_s: stop after this many blocks / seconds (null: until stopped).
    install_driver: install missing USB drivers first (Windows, SE1-S4).
    """

    PARAMETERS = {"cameras": None, "block_ms": None, "max_blocks": None, "duration_s": None, "install_driver": False}

    def validate(self):
        _positive_number(self, "block_ms")
        _positive_number(self, "max_blocks", integer=True)
        _positive_number(self, "duration_s")
        cameras = self.params["cameras"]
        if isinstance(cameras, str):
            self.params["cameras"] = [cameras]
        elif cameras is not None and not (isinstance(cameras, list) and all(isinstance(c, str) for c in cameras)):
            raise ValueError("step camera: cameras must be a camera name, a list of names or null")

    def items(self, stop):
        from ..camera import open_cameras

        configs = self.context.camera_configs()
        names = self.params["cameras"]
        if names is not None:
            by_name = {config.name: config for config in configs}
            missing = [name for name in names if name not in by_name]
            if missing:
                raise ValueError(f"step camera: unknown camera(s) {', '.join(missing)} (available: {', '.join(by_name)})")
            configs = [by_name[name] for name in names]
        max_blocks, duration_s = self._limits()
        started = time.monotonic()
        index = 0
        cameras = open_cameras(configs, install_driver=bool(self.params["install_driver"]))
        try:
            while not stop.is_set():
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    return
                for camera in cameras.values():
                    for window in camera.read_windows(self.params["block_ms"]):
                        if stop.is_set():
                            return
                        yield WorkItem(index=index, camera=window.camera, block=window)
                        index += 1
                        if max_blocks is not None and index >= max_blocks:
                            return
        finally:
            for camera in cameras.values():
                camera.close()


@register_step("raw")
class RawSource(_Source):
    """RAW recordings: one item per block of block_ms.

    path: file, glob pattern ("recordings/*.raw") or list (relative to the workflow file).
    camera: camera name in the camera configuration whose filter, output format and window are
        used (null: defaults; the camera name is then the file name).
    speed: 1.0 = real time, null = as fast as possible.
    chunk_kb: bytes read at once (null: 1024 KB, or 16 KB with a speed, for smooth playback).
    """

    PARAMETERS = {"path": None, "camera": None, "block_ms": None, "speed": None, "max_blocks": None, "chunk_kb": None}

    def validate(self):
        if not self.params["path"]:
            raise ValueError("step raw: path is required")
        _positive_number(self, "block_ms")
        _positive_number(self, "speed")
        _positive_number(self, "max_blocks", integer=True)
        _positive_number(self, "chunk_kb", integer=True)

    def paths(self) -> list[pathlib.Path]:
        values = self.params["path"]
        values = [values] if isinstance(values, (str, os.PathLike)) else values
        paths: list[pathlib.Path] = []
        for value in values:
            pattern = str(self.context.path(value))
            matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
            paths.extend(pathlib.Path(match) for match in matches)
        if not paths:
            raise ValueError(f"step raw: no file matches {values}")
        return paths

    def items(self, stop):
        from ..raw import open_raw

        config = None
        if self.params["camera"] is not None:
            configs = {c.name: c for c in self.context.camera_configs()}
            if self.params["camera"] not in configs:
                raise ValueError(f"step raw: unknown camera {self.params['camera']!r}")
            config = configs[self.params["camera"]]
        block_ms = self.params["block_ms"]
        index = 0
        for path in self.paths():
            chunk_kb = self.params["chunk_kb"] or (16 if self.params["speed"] else 1024)
            with open_raw(path, config, speed=self.params["speed"], chunk_bytes=chunk_kb * 1024) as raw:
                if block_ms is None and raw.window_config is None:
                    block_ms = 40
                while not stop.is_set():
                    windows = raw.read_windows(block_ms)
                    if windows is None:
                        break
                    for window in windows:
                        if stop.is_set():
                            return
                        yield WorkItem(index=index, camera=window.camera, block=window)
                        index += 1
                        if self.params["max_blocks"] is not None and index >= self.params["max_blocks"]:
                            return
            if stop.is_set():
                return


# ------------------------------------------------------------------ processing steps


def sub_window(block: EventWindow, start_t: int, end_t: int) -> EventWindow:
    """The events of block between start_t (included) and end_t (excluded) as a new window."""
    t = block.polarity_events["t"]
    first, last = np.searchsorted(t, [start_t, end_t], side="left")
    return EventWindow(
        camera=block.camera,
        start_t=int(start_t),
        end_t=int(end_t),
        width=block.width,
        height=block.height,
        polarity_events=block.polarity_events[first:last],
        output_config=block.output_config,
    )


def window_bounds(start_t: int, length_us: float, count: int, overlap_us: float) -> list[tuple[int, int]]:
    """count windows of equal size covering length_us, neighbours overlapping by overlap_us."""
    size = (length_us + (count - 1) * overlap_us) / count
    step = size - overlap_us
    if step <= 0:
        raise ValueError("the overlap is too large for the number of windows")
    bounds = []
    for index in range(count):
        start = round(start_t + index * step)
        bounds.append((start, round(start + size)))
    return bounds


@register_step("windows")
class WindowsStep(Step):
    """Splits each block into time windows.

    count / overlap_ms: count windows of equal size that cover the block and overlap by
        overlap_ms (for instance a 1000 ms block, count 3, overlap 200 ms: 0-467, 267-733, 533-1000 ms).
    duration_ms / step_ms: alternatively windows of duration_ms every step_ms (sliding windows).
    """

    PARAMETERS = {"count": None, "overlap_ms": 0, "duration_ms": None, "step_ms": None}

    def validate(self):
        _positive_number(self, "count", integer=True)
        _positive_number(self, "duration_ms")
        _positive_number(self, "step_ms")
        overlap = self.params["overlap_ms"]
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or overlap < 0:
            raise ValueError("step windows: overlap_ms must be a number >= 0")
        if (self.params["count"] is None) == (self.params["duration_ms"] is None):
            raise ValueError("step windows: set either count (with overlap_ms) or duration_ms (with step_ms)")

    def process(self, item):
        block = item.block
        length = block.end_t - block.start_t
        if self.params["count"] is not None:
            bounds = window_bounds(block.start_t, length, self.params["count"], self.params["overlap_ms"] * 1000)
        else:
            duration = self.params["duration_ms"] * 1000
            step = (self.params["step_ms"] or self.params["duration_ms"]) * 1000
            bounds = []
            start = block.start_t
            while start < block.end_t:
                bounds.append((round(start), round(min(start + duration, block.end_t))))
                if start + duration >= block.end_t:
                    break
                start += step
        item.windows = [sub_window(block, start, end) for start, end in bounds]
        item.frames = []
        item.results = []
        return item


@register_step("frames")
class FramesStep(Step):
    """Builds frames from each window with one or more frame configurations.

    frame: name of a frame configuration (from "frames"), an inline configuration (object), or a
        list of these. Without tiles in the configuration, use the step "tiles" afterwards.
    min_events_per_ms: minimum events per tile relative to the window length (overrides the
        "min_events" of the configuration), for instance 0.15 = 15 events per 100 ms.
    """

    PARAMETERS = {"frame": None, "min_events_per_ms": None}

    def validate(self):
        frames = self.params["frame"]
        if frames is None:
            raise ValueError("step frames: frame is required (name or configuration)")
        self.configs = [self.context.frame_config(value) for value in (frames if isinstance(frames, list) else [frames])]
        _positive_number(self, "min_events_per_ms")

    def process(self, item):
        frames = []
        per_ms = self.params["min_events_per_ms"]
        for window in item.windows:
            built = {}
            for config in self.configs:
                if per_ms is not None and config.tiles is not None:
                    tiles = {"size": config.tiles.size, "min_events": max(1, math.ceil(per_ms * window.duration_ms))}
                    config = config.copy(tiles=tiles)
                built[config.name] = window.frame(config)
            frames.append(built)
        item.frames = frames
        return item


@register_step("tiles")
class TilesStep(Step):
    """Splits the frames (built without tiles, possibly processed) into tiles.

    frame: name of the frame configuration whose frames are split (null: all).
    size: tile size in pixels. min_events: minimum events per tile (0 = all tiles).
    min_events_per_ms: minimum events relative to the window length (overrides min_events).
    """

    PARAMETERS = {"frame": None, "size": 80, "min_events": 1, "min_events_per_ms": None}

    def validate(self):
        _positive_number(self, "size", allow_none=False, integer=True)
        _positive_number(self, "min_events_per_ms")
        value = self.params["min_events"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("step tiles: min_events must be an integer >= 0")

    def process(self, item):
        if not item.frames:
            raise ValueError("step tiles needs frames: add the step \"frames\" before it")
        per_ms = self.params["min_events_per_ms"]
        for window, frames in zip(item.windows, item.frames):
            minimum = self.params["min_events"] if per_ms is None else max(1, math.ceil(per_ms * window.duration_ms))
            for name, frame in frames.items():
                if self.params["frame"] is None or name == self.params["frame"]:
                    frame.split(self.params["size"], minimum)
        return item


@register_step("agent")
class AgentStep(Step):
    """Runs agents (models) on the frames / tiles of each window.

    agent: agent name (from "agents"), inline agent configuration, or a list of these.
    frame: frame configuration name the agent uses (default: the agent's "frame" setting, else
        the only frame of the window).
    Each thread loads its own copy of the model.
    """

    PARAMETERS = {"agent": None, "frame": None}

    def validate(self):
        agents = self.params["agent"]
        if agents is None:
            raise ValueError("step agent: agent is required")
        self.configs = [self.context.agent_config(value) for value in (agents if isinstance(agents, list) else [agents])]

    def setup(self):
        from ..agents import Agent

        self.agents = [Agent(config).load() for config in self.configs]

    def _frame(self, agent, frames: dict):
        name = self.params["frame"] or agent.config.frame
        if name is None:
            if len(frames) != 1:
                raise ValueError(f"step agent: set \"frame\" (the windows have frames {', '.join(frames)})")
            return next(iter(frames.values()))
        if name not in frames:
            raise ValueError(f"step agent: agent {agent.name} needs frame {name!r} (built: {', '.join(frames) or 'none'})")
        return frames[name]

    def process(self, item):
        if not item.frames:
            raise ValueError("step agent needs frames: add the step \"frames\" before it")
        if len(item.results) != len(item.windows):
            item.results = [{} for _ in item.windows]
        for frames, results in zip(item.frames, item.results):
            for agent in self.agents:
                results[agent.name] = agent.predict(self._frame(agent, frames))
        return item

    def close(self):
        for agent in getattr(self, "agents", []):
            agent.close()


@register_step("require")
class RequireStep(Step):
    """Keeps only items in which an agent found something.

    agent: agent name. min_windows: minimum number of windows with at least one selected
    prediction (tile / pixel). min_pixels: minimum target pixels in such a window (segmentation).
    """

    PARAMETERS = {"agent": None, "min_windows": 1, "min_pixels": None}

    def validate(self):
        if not isinstance(self.params["agent"], str):
            raise ValueError("step require: agent (name) is required")
        _positive_number(self, "min_windows", allow_none=False, integer=True)
        _positive_number(self, "min_pixels", integer=True)

    def process(self, item):
        agent = self.params["agent"]
        positive = 0
        for results in item.results:
            result = results.get(agent)
            if result is None:
                continue
            selected = result.selected
            if self.params["min_pixels"] is not None:
                selected = [p for p in selected if p.pixel_count >= self.params["min_pixels"]]
            positive += bool(selected)
        return item if positive >= self.params["min_windows"] else None


@register_step("label_events")
class LabelEventsStep(Step):
    """Labels the events of the block with the predictions of an agent.

    An event is labeled if, in a window that contains its time stamp, the agent predicted a
    target pixel at its position (segmentation) or selected the tile it lies in (classification).
    agent: agent name. label: value for labeled events (null: the predicted class id).
    field: name of the label field. only_labeled: keep only labeled events (else all, with 0).
    Writes item.events (output format of the camera plus the label field).
    """

    PARAMETERS = {"agent": None, "label": None, "field": "l", "only_labeled": True}

    def validate(self):
        if not isinstance(self.params["agent"], str):
            raise ValueError("step label_events: agent (name) is required")
        if not isinstance(self.params["field"], str) or not self.params["field"]:
            raise ValueError("step label_events: field must be a name")

    def process(self, item):
        block = item.block
        events = block.polarity_events
        width = block.width
        labels = np.zeros(len(events), np.int32)
        matched = np.zeros(len(events), bool)
        position = events["y"].astype(np.int64) * width + events["x"].astype(np.int64)
        t = events["t"]
        for window, results in zip(item.windows, item.results):
            result = results.get(self.params["agent"])
            if result is None or not result.selected:
                continue
            first, last = np.searchsorted(t, [window.start_t, window.end_t], side="left")
            if first == last:
                continue
            lookup = np.zeros(block.width * block.height, np.int32)  # class id + 1 per pixel, 0 = none
            for prediction in result.selected:
                if prediction.pixels is not None:
                    if len(prediction.pixels):
                        index = prediction.pixels["y"].astype(np.int64) * width + prediction.pixels["x"].astype(np.int64)
                        lookup[index] = prediction.pixels["class_id"].astype(np.int32) + 1
                elif prediction.tile is not None:
                    tile = prediction.tile
                    rows = np.arange(tile.y, tile.y + tile.height)[:, None] * width
                    lookup[(rows + np.arange(tile.x, tile.x + tile.width)[None, :]).ravel()] = (prediction.class_id or 0) + 1
            values = lookup[position[first:last]]
            new = (values > 0) & ~matched[first:last]  # with overlapping windows the first match counts
            window_labels = labels[first:last]
            window_labels[new] = values[new] - 1 if self.params["label"] is None else self.params["label"]
            matched[first:last] |= new
        output = block.output_config.format(events)
        dtype = np.dtype([(name, output.dtype.fields[name][0]) for name in output.dtype.names] + [(self.params["field"], "<i4")])
        labeled = np.zeros(len(output), dtype)
        for name in output.dtype.names:
            labeled[name] = output[name]
        labeled[self.params["field"]] = labels
        item.events = labeled[matched] if self.params["only_labeled"] else labeled
        item.data["labeled_events"] = int(matched.sum())
        return item


@register_step("log")
class LogStep(Step):
    """Prints a line per item: block, events, windows, tiles, selected predictions, timings."""

    PARAMETERS = {"agent": None, "timings": True}

    def process(self, item):
        tiles = sum(len(frame.tiles) for frames in item.frames for frame in frames.values())
        parts = [f"[{self.context.workflow}] block {item.index} {item.camera}", f"events {item.event_count:,}", f"windows {len(item.windows)}"]
        if item.frames:
            parts.append(f"tiles {tiles}")
        agents = [self.params["agent"]] if self.params["agent"] else sorted({n for r in item.results for n in r})
        for name in agents:
            positive = item.positive_windows(name)
            pixels = sum(
                p.pixel_count for results in item.results if name in results for p in results[name].selected
            )
            parts.append(f"{name}: {len(positive)}/{len(item.results)} windows, {pixels} pixels")
        if item.events is not None:
            parts.append(f"output events {len(item.events):,}")
        if self.params["timings"] and item.timings:
            parts.append(" ".join(f"{name}={seconds * 1000:.0f}ms" for name, seconds in item.timings.items()))
        self.context.log(" | ".join(parts))
        return item


@register_step("callback")
class CallbackStep(Step):
    """Calls a Python function with each item (from code: workflow.add_step(function)).
    The function returns the item (or a changed one), or None to drop it."""

    PARAMETERS = {"function": None}

    def validate(self):
        if not callable(self.params["function"]):
            raise ValueError("step callback: function must be callable")

    def process(self, item):
        result = self.params["function"](item)
        return item if result is True else (None if result is False else result)


# ------------------------------------------------------------------ sinks


def _format_path(context: StepContext, pattern: str, item: WorkItem) -> pathlib.Path:
    started = context.started
    values = {
        "workflow": context.workflow,
        "camera": item.camera,
        "date": started.strftime("%Y%m%d"),
        "time": started.strftime("%H%M"),
        "datetime": started.strftime("%Y%m%d_%H%M%S"),
        "index": item.index,
    }
    try:
        return context.path(pattern.format(**values))
    except KeyError as error:
        raise ValueError(f"unknown placeholder {error} in path {pattern!r} (supported: {', '.join(values)})") from None


@register_step("write")
class WriteStep(Step):
    """Writes the results to files.

    format: "csv" (item.events, or the block events if no step set events; one row per event)
        or "jsonl" (one JSON line per item: block, windows and selected predictions with pixels).
    path: file name pattern with {workflow}, {camera}, {date}, {time}, {datetime}, {index},
        for instance "output/{workflow}_{camera}_{date}_{time}.csv". Files are appended.
    camera_column: add the camera name as first CSV column.
    """

    kind = "sink"
    MAX_THREADS = 1
    PARAMETERS = {"format": "csv", "path": "output/{workflow}_{date}_{time}.csv", "camera_column": False}

    def validate(self):
        if self.params["format"] not in ("csv", "jsonl"):
            raise ValueError("step write: format must be csv or jsonl")
        if not isinstance(self.params["path"], str):
            raise ValueError("step write: path must be a file name pattern")
        self._files: dict[pathlib.Path, typing.TextIO] = {}
        self._lock = threading.Lock()
        self.written: dict[str, int] = {}

    def _file(self, path: pathlib.Path):
        if path not in self._files:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists() or path.stat().st_size == 0
            self._files[path] = (open(path, "a", encoding="utf-8", newline=""), new)
        return self._files[path]

    def process(self, item):
        path = _format_path(self.context, self.params["path"], item)
        with self._lock:
            handle, new = self._file(path)
            if self.params["format"] == "jsonl":
                handle.write(json.dumps(item.summary()) + "\n")
            else:
                events = item.events if item.events is not None else item.block.events
                names = list(events.dtype.names)
                if new:
                    header = (["camera"] if self.params["camera_column"] else []) + names
                    handle.write(",".join(header) + "\n")
                    self._files[path] = (handle, False)
                if len(events):
                    columns = [events[name].astype(np.int64) if events[name].dtype != np.bool_ else events[name].astype(np.int8) for name in names]
                    rows = np.column_stack(columns)
                    prefix = f"{item.camera}," if self.params["camera_column"] else ""
                    handle.write("".join(prefix + ",".join(map(str, row)) + "\n" for row in rows.tolist()))
            handle.flush()
            self.written[str(path)] = self.written.get(str(path), 0) + (1 if self.params["format"] == "jsonl" else len(item.events if item.events is not None else item.block))
        return item

    def close(self):
        with self._lock:
            for handle, _ in self._files.values():
                handle.close()
            self._files.clear()


@register_step("collect")
class CollectStep(Step):
    """Keeps the items in memory: workflow.results (for use from code and tests)."""

    kind = "sink"
    MAX_THREADS = 1
    PARAMETERS = {"max_items": None}

    def validate(self):
        _positive_number(self, "max_items", integer=True)
        self.items_collected: list[WorkItem] = []
        self._lock = threading.Lock()

    def process(self, item):
        with self._lock:
            self.items_collected.append(item)
            if self.params["max_items"] is not None and len(self.items_collected) > self.params["max_items"]:
                self.items_collected.pop(0)
        return item
