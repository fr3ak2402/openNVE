"""Processing steps: windows, frames, tiles, agent, require, label_events, log, callback."""

from __future__ import annotations

import math
import threading

import numpy as np

from ...events.window import EventWindow
from ...system import MAX_AUTO_AGENT_THREADS, is_auto
from .base import Step, _positive_number, register_step


def _light_threads(cpus: int) -> int:
    """Threads of light steps (frames, tiles): ~0.1 ms per tile against ~4 ms in the agent, and
    mostly Python (GIL), so one thread is enough up to 8 cores, then two."""
    return 1 if cpus < 8 else 2



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
    align: where the windows start: "block" (start of the block) or "first_event" (time stamp of
        the first event of the block, as in the previous workflow engine).
    """

    PARAMETERS = {"count": None, "overlap_ms": 0, "duration_ms": None, "step_ms": None, "align": "block"}

    def validate(self):
        _positive_number(self, "count", integer=True)
        _positive_number(self, "duration_ms")
        _positive_number(self, "step_ms")
        overlap = self.params["overlap_ms"]
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or overlap < 0:
            raise ValueError("step windows: overlap_ms must be a number >= 0")
        if (self.params["count"] is None) == (self.params["duration_ms"] is None):
            raise ValueError("step windows: set either count (with overlap_ms) or duration_ms (with step_ms)")
        if self.params["align"] not in ("block", "first_event"):
            raise ValueError("step windows: align must be block or first_event")

    def process(self, item):
        block = item.block
        origin = block.start_t
        if self.params["align"] == "first_event" and len(block):
            origin = int(block.polarity_events["t"][0])
        length = block.end_t - block.start_t
        if self.params["count"] is not None:
            bounds = window_bounds(origin, length, self.params["count"], self.params["overlap_ms"] * 1000)
        else:
            duration = self.params["duration_ms"] * 1000
            step = (self.params["step_ms"] or self.params["duration_ms"]) * 1000
            bounds = []
            start = origin
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
    DEFAULT_THREADS = "auto"

    def auto_threads(self, cpus):
        return _light_threads(cpus)

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
    DEFAULT_THREADS = "auto"

    def auto_threads(self, cpus):
        return _light_threads(cpus)

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

    Threads ("auto" by default): TensorFlow Lite agents get one thread per core with one
    interpreter thread each (several interpreters in parallel are faster than one with several
    threads). With a fixed number of step threads, agents with "threads": "auto" share the cores:
    interpreter threads = cores / step threads. TensorFlow models run in one thread (TensorFlow
    uses all cores itself). GPU / NPU agents get 2 threads.

    Several devices ("device": ["gpu", "cpu"]): 2 threads per accelerator, the other cores run
    the model on the CPU (one thread each); every thread takes the next window when it is free,
    so fast devices get more work. The statistics show how many inputs each device ran.
    """

    PARAMETERS = {"agent": None, "frame": None}
    DEFAULT_THREADS = "auto"
    HEAVY = True

    def _tflite(self):
        """Agents whose threads the step plans: TensorFlow Lite and PyTorch (one model per thread)."""
        return [config for config in self.configs if config.resolved_backend in ("tflite", "pytorch")]

    @staticmethod
    def _devices(config) -> list:
        return list(config.device) if isinstance(config.device, list) else []

    def _mixed(self):
        return [config for config in self._tflite() if len(self._devices(config)) > 1 and config.delegate is None]

    @staticmethod
    def _mixed_threads(devices: list, cpus: int) -> tuple[int, int]:
        """(accelerator threads, CPU threads) for a device list: 2 per accelerator, one core per
        accelerator to feed it, the other cores for CPU threads."""
        accelerators = [device for device in devices if device != "cpu"]
        accelerator_threads = 2 * len(accelerators)
        cpu_threads = 0
        if "cpu" in devices:
            cpu_threads = max(1, min(cpus - len(accelerators), MAX_AUTO_AGENT_THREADS))
        return accelerator_threads, cpu_threads

    @staticmethod
    def _accelerated(config) -> bool:
        """The agent runs on a GPU / NPU (device "auto": if the machine has one)."""
        if isinstance(config.device, list):
            return "cpu" not in config.device or len(config.device) == 1 and config.device[0] != "cpu"
        if config.device in ("gpu", "npu") or config.delegate is not None:
            return True
        if config.device == "auto":
            from ...system import accelerators

            return bool(accelerators())
        return False

    def auto_threads(self, cpus):
        tflite = self._tflite()
        if not tflite:  # TensorFlow (or own backends): the backend parallelizes itself
            return 1
        mixed = self._mixed()
        if mixed:
            return max(sum(self._mixed_threads(self._devices(config), cpus)) for config in mixed)
        if all(self._accelerated(config) for config in tflite):
            # GPU / NPU: one model per thread on the same accelerator; two threads overlap the
            # preparation and decoding (CPU) with the inference
            return max(1, min(cpus, 2))
        fixed = [config.threads for config in tflite if not is_auto(config.threads)]
        if fixed:
            return max(1, cpus // max(fixed))
        return max(1, min(cpus, MAX_AUTO_AGENT_THREADS))

    def thread_plan(self, threads, cpus):
        plan = {}
        devices = {config.name: config.device for config in self.configs if config.device is not None}
        if devices:
            plan["devices"] = devices
        if self._tflite():
            auto = max(1, cpus // threads)
            plan["interpreter_threads"] = {
                config.name: auto if is_auto(config.threads) else config.threads for config in self._tflite()
            }
        mixed = self._mixed()
        if mixed:
            plan["device_threads"] = {}
            for config in mixed:
                devices = self._devices(config)
                accelerators = [device for device in devices if device != "cpu"]
                accelerator_threads, _ = self._mixed_threads(devices, cpus)
                if "cpu" in devices:  # at least one CPU thread
                    accelerator_threads = min(accelerator_threads, max(0, threads - 1))
                else:
                    accelerator_threads = threads
                assignment = [accelerators[index % len(accelerators)] for index in range(accelerator_threads)]
                assignment += ["cpu"] * (threads - accelerator_threads)
                plan["device_threads"][config.name] = assignment
                cpu_threads = assignment.count("cpu")
                if is_auto(config.threads) and cpu_threads:
                    plan["interpreter_threads"][config.name] = max(1, (cpus - len(accelerators)) // cpu_threads)
        return plan

    def configure_threads(self, plan):
        threads = plan.get("interpreter_threads") or {}
        devices = plan.get("device_threads") or {}
        configs = []
        for config in self.configs:
            changes = {}
            if config.name in devices:
                device = devices[config.name][self.thread_index % len(devices[config.name])]
                changes["device"] = device
                if is_auto(config.threads):
                    changes["threads"] = threads.get(config.name, 1) if device == "cpu" else 1
            elif is_auto(config.threads) and config.name in threads:
                changes["threads"] = threads[config.name]
            configs.append(config.copy(**changes) if changes else config)
        self.configs = configs

    def validate(self):
        agents = self.params["agent"]
        if agents is None:
            raise ValueError("step agent: agent is required")
        self.configs = [self.context.agent_config(value) for value in (agents if isinstance(agents, list) else [agents])]

    _reported: set = set()
    _report_lock = threading.Lock()

    def setup(self):
        from ...agents import Agent

        self.agents = [Agent(config).load() for config in self.configs]
        self.inputs_per_device: dict[str, int] = {}
        for agent in self.agents:  # where the agent runs, once per workflow, agent and device
            backend = agent.backend
            if agent.config.device is None and backend.device == "cpu" and not backend.device_note:
                continue
            key = (id(self.context), agent.name, agent.config.device if isinstance(agent.config.device, str) else None)
            with AgentStep._report_lock:
                if key in AgentStep._reported:
                    continue
                AgentStep._reported.add(key)
            message = f"[{self.name}] agent {agent.name} runs on {agent.device_description()}"
            self.context.log(message)

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
                result = agent.predict(self._frame(agent, frames))
                results[agent.name] = result
                device = agent.device
                self.inputs_per_device[device] = self.inputs_per_device.get(device, 0) + max(1, len(result.predictions))
        return item

    def statistics_details(self):
        return {"inputs_per_device": dict(getattr(self, "inputs_per_device", {}))}

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
        field = self.params["field"]
        names = [name for name in output.dtype.names if name != field]  # an "l" output field is replaced
        dtype = np.dtype([(name, output.dtype.fields[name][0]) for name in names] + [(field, "<i4")])
        labeled = np.zeros(len(output), dtype)
        for name in names:
            labeled[name] = output[name]
        labeled[field] = labels
        item.events = labeled[matched] if self.params["only_labeled"] else labeled
        item.data["labeled_events"] = int(matched.sum())
        return item


@register_step("log")
class LogStep(Step):
    """Prints a line per item: block, events, windows, tiles, selected predictions, timings.
    Runs in one thread and prints the items in their order (if the workflow is "ordered")."""

    MAX_THREADS = 1
    ORDERED = True
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
