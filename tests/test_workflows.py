"""Tests for workflows (steps, threads, order, outputs)."""

import csv
import json
import threading
import time

import numpy as np
import pytest

import opennve
from opennve.workflows.steps import window_bounds
from test_agents import DATA, _has_tflite
from test_raw import _header, _random_events, encode_evt3

needs_tflite = pytest.mark.skipif(not _has_tflite(), reason="LiteRT (ai-edge-litert) is not installed")


@pytest.fixture
def raw_file(tmp_path):
    events = _random_events(11, count=20_000, span=1_000_000, width=64, height=32)
    path = tmp_path / "rec.raw"
    path.write_bytes(_header("EVT3;height=32;width=64") + encode_evt3(events))
    return path


def test_window_bounds():
    # 1000 ms block, 3 windows overlapping by 200 ms (the previous build_bounds)
    assert window_bounds(0, 1_000_000, 3, 200_000) == [(0, 466667), (266667, 733334), (533333, 1000000)]  # as before
    assert window_bounds(10, 300, 1, 0) == [(10, 310)]
    with pytest.raises(ValueError, match="overlap"):
        window_bounds(0, 100, 2, 200)


def test_workflow_config_validation(tmp_path):
    path = tmp_path / "workflows.json"
    path.write_text(json.dumps({"workflows": [
        {"name": "a", "default": True, "steps": [{"type": "raw", "path": "x.raw"}, {"type": "collect"}]},
        {"name": "b", "steps": [{"type": "raw", "path": "x.raw"}, {"type": "log"}]},
    ]}))
    assert opennve.load_workflow_config(path).name == "a"  # default
    assert opennve.load_workflow(path, "b").name == "b"
    with pytest.raises(ValueError, match="no workflow named 'c'"):
        opennve.load_workflow_config(path, "c")
    for steps, message in (
        ([{"type": "collect"}], "first step must be a source"),
        ([{"type": "raw", "path": "x"}, {"type": "raw", "path": "y"}], "only the first step"),
        ([{"type": "raw", "path": "x"}, {"type": "teleport"}], "unknown step type 'teleport'"),
        ([{"type": "raw", "path": "x"}, {"type": "windows", "count": 3, "duration_ms": 10}], "either count"),
        ([{"type": "raw", "path": "x"}, {"type": "tiles", "size": 0}], "size must be a positive integer"),
        ([{"type": "raw", "path": "x"}, {"type": "frames", "frame": "missing"}], "no frame configuration named"),
        ([{"type": "raw", "path": "x"}, {"type": "write", "threads": 2}], "at most 1 thread"),
        ([{"type": "raw", "path": "x"}, {"type": "log", "color": "red"}], "unknown parameter"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.Workflow({"name": "w", "steps": steps})
    with pytest.raises(ValueError, match="unknown workflow setting"):
        opennve.WorkflowConfig.from_dict({"name": "w", "steps": [], "threads": 2})


def test_raw_windows_frames_tiles_callback(raw_file):
    seen = []
    lock = threading.Lock()

    def check(item):
        with lock:
            seen.append(threading.current_thread().name)
        return item

    workflow = opennve.Workflow.from_steps(
        "basic",
        [
            {"type": "raw", "path": str(raw_file), "block_ms": 250},
            {"type": "windows", "count": 3, "overlap_ms": 50},
            {"type": "frames", "frame": {"name": "full", "mode": "binary"}, "threads": 3},
            {"type": "tiles", "size": 16, "min_events_per_ms": 0.1, "threads": 2},
            check,
            {"type": "collect"},
        ],
    )
    workflow.run()
    items = workflow.results
    assert [item.index for item in items] == [0, 1, 2, 3]  # ordered
    item = items[1]
    assert item.block.start_t == item.windows[0].start_t and item.windows[-1].end_t == item.block.end_t
    assert len(item.windows) == 3 and sum(len(w) for w in item.windows) > len(item.block)  # overlap
    frame = item.frames[0]["full"]
    minimum = int(np.ceil(0.1 * item.windows[0].duration_ms))
    assert frame.tiles and all(tile.event_count >= minimum for tile in frame.tiles)
    stats = workflow.statistics()
    assert [s["items"] for s in stats["steps"]] == [4, 4, 4, 4, 4, 4]
    assert "frames" in item.timings and "tiles" in item.timings
    assert "callback" in workflow.statistics_report()


def test_drop_order_and_threads(raw_file):
    def slow_odd(item):
        time.sleep(0.05 if item.index % 2 else 0.0)  # odd items finish later
        return item if item.index != 2 else None    # item 2 is dropped

    workflow = opennve.Workflow.from_steps(
        "order",
        [{"type": "raw", "path": str(raw_file), "block_ms": 100}, {"type": "callback", "function": slow_odd, "threads": 4}, {"type": "collect"}],
    )
    workflow.run()
    assert [item.index for item in workflow.results] == [0, 1, 3, 4, 5, 6, 7, 8, 9]
    assert workflow.statistics()["steps"][1]["dropped"] == 1

    unordered = opennve.Workflow.from_steps(
        "unordered",
        [{"type": "raw", "path": str(raw_file), "block_ms": 100}, {"type": "callback", "function": slow_odd, "threads": 4}, {"type": "collect"}],
        ordered=False,
    )
    unordered.run()
    assert sorted(item.index for item in unordered.results) == [0, 1, 3, 4, 5, 6, 7, 8, 9]


def test_errors_and_stop(raw_file):
    def broken(item):
        if item.index == 3:
            raise RuntimeError("model failed")
        return item

    workflow = opennve.Workflow.from_steps(
        "broken", [{"type": "raw", "path": str(raw_file), "block_ms": 10}, {"type": "callback", "function": broken, "threads": 2}, {"type": "collect"}]
    )
    with pytest.raises(RuntimeError, match="model failed"):
        workflow.run()

    # stop: the source ends, the items already read are processed
    long_file = raw_file.parent / "long.raw"
    events = _random_events(12, count=400_000, span=20_000_000, width=64, height=32)
    long_file.write_bytes(_header("EVT3;height=32;width=64") + encode_evt3(events))
    workflow = opennve.Workflow.from_steps(
        "stop", [{"type": "raw", "path": str(long_file), "block_ms": 10, "speed": 1.0}, {"type": "collect"}]
    )
    workflow.start()
    time.sleep(0.5)
    workflow.stop()
    assert workflow.join(timeout=10)
    count = len(workflow.results)
    assert 0 < count < 1000  # 20 s at real time: stopped long before the end (2000 blocks)
    with pytest.raises(RuntimeError, match="already started"):
        workflow.start()


def pixel_model(batch):
    on = batch[..., 1] > 0
    logits = np.zeros(batch.shape[:3] + (3,), np.float32)
    logits[..., 0] = 1.0
    logits[..., 2] = np.where(on, 5.0, -1.0)
    return logits


def test_label_events_and_write(raw_file, tmp_path):
    # register an agent backed by a Python model through a custom backend
    class PixelBackend(opennve.agents.Backend):
        def __init__(self, config):
            self.input = opennve.agents.ModelInput((None, 16, 16, 2), np.dtype(np.float32))

        def predict(self, batch):
            return [pixel_model(batch)]

    opennve.register_backend("pixel_test", PixelBackend, suffixes=[".pixeltest"])
    (tmp_path / "model.pixeltest").write_text("")
    agents = [{"name": "insects", "model": "model.pixeltest", "frame": "t16", "prediction": {
        "task": "segmentation", "activation": "softmax", "labels": ["background", "noise", "insect"], "target_classes": [2]}}]
    workflows = tmp_path / "workflows.json"
    workflows.write_text(json.dumps({"workflows": [{
        "name": "insects",
        "frames": [{"name": "t16", "mode": "binary", "tiles": {"size": 16, "min_events": 1}}],
        "agents": agents,
        "steps": [
            {"type": "raw", "path": raw_file.name, "block_ms": 500},
            {"type": "windows", "count": 2, "overlap_ms": 100},
            {"type": "frames", "frame": "t16", "threads": 2},
            {"type": "agent", "agent": "insects", "threads": 2},
            {"type": "require", "agent": "insects", "min_windows": 2},
            {"type": "label_events", "agent": "insects", "label": 2},
            {"type": "write", "format": "csv", "path": "out/{workflow}_{camera}.csv", "camera_column": True},
            {"type": "write", "format": "jsonl", "path": "out/{workflow}.jsonl"},
            {"type": "collect"},
        ],
    }]}))
    workflow = opennve.load_workflow(workflows)
    workflow.run()
    items = workflow.results
    assert len(items) == 2
    item = items[0]
    # every ON event is labeled 2 (the model marks pixels with ON events in a window), no OFF-only event
    events = item.block.polarity_events
    on_positions = set()
    for window in item.windows:
        on = window.polarity_events[window.polarity_events["on"]]
        on_positions |= {(w, int(x), int(y)) for w, x, y in zip([window.start_t] * len(on), on["x"], on["y"])}
    labeled = item.events
    assert labeled.dtype.names == ("x", "y", "p", "t", "l") and set(labeled["l"]) == {2}
    expected = 0
    for event in events:
        for window in item.windows:
            if window.start_t <= event["t"] < window.end_t and (window.start_t, int(event["x"]), int(event["y"])) in on_positions:
                expected += 1
                break
    assert len(labeled) == expected and expected > 0
    # CSV: header once, camera column, all labeled events of both items
    path = tmp_path / "out" / "insects_rec.csv"
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["camera", "x", "y", "p", "t", "l"] and len(rows) == 1 + sum(len(i.events) for i in items)
    assert rows[1][0] == "rec" and rows[1][-1] == "2"
    lines = (tmp_path / "out" / "insects.jsonl").read_text().splitlines()
    summary = json.loads(lines[0])
    assert summary["index"] == 0 and len(summary["windows"]) == 2
    assert summary["windows"][0]["agents"]["insects"]["predictions"][0]["pixels"][0]["scores"]["insect"] > 0.9


@needs_tflite
def test_tflite_agent_threads(raw_file, tmp_path):
    workflow = opennve.Workflow.from_steps(
        "tflite",
        [
            {"type": "raw", "path": str(raw_file), "block_ms": 100},
            {"type": "frames", "frame": {"name": "t16", "tiles": {"size": 16, "min_events": 0}}},
            {"type": "agent", "agent": {"name": "tiles", "model": str(DATA / "agent_tiles.tflite")}, "threads": 3},
            {"type": "collect"},
        ],
    )
    workflow.run()
    results = [item.results[0]["tiles"] for item in workflow.results]
    assert len(results) == 10 and all(len(result) == 8 for result in results)  # 64 x 32 / 16 x 16
    reference = opennve.Agent({"model": str(DATA / "agent_tiles.tflite")}).predict(workflow.results[4].frames[0]["t16"])
    np.testing.assert_allclose(results[4].outputs[0], reference.outputs[0], rtol=1e-6)


def test_custom_step_and_cli(raw_file, tmp_path, capsys):
    @opennve.register_step("count_on")
    class CountOn(opennve.Step):
        PARAMETERS = {"key": "on"}

        def process(self, item):
            item.data[self.params["key"]] = int(item.block.polarity_events["on"].sum())
            return item

    path = tmp_path / "workflows.json"
    path.write_text(json.dumps({"workflows": [{"name": "custom", "default": True, "steps": [
        {"type": "raw", "path": raw_file.name, "block_ms": 500},
        {"type": "count_on", "threads": 2},
        {"type": "log", "timings": False},
        {"type": "write", "format": "jsonl", "path": "{workflow}.jsonl"},
    ]}]}))
    from opennve.cli import main

    assert main(["check", str(path)]) == 0
    assert "workflow custom (default)" in capsys.readouterr().out
    assert main(["workflow", "run", str(path), "--no-input"]) == 0
    output = capsys.readouterr().out
    assert "[custom] block 0 rec | events" in output and "count_on" in output
    assert len((tmp_path / "custom.jsonl").read_text().splitlines()) == 2


def test_windows_align_first_event_and_ordered_log(raw_file, capsys):
    def slow(item):
        time.sleep(0.03 if item.index % 2 else 0.0)
        return item

    workflow = opennve.Workflow.from_steps(
        "align",
        [
            {"type": "raw", "path": str(raw_file), "block_ms": 100},
            {"type": "windows", "count": 3, "overlap_ms": 20, "align": "first_event"},
            {"type": "callback", "function": slow, "threads": 3},
            {"type": "log", "timings": False},
            {"type": "collect"},
        ],
    )
    workflow.run()
    item = workflow.results[2]
    first = int(item.block.polarity_events["t"][0])
    assert item.windows[0].start_t == first and first > item.block.start_t
    # the windows cover the block length from the first event (as the previous build_bounds)
    assert [(w.start_t - first, w.end_t - first) for w in item.windows] == [(0, 46667), (26667, 73334), (53333, 100000)]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[align] block")]
    assert [int(line.split()[2]) for line in lines] == list(range(10))  # log in order
    with pytest.raises(ValueError, match="align must be"):
        opennve.Workflow.from_steps("x", [{"type": "raw", "path": "a.raw"}, {"type": "windows", "count": 2, "align": "end"}])


# ------------------------------------------------------------------ automatic threads


def test_cpu_count_and_system_info(monkeypatch):
    monkeypatch.setenv("OPENNVE_CPUS", "6")
    assert opennve.cpu_count() == 6
    info = opennve.system_info()
    assert info["usable_cpus"] == 6 and info["usable_cpus_source"] == "OPENNVE_CPUS" and info["logical_cpus"] >= 1
    monkeypatch.setenv("OPENNVE_CPUS", "nonsense")
    assert opennve.cpu_count() >= 1 and opennve.system_info()["usable_cpus_source"] != "OPENNVE_CPUS"
    json.dumps(info)


def _thread_steps(workflow):
    return {step["name"]: step for step in workflow.thread_plan()["steps"]}


@needs_tflite
def test_automatic_threads(raw_file):
    tiles = {"name": "t16", "mode": "binary", "tiles": {"size": 16, "min_events": 1}}
    agent = {"name": "tiles", "model": str(DATA / "agent_tiles.tflite"), "frame": "t16"}
    steps = [{"type": "raw", "path": str(raw_file), "block_ms": 250}, {"type": "windows", "count": 1},
             {"type": "frames", "frame": "t16"}, {"type": "agent", "agent": "tiles"}, {"type": "collect"}]

    def workflow(cpus, agents=(agent,), step_threads=None, extra=()):
        agent_step = {**steps[3], **({"threads": step_threads} if step_threads else {})}
        return opennve.Workflow.from_steps("auto", [*steps[:3], agent_step, *extra, steps[4]], frames=[tiles],
                                           agents=list(agents), cpus=cpus)

    # defaults ("auto"): the agent gets all cores with one interpreter thread each
    plan = _thread_steps(workflow(12))
    assert plan["raw"]["threads"] == 1 and plan["frames"]["threads"] == 2 and plan["frames"]["auto"]
    assert plan["agent"]["threads"] == 12 and plan["agent"]["interpreter_threads"] == {"tiles": 1}
    assert workflow(12).queue_size == 24 and workflow(2).queue_size == 8
    assert _thread_steps(workflow(4))["frames"]["threads"] == 1
    assert _thread_steps(workflow(64))["agent"]["threads"] == 16  # capped, ...
    assert _thread_steps(workflow(64))["agent"]["interpreter_threads"] == {"tiles": 4}  # ... the rest to the interpreters
    # fixed step threads: the interpreters share the cores
    plan = _thread_steps(workflow(8, step_threads=2))
    assert plan["agent"]["threads"] == 2 and not plan["agent"]["auto"] and plan["agent"]["interpreter_threads"] == {"tiles": 4}
    # fixed interpreter threads: the step threads follow
    plan = _thread_steps(workflow(8, agents=[{**agent, "threads": 2}]))
    assert plan["agent"]["threads"] == 4 and plan["agent"]["interpreter_threads"] == {"tiles": 2}
    # two agent steps share the cores
    second = {"type": "agent", "agent": "tiles", "name": "second"}
    plan = _thread_steps(workflow(8, extra=[second]))
    assert plan["agent"]["threads"] == 4 and plan["second"]["threads"] == 4
    # single-thread steps stay at one thread with "auto"
    assert _thread_steps(opennve.Workflow.from_steps("w", [{**steps[0], "threads": "auto"}, steps[4]]))["raw"]["threads"] == 1
    assert "agent            x12 (auto), interpreter threads tiles 1" in workflow(12).thread_report()
    # the workflow runs with the planned threads and every agent instance has its interpreter threads
    run = workflow(3).run()
    assert len(run.results) == 4 and run.statistics()["cpus"] == 3
    agents = [instance for stage in run._stages for instance in stage.instances if type(instance).__name__ == "AgentStep"]
    assert len(agents) == 3 and all(step.configs[0].threads == 1 for step in agents)
    with pytest.raises(ValueError, match="threads"):
        opennve.Workflow.from_steps("w", [steps[0], {"type": "frames", "frame": "t16", "threads": "many"}, steps[4]], frames=[tiles])
    with pytest.raises(ValueError, match="cpus"):
        opennve.Workflow.from_steps("w", [steps[0], steps[4]], cpus=0)


@needs_tflite
def test_agent_threads_auto(monkeypatch):
    monkeypatch.setenv("OPENNVE_CPUS", "3")
    agent = opennve.Agent(opennve.AgentConfig(model=str(DATA / "agent_tiles.tflite"), threads="auto")).load()
    assert agent.backend.threads == 3
    monkeypatch.setenv("OPENNVE_CPUS", "32")
    assert opennve.Agent(opennve.AgentConfig(model=str(DATA / "agent_tiles.tflite"))).load().backend.threads == 4
    assert opennve.Agent(opennve.AgentConfig(model=str(DATA / "agent_tiles.tflite"), threads=2)).load().backend.threads == 2
    assert opennve.AgentConfig(model="m.tflite", threads="auto").to_dict()["threads"] == "auto"
    with pytest.raises(ValueError, match="threads"):
        opennve.AgentConfig(model="m.tflite", threads="all")


def test_system_info_memory_and_devices(tmp_path, monkeypatch):
    from opennve import system

    info = opennve.system_info()
    for key in ("memory", "gpus", "npus", "model"):
        assert key in info
    assert info["memory"]["total"] is None or info["memory"]["total"] > 0
    assert opennve.system_info(devices=False)["gpus"] == []
    json.dumps(info)
    assert system.format_bytes(16 * 1024**3) == "16.0 GB" and system.format_bytes(None) == "-"

    # Linux: PCI GPU (AMD with VRAM), Raspberry Pi GPU, Intel NPU, Hailo
    monkeypatch.setattr(system, "_nvidia_smi", lambda: [])

    def device(name, driver, vendor=None, **files):
        path = tmp_path / "devices" / name
        (path / "drivers").mkdir(parents=True)
        (path / "uevent").write_text(f"DRIVER={driver}\n")
        if vendor:
            (path / "vendor").write_text(vendor + "\n")
        for key, value in files.items():
            (path / key).write_text(value)
        return path

    def link(cls, name, target):
        (tmp_path / "sys/class" / cls).mkdir(parents=True, exist_ok=True)
        (tmp_path / "sys/class" / cls / name).mkdir()
        (tmp_path / "sys/class" / cls / name / "device").symlink_to(target)

    link("drm", "card0", device("amd", "amdgpu", "0x1002", mem_info_vram_total=str(8 * 1024**3)))
    (tmp_path / "sys/class/drm/card0-HDMI-A-1").mkdir()
    link("drm", "card1", device("v3d", "v3d"))
    link("accel", "accel0", device("npu", "intel_vpu", "0x8086"))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/hailo0").write_text("")
    found = system._linux_devices(tmp_path)
    assert [(d["type"], d["name"], d["vendor"]) for d in found] == [
        ("gpu", "AMD GPU (amdgpu)", "AMD"),
        ("gpu", "VideoCore GPU (V3D)", "Broadcom"),
        ("npu", "Hailo AI accelerator", "Hailo"),
        ("npu", "Intel NPU (AI Boost)", "Intel"),
    ]
    assert found[0]["memory"] == 8 * 1024**3

    # Windows: graphics adapters and compute accelerators (WMI as JSON)
    monkeypatch.setattr(system, "_powershell", lambda script: json.dumps({
        "gpus": [{"Name": "Intel(R) Arc(TM) Graphics", "AdapterCompatibility": "Intel Corporation", "AdapterRAM": 2147483648},
                 {"Name": "Microsoft Basic Display Adapter", "AdapterCompatibility": "Microsoft", "AdapterRAM": 0}],
        "npus": {"Name": "Intel(R) AI Boost", "Manufacturer": "Intel Corporation"},
    }))
    found = system._windows_devices()
    assert [(d["type"], d["name"]) for d in found] == [("gpu", "Intel(R) Arc(TM) Graphics"), ("npu", "Intel(R) AI Boost")]

    # macOS (Apple silicon): GPU from system_profiler, Neural Engine
    monkeypatch.setattr(system, "_run", lambda command, timeout=5: json.dumps({"SPDisplaysDataType": [
        {"_name": "Apple M2", "sppci_model": "Apple M2", "sppci_cores": "10", "spdisplays_vendor": "sppci_vendor_Apple"}]}))
    monkeypatch.setattr(system.platform, "machine", lambda: "arm64")
    found = system._macos_devices()
    assert [(d["type"], d["name"]) for d in found] == [("gpu", "Apple M2 (10 cores)"), ("npu", "Apple Neural Engine")]


def test_system_processor_names_and_virtual_gpus(tmp_path, monkeypatch):
    from opennve import system

    cpuinfo = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\n\nprocessor\t: 1\nCPU implementer\t: 0x41\nCPU part\t: 0xd05\nModel\t: Raspberry Pi 5\n"
    monkeypatch.setattr(system, "_read", lambda path: cpuinfo if "cpuinfo" in str(path) else "")
    assert system._linux_processor() == "ARM Cortex-A76 + ARM Cortex-A55"
    monkeypatch.undo()
    monkeypatch.setattr(system, "_nvidia_smi", lambda: [])
    path = tmp_path / "devices/hyperv"
    path.mkdir(parents=True)
    (path / "uevent").write_text("DRIVER=hyperv_drm\n")
    (tmp_path / "sys/class/drm/card0").mkdir(parents=True)
    (tmp_path / "sys/class/drm/card0/device").symlink_to(path)
    assert system._linux_devices(tmp_path) == []
    monkeypatch.setattr(system, "_powershell", lambda script: json.dumps({"gpus": {"Name": "Microsoft Hyper-V Video"}, "npus": None}))
    assert system._windows_devices() == []


@needs_tflite
def test_gpu_agents_get_two_threads(raw_file):
    tiles = {"name": "t16", "mode": "binary", "tiles": {"size": 16, "min_events": 1}}
    agent = {"name": "tiles", "model": str(DATA / "agent_tiles.tflite"), "frame": "t16", "device": "gpu"}
    steps = [{"type": "raw", "path": str(raw_file), "block_ms": 250}, {"type": "windows", "count": 1},
             {"type": "frames", "frame": "t16"}, {"type": "agent", "agent": "tiles"}, {"type": "collect"}]
    workflow = opennve.Workflow.from_steps("gpu", steps, frames=[tiles], agents=[agent], cpus=8)
    plan = {step["name"]: step for step in workflow.thread_plan()["steps"]}
    assert plan["agent"]["threads"] == 2 and plan["agent"]["devices"] == {"tiles": "gpu"}
    assert "device tiles gpu" in workflow.thread_report()
    run = opennve.Workflow.from_steps("gpu", steps, frames=[tiles], agents=[agent], cpus=2).run()
    assert len(run.results) == 4  # runs on the GPU or falls back to the CPU


def test_os_info(monkeypatch):
    from opennve import system

    info = opennve.os_info()
    assert info["name"] and info["family"] in ("Linux", "macOS", "Windows") and info["kernel"]
    assert opennve.system_info(devices=False)["os"] == info
    monkeypatch.setattr(system.sys, "platform", "darwin")
    monkeypatch.setattr(system.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    monkeypatch.setattr(system, "_run", lambda command, timeout=5: "24B83\n")
    assert system.os_info() == {"family": "macOS", "name": "macOS 15.1 Sequoia", "version": "15.1", "build": "24B83",
                                "kernel": "Darwin " + system.platform.release()}
