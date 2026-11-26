"""Command line interface.

    python -m opennve list [--json] [--config cameras.json]
    python -m opennve template SE1-S4 [--serial 05110634] [--name left]
    python -m opennve frame-template [--name tiles80] [--width 1280 --height 720] [--tiles 80]
    python -m opennve check cameras.json frames.json agents.json workflows.json [--load]
    python -m opennve workflow run workflows.json [--name insects] [--duration 60] [--debug]
    python -m opennve workflow list workflows.json
    python -m opennve models
    python -m opennve driver [--install] [--model SE1-S4] [--force]
    python -m opennve raw recording.raw [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from .cameras.config import CameraConfig, load_configs
from .frames import FrameConfig, TileConfig, TimeSurfaceConfig, frame_configs_from_data
from .agents import Agent, agent_configs_from_data
from .cameras.discovery import list_cameras
from .cameras.models import MODELS


def _list(args) -> int:
    configs = load_configs(args.config) if args.config else []
    cameras = list_cameras(configs)
    if args.json:
        print(json.dumps([camera.to_dict() for camera in cameras], indent=4))
        return 0
    if not cameras:
        print("No camera found")
        return 0
    for camera in cameras:
        name = f" ({camera.name})" if camera.name else ""
        state = "ok" if camera.available else f"error: {camera.error}"
        print(f"{camera.model:<10} {camera.serial or '-':<10} {camera.connection:<32} {state}{name}")
    return 0


def _template(args) -> int:
    config = CameraConfig(model=args.model, serial=args.serial, name=args.name)
    print(config.to_json())
    return 0


def _frame_template(args) -> int:
    config = FrameConfig(
        name=args.name,
        width=args.width,
        height=args.height,
        mode=args.mode,
        tiles=None if args.tiles is None else TileConfig(size=args.tiles),
        time_surface=TimeSurfaceConfig(decay=args.decay),
    )
    config.validate()
    print(json.dumps({"frames": [config.to_dict()]}, indent=4))
    return 0


def _describe_frame(config: FrameConfig) -> str:
    size = f"{config.width or 'sensor'} x {config.height or 'sensor'}"
    mode = config.mode + (f" ({config.time_surface.decay})" if config.mode == "time_surface" else "")
    tiles = f"tiles {config.tiles.size} px" if config.tiles else "no tiles"
    return f"{size}, {mode}, {tiles}"


def _workflow(args) -> int:
    from .workflows import Workflow, load_workflow_config, load_workflow_configs

    if args.action == "list":
        for config in load_workflow_configs(args.path):
            default = " (default)" if config.default else ""
            print(f"{config.name}{default}: {' -> '.join(step['type'] for step in config.steps)}")
        return 0
    config = load_workflow_config(args.path, args.name)
    if args.debug:
        config.debug = True
    if args.cpus is not None:
        config.cpus = args.cpus
    workflow = Workflow(config)
    if args.action == "threads":
        print(workflow.thread_report())
        return 0
    print(workflow.describe())
    print(workflow.thread_report())
    if not args.no_input:
        import sys
        import threading

        print("Enter 'stop' (or press Ctrl+C) to stop.", flush=True)

        def listen():
            for line in sys.stdin:
                if line.strip().lower() in ("stop", "q", "quit", "exit"):
                    print("stopping ...", flush=True)
                    workflow.stop()
                    return

        threading.Thread(target=listen, daemon=True).start()
    workflow.run(duration_s=args.duration)
    print(workflow.statistics_report())
    return 0


def _version(args) -> int:
    from ._version import version_info

    from .plugins import installed_plugins

    info = version_info()
    info["plugins"] = {name: plugin.version or "?" for name, plugin in installed_plugins().items()}
    if args.json:
        print(json.dumps(info, indent=4))
        return 0
    print(f"openNVE {info['version']} (build {info['build']})")
    source = {"wheel": "", "git": ", from the last git commit", "today": ", not a built wheel"}[info["build_source"]]
    print(f"  build date            {info['build_date']}{source}")
    for key, label in (
        ("neuromorphic_drivers", "neuromorphic_drivers"),
        ("python", "Python"),
        ("platform", "platform"),
        ("numpy", "numpy"),
        ("opencv", "OpenCV"),
        ("h5py", "h5py"),
        ("litert", "LiteRT"),
        ("flatbuffers", "flatbuffers"),
    ):
        print(f"  {label:<21} {info[key] or '-'}")
    print("  optional (extras):")
    for key, label, extra in (
        ("tensorflow", "TensorFlow", "tensorflow"),
        ("pytorch", "PyTorch", "pytorch"),
        ("pandas", "pandas", "pandas"),
        ("tables", "PyTables", "pandas"),
        ("hdf5plugin", "hdf5plugin", "hdf5"),
        ("npu_intel", "NPU Intel SDK", "npu-intel"),
        ("npu_qualcomm", "NPU Qualcomm SDK", "npu-qualcomm"),
    ):
        hint = f"extra {extra}"
        print(f"  {label:<21} {info[key] or f'- ({hint})'}")
    print("  plugins:" + ("" if info["plugins"] else "               - (opennve plugin list)"))
    for name, version in info["plugins"].items():
        print(f"  {name:<21} {version}")
    return 0


def _system(args) -> int:
    from .system import format_bytes, system_info

    info = system_info()
    if args.json:
        print(json.dumps(info, indent=4))
        return 0
    memory = info["memory"]
    ram = format_bytes(memory["total"])
    if memory["available"] is not None:
        ram += f" ({format_bytes(memory['available'])} available)"
    if memory["limit"] is not None:
        ram += f", limit {format_bytes(memory['limit'])}"

    def devices(items):
        if not items:
            return ["none found"]
        lines = []
        for item in items:
            details = [value for value in (format_bytes(item["memory"]) if item["memory"] else None, item["driver"]) if value]
            lines.append(item["name"] + (f" ({', '.join(details)})" if details else ""))
        return lines

    system = info["os"]
    build = f", build {system['build']}" if system.get("build") else ""
    print(f"{system['name']} ({info['machine']}{build}), Python {info['python']}")
    print(f"  kernel                {system['kernel']}")
    if info["model"]:
        print(f"  model                 {info['model']}")
    print(f"  processor             {info['processor']}")
    print(f"  logical CPUs          {info['logical_cpus']}")
    print(f"  physical cores        {info['physical_cores'] or '-'}")
    print(f"  usable by openNVE     {info['usable_cpus']} ({info['usable_cpus_source']})")
    print(f"  RAM                   {ram}")
    for label, items in (("GPU", info["gpus"]), ("NPU", info["npus"])):
        for number, line in enumerate(devices(items)):
            print(f"  {label if number == 0 else '':<22}{line}")
    litert = info.get("litert") or {}
    if litert.get("version"):
        parts = [f"GPU via {litert['gpu']}" if litert.get("gpu") else "no GPU accelerator"]
        for npu in litert.get("npu", []):
            parts.append(f"NPU {npu['name']}" + ("" if npu["sdk_installed"] else f" (needs pip install {npu['install']})"))
        if not litert.get("compiled_model"):
            parts = ["CPU only (GPU / NPU need ai-edge-litert 2.4 or newer)"]
        print(f"  LiteRT {litert['version']:<15}{', '.join(parts)}")
    else:
        print("  LiteRT                not installed")
    pytorch = info.get("pytorch") or {}
    if pytorch.get("installed"):
        gpu = {"cuda": "CUDA", "mps": "Apple MPS", "xpu": "Intel XPU"}.get(pytorch.get("gpu"), "")
        if pytorch.get("gpu_name"):
            gpu += f" ({pytorch['gpu_name']})"
        print(f"  PyTorch {pytorch['version']:<14}{'GPU via ' + gpu if gpu else 'CPU only (no GPU found)'}")
    from .plugins import catalog, installed_plugins

    plugins = installed_plugins()
    for name, installed in plugins.items():
        plugin = installed.plugin
        if plugin is not None and plugin.system_lines is not None:
            try:
                for line in plugin.system_lines(info.get("plugins", {}).get(name, {})):
                    print(f"  {line}")
            except Exception as error:
                print(f"  plugin {name}: {error}")
    # hardware that a plugin not yet installed would use
    if any(device.get("vendor") == "Hailo" for device in info.get("npus", [])) and "hailo" not in plugins:
        print("  Hailo                 plugin not installed: opennve plugin install hailo")
    kria = any(device.get("driver") == "zocl" for device in info.get("npus", [])) or any(
        name in (info.get("model") or "") for name in ("Kria", "K26", "K24", "KV260", "KR260", "KD240"))
    if kria and "vitis" not in plugins:
        print("  AMD Kria DPU          plugin not installed: opennve plugin install vitis")
    return 0


def _benchmark(args) -> int:
    from .agents import benchmark_agent, benchmark_report, load_agent_config
    from .frames import load_frame_config

    config = load_agent_config(args.agents, args.agent)
    frames = None
    if args.recording:
        if not args.frames:
            print("--recording needs --frames (the frame configuration file)", file=sys.stderr)
            return 2
        from .recordings import open_recording

        frame_config = load_frame_config(args.frames, args.frame or config.frame)
        frames = []
        with open_recording(args.recording, window={"duration_ms": args.window_ms}) as recording:
            for window in recording.windows():
                frame = window.frame(frame_config)
                if config.input != "tiles" or frame.tiles:
                    frames.append(frame)
                if len(frames) >= args.windows:
                    break
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    print(f"agent {config.name}: {config.model_path.name}")
    results = benchmark_agent(config, devices, frames=frames, count=args.inputs)
    if args.json:
        print(json.dumps(results, indent=4))
    else:
        print(benchmark_report(results))
    return 0


def _quantize(args) -> int:
    import glob

    from .agents import load_agent_config, quantize_agent, quantize_report
    from .frames import load_frame_config

    config = load_agent_config(args.agents, args.agent)
    if config.frame is None and not args.frame:
        print("the agent has no \"frame\": give the frame configuration with --frame", file=sys.stderr)
        return 2
    recordings = []
    for pattern in args.data:
        matches = sorted(glob.glob(pattern, recursive=True))
        recordings.extend(matches or [pattern])
    missing = [path for path in recordings if not os.path.exists(path)]
    if missing:
        print(f"not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    frame_config = load_frame_config(args.frames, args.frame or config.frame)
    progress = None if args.json else (lambda message: print(message, flush=True))
    report = quantize_agent(
        config, recordings, frame_config, args.output, samples=args.samples, validation=args.validation,
        window_ms=args.window_ms, stride=args.stride, max_windows=args.max_windows, int8_io=args.int8_io,
        qat=args.qat, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        target_weight=args.target_weight, seed=args.seed, sampling=args.sampling, width=args.width,
        height=args.height, target_share=args.target_share, progress=progress,
    )
    if args.json:
        print(json.dumps(report, indent=4))
    else:
        print(quantize_report(report))
        output = pathlib.Path(report["output"])
        try:
            model = os.path.relpath(output, pathlib.Path(args.agents).resolve().parent)
        except ValueError:
            model = str(output)
        print(f"\nagent with the quantized model: \"model\": \"{pathlib.Path(model).as_posix()}\" "
              f"(opennve benchmark compares it on CPU / GPU / NPU)")
    return 0


def _check(args) -> int:
    for path in args.paths:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and "workflows" in data:
            from .workflows import Workflow, load_workflow_configs

            print(path)
            for config in load_workflow_configs(path):
                print("  " + Workflow(config).describe().replace("\n", "\n  "))
            print("  configuration is valid")
            continue
        has_frames = isinstance(data, dict) and "frames" in data
        has_agents = isinstance(data, dict) and "agents" in data
        has_cameras = (not has_frames and not has_agents) or (isinstance(data, dict) and "cameras" in data)
        print(path)
        if has_cameras:
            configs = load_configs(path)
            for config in configs:
                print(f"  camera {config.name}: {config.model.id} {config.serial or '(first available)'}")
        if has_frames:
            frames = frame_configs_from_data(data, path)
            for frame in frames:
                print(f"  frame {frame.name}: {_describe_frame(frame)}")
        if has_agents:
            import pathlib

            agents = agent_configs_from_data(data, path, pathlib.Path(path).resolve().parent)
            for config in agents:
                frame = f", frame {config.frame}" if config.frame else ""
                print(f"  agent {config.name}: {config.model} ({config.resolved_backend}), input {config.input}{frame}")
                if args.load:
                    agent = Agent(config).load()
                    print(f"    model input {list(agent.input.shape)} {agent.input.dtype}")
                    agent.close()
        print("  configuration is valid")
    return 0


def _driver(args) -> int:
    from .cameras.driver import driver_status, install_driver

    if args.install:
        return 0 if install_driver(args.model, force=args.force) else 1
    statuses = driver_status(args.model)
    if not statuses:
        print("No camera found")
    for status in statuses:
        state = "ready" if status.ready else f"not ready: {status.message}"
        usb_id = f" ({status.usb_id})" if status.usb_id else ""
        driver = f" driver: {status.driver or '(none)'}" if status.usb_id else ""
        print(f"{status.device}{usb_id}{driver} | {state}")
    return 0 if all(status.ready for status in statuses) else 1


def _plugin(args) -> int:
    from .plugins import catalog, catalog_entry, installed_plugins, manager

    if args.action == "install":
        if not args.name:
            print("which plugin? opennve plugin list", file=sys.stderr)
            return 2
        try:
            return manager.install(args.name, args.extra, ref=args.ref, source=args.source)
        except KeyError as error:
            print(error.args[0], file=sys.stderr)
            return 2
    if args.action == "remove":
        return manager.remove(args.name) if args.name else 2
    installed = installed_plugins(reload=True)
    if args.action == "info":
        entry = catalog_entry(args.name) if args.name else None
        plugin = installed.get(args.name) if args.name else None
        if entry is None and plugin is None:
            print(f"unknown plugin {args.name!r}", file=sys.stderr)
            return 1
        print(f"{args.name}: {(plugin.description if plugin else entry.description)}")
        if plugin:
            state = f"installed {plugin.version or ''} ({plugin.distribution})"
            print(f"  state:       {state}" + (f", cannot be loaded: {plugin.error}" if plugin.error else ""))
        else:
            print("  state:       not installed" + ("" if entry.supported else f" (not available on {sys.platform})"))
        if entry:
            print(f"  repository:  {entry.repository}")
            print(f"  install:     opennve plugin install {entry.name}  ({entry.method})")
            if entry.backends:
                print(f"  backends:    {', '.join(entry.backends)} ({', '.join(entry.suffixes)} models)")
            if entry.commands:
                print(f"  commands:    {', '.join('opennve ' + command for command in entry.commands)}")
            if entry.notes:
                print(f"  notes:       {entry.notes}")
        return 0
    if args.json:
        result = {
            "installed": [{"name": p.name, "version": p.version, "distribution": p.distribution, "error": p.error,
                           "description": p.description} for p in installed.values()],
            "available": [{"name": e.name, "description": e.description, "installed": e.name in installed,
                           "supported": e.supported, "repository": e.repository} for e in catalog()],
        }
        print(json.dumps(result, indent=2))
        return 0
    print("Installed plugins:")
    if not installed:
        print("  (none)")
    for plugin in installed.values():
        state = f"{plugin.version or ''}" + (f"  cannot be loaded: {plugin.error}" if plugin.error else "")
        print(f"  {plugin.name:<12} {state:<10} {plugin.description}")
    print("Installable plugins (opennve plugin install <name>):")
    available = [entry for entry in catalog() if entry.name not in installed]
    if not available:
        print("  (all installed)")
    for entry in available:
        note = "" if entry.supported else f"  [not on {sys.platform}]"
        print(f"  {entry.name:<12} {entry.description}{note}")
    return 0


def _raw(args) -> int:
    import numpy as np

    from .recordings import HDF5_SUFFIXES, is_hdf5, open_hdf5, open_raw

    path = pathlib.Path(args.path)
    if path.suffix.lower() in HDF5_SUFFIXES or (path.suffix.lower() != ".raw" and is_hdf5(path)):
        reader = open_hdf5(path, dataset=args.dataset, width=args.width, height=args.height)
    else:
        reader = open_raw(path, format=args.format, width=args.width, height=args.height)
    with reader as raw:
        info = raw.info()
        count = on = 0
        triggers = 0
        first_t = last_t = None
        labels: dict[int, int] = {}
        for packet in raw.events():
            events = packet.polarity_events
            if len(events) > 0:
                count += len(events)
                on += int(np.count_nonzero(events["on"]))
                if "l" in events.dtype.names:
                    values, counts = np.unique(events["l"], return_counts=True)
                    for value, number in zip(values.tolist(), counts.tolist()):
                        labels[value] = labels.get(value, 0) + number
                first_t = int(events["t"][0]) if first_t is None else first_t
                last_t = int(events["t"][-1])
            if packet.trigger_events is not None:
                triggers += len(packet.trigger_events)
    duration_s = 0.0 if first_t is None else (last_t - first_t) / 1e6
    info.update(
        {
            "events": count,
            "on_events": on,
            "off_events": count - on,
            "trigger_events": triggers,
            "first_t": first_t,
            "last_t": last_t,
            "duration_s": duration_s,
            "event_rate": count / duration_s if duration_s > 0 else None,
        }
    )
    if labels:
        info["label_counts"] = {str(label): number for label, number in sorted(labels.items())}
    for key in ("filter", "output", "window"):
        info.pop(key)
    if args.json:
        print(json.dumps(info, indent=4))
        return 0
    print(f"{info['path']}")
    print(f"  format:     {info['format']}, {info['resolution']['width']} x {info['resolution']['height']}")
    if "serial" in info:
        print(f"  serial:     {info['serial'] or '-'}")
    if "layout" in info:
        print(f"  dataset:    {info['dataset']} ({info['layout']}, time unit {info['time_unit']})")
    print(f"  size:       {info['size_bytes'] / 1e6:.1f} MB")
    print(f"  events:     {count} ({on} ON, {count - on} OFF), {triggers} trigger events")
    if first_t is not None:
        rate = f", {info['event_rate'] / 1e6:.2f} Mev/s" if info["event_rate"] else ""
        print(f"  time:       {first_t / 1e6:.6f} s to {last_t / 1e6:.6f} s ({duration_s:.3f} s{rate})")
    if labels:
        print("  labels:     " + ", ".join(f"{label}: {number}" for label, number in sorted(labels.items())))
    return 0


def _models(args) -> int:
    for model in MODELS:
        print(f"{model.id:<16} {model.name}")
    return 0


def main(argv=None) -> int:
    from ._version import version_string

    parser = argparse.ArgumentParser(prog="opennve", description="openNVE camera tools")
    parser.add_argument("--version", action="version", version=version_string())
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("version", help="show version, build number and dependencies")
    command.add_argument("--json", action="store_true", help="output as JSON")
    command.set_defaults(function=_version)

    command = commands.add_parser("system", help="show processor, CPU cores, RAM, GPUs and NPUs")
    command.add_argument("--json", action="store_true", help="output as JSON")
    command.set_defaults(function=_system)

    command = commands.add_parser("benchmark", help="compare CPU, GPU and NPU for an agent (speed and results)")
    command.add_argument("agents", help="agent configuration file (agents.json)")
    command.add_argument("--agent", help="agent name (default: the first)")
    command.add_argument("--devices", default="cpu,gpu,npu", help="devices to compare (default cpu,gpu,npu)")
    command.add_argument("--recording", help="take the inputs from this recording (.h5, .hdf5, .raw)")
    command.add_argument("--frames", help="frame configuration file (with --recording)")
    command.add_argument("--frame", help="frame configuration name (default: the agent's \"frame\")")
    command.add_argument("--windows", type=int, default=20, help="time windows to read from the recording (default 20)")
    command.add_argument("--window-ms", type=float, default=40.0, help="window length in ms (default 40)")
    command.add_argument("--inputs", type=int, default=0, help="number of inputs (default: all of the windows, or 64 random)")
    command.add_argument("--json", action="store_true", help="output as JSON")
    command.set_defaults(function=_benchmark)

    command = commands.add_parser(
        "quantize", help="int8 TensorFlow Lite model of an agent, calibrated with recordings (training data)",
        epilog="example: opennve quantize agents.json --agent prototyp --data \"data/*.h5\" --frames frames.json",
    )
    command.add_argument("agents", help="agent configuration file (agents.json); the agent's model is a .keras / .h5 / SavedModel")
    command.add_argument("--agent", help="agent name (default: the first)")
    command.add_argument("--data", nargs="+", required=True, help="recordings with typical data (.h5 / .hdf5 / .raw, globs allowed)")
    command.add_argument("--frames", required=True, help="frame configuration file (frames.json)")
    command.add_argument("--frame", help="frame configuration name (default: the agent's \"frame\")")
    command.add_argument("--output", help="quantized model (default: <model>_int8.tflite next to the model)")
    command.add_argument("--qat", action="store_true",
                         help="quantization aware training: train the U-Net further with the int8 rounding so that it gives "
                              "the same classes as the float network (as accurate as the original; takes longer)")
    command.add_argument("--samples", type=int, default=500,
                         help="inputs for the calibration (default 500), with --qat for the training (default 20000)")
    command.add_argument("--validation", type=int,
                         help="other inputs to compare with the original model (default 200, with --qat 1000; 0: no comparison)")
    command.add_argument("--epochs", type=int, default=5, help="--qat: passes over the training inputs (default 5)")
    command.add_argument("--batch-size", type=int, default=32, help="--qat: inputs per training step (default 32)")
    command.add_argument("--learning-rate", type=float, default=1e-4, help="--qat: start learning rate (default 1e-4, cosine decay)")
    command.add_argument("--target-weight", type=float, default=10.0,
                         help="--qat: weight of the pixels of the agent's target classes in the training (default 10)")
    command.add_argument("--target-share", type=float,
                         help="--qat: share of the training inputs that are tiles in which the float model finds one of "
                              "the agent's target classes (default 0.5 with HDF5 recordings; 0: all at random); the "
                              "comparison inputs stay at random")
    command.add_argument("--sampling", choices=["auto", "random", "sequential"], default="auto",
                         help="random: time windows at random places, read directly (default for HDF5, fast for very "
                              "large files); sequential: every --stride-th window of the whole files (default for .raw)")
    command.add_argument("--stride", type=int, default=5, help="sequential: use every n-th time window (default 5)")
    command.add_argument("--width", type=int, help="sensor width if the HDF5 file does not store it (else estimated)")
    command.add_argument("--height", type=int, help="sensor height if the HDF5 file does not store it (else estimated)")
    command.add_argument("--window-ms", type=float, default=40.0, help="window length in ms (default 40)")
    command.add_argument("--max-windows", type=int, help="use at most this many windows (default: all)")
    command.add_argument("--int8-io", action="store_true", help="int8 inputs and outputs too (default: float32, openNVE converts either way)")
    command.add_argument("--seed", type=int, default=0, help="random seed for choosing the inputs (default 0)")
    command.add_argument("--json", action="store_true", help="output the report as JSON")
    command.set_defaults(function=_quantize)

    command = commands.add_parser("list", help="list connected cameras")
    command.add_argument("--json", action="store_true", help="output JSON")
    command.add_argument("--config", help="configuration file, adds the camera names")
    command.set_defaults(function=_list)

    command = commands.add_parser("template", help="print a default JSON configuration")
    command.add_argument("model", help="camera model, for instance SE1-S4")
    command.add_argument("--serial")
    command.add_argument("--name")
    command.set_defaults(function=_template)

    command = commands.add_parser("frame-template", help="print a JSON frame configuration file")
    command.add_argument("--name", default="frame1")
    command.add_argument("--width", type=int, help="frame width in pixels (default: sensor width)")
    command.add_argument("--height", type=int, help="frame height in pixels (default: sensor height)")
    command.add_argument("--mode", default="time_surface", choices=("time_surface", "binary", "count"))
    command.add_argument("--decay", default="linear", choices=("linear", "exponential"))
    command.add_argument("--tiles", type=int, help="tile size in pixels, for instance 80 (default: no tiles)")
    command.set_defaults(function=_frame_template)

    command = commands.add_parser("workflow", help="list or run workflows, or show their threads")
    command.add_argument("action", choices=("run", "list", "threads"))
    command.add_argument("path", help="workflow configuration file")
    command.add_argument("--name", help="workflow name (default: the default workflow)")
    command.add_argument("--duration", type=float, help="stop after this many seconds")
    command.add_argument("--debug", action="store_true", help="print step timings")
    command.add_argument("--no-input", action="store_true", help="do not read 'stop' from the console")
    command.add_argument("--cpus", type=int, help="size the automatic threads for this many CPU cores")
    command.set_defaults(function=_workflow)

    command = commands.add_parser("check", help="validate camera, frame and agent configuration files")
    command.add_argument("paths", nargs="+")
    command.add_argument("--load", action="store_true", help="also load the agent models")
    command.set_defaults(function=_check)

    command = commands.add_parser("driver", help="show or install the USB driver of the cameras (WinUSB on Windows)")
    command.add_argument("--install", action="store_true", help="install the driver where it is missing")
    command.add_argument("--model", help="only cameras of this model, for instance SE1-S4")
    command.add_argument("--force", action="store_true", help="also replace another (vendor) driver")
    command.set_defaults(function=_driver)

    command = commands.add_parser(
        "plugin", help="list, install or remove plugins (for instance hailo)",
        epilog="options for the installer of a plugin come after --: opennve plugin install hailo -- --yes --reboot",
    )
    command.add_argument("action", nargs="?", choices=["list", "info", "install", "remove"], default="list")
    command.add_argument("name", nargs="?", help="plugin name, see opennve plugin list")
    command.add_argument("--ref", help="install: branch or tag of the plugin (script plugins)")
    command.add_argument("--source", help="install: a local copy of the plugin instead of downloading it")
    command.add_argument("--json", action="store_true", help="list: output as JSON")
    command.set_defaults(function=_plugin)

    # commands of the installed plugins, for instance `opennve hailo`
    from .plugins import installed_plugins as _installed_plugins

    for _name, _installed in _installed_plugins().items():
        if _installed.plugin is not None and _installed.plugin.commands is not None:
            try:
                _installed.plugin.commands(commands)
            except Exception as error:  # a broken plugin must not break the CLI
                print(f"plugin {_name}: commands not available ({error})", file=sys.stderr)

    for command_name in ("recording", "raw"):
        command = commands.add_parser(
            command_name, help="show the information and statistics of a recording (.raw or HDF5 .h5 / .hdf5)"
        )
        _recording_arguments(command)

    command = commands.add_parser("models", help="list supported camera models")
    command.set_defaults(function=_models)

    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:  # opennve plugin install hailo -- --yes: options for the plugin's installer
        extra = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    args = parser.parse_args(argv)
    args.extra = extra
    try:
        return args.function(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _recording_arguments(command):
    command.add_argument("path")
    command.add_argument("--json", action="store_true", help="output JSON")
    command.add_argument("--format", help="event format for files without header (EVT3, EVT2, EVT21)")
    command.add_argument("--width", type=int)
    command.add_argument("--height", type=int)
    command.set_defaults(function=_raw)
    command.add_argument("--dataset", help="HDF5: group or dataset with the events")


if __name__ == "__main__":
    sys.exit(main())
