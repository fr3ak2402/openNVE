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

    info = version_info()
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
        ("litert", "LiteRT"),
        ("tensorflow", "TensorFlow"),
    ):
        print(f"  {label:<21} {info[key] or '-'}")
    return 0


def _system(args) -> int:
    from .system import system_info

    info = system_info()
    if args.json:
        print(json.dumps(info, indent=4))
        return 0
    print(f"{info['platform']} ({info['machine']}), Python {info['python']}")
    print(f"  processor             {info['processor']}")
    print(f"  logical CPUs          {info['logical_cpus']}")
    print(f"  physical cores        {info['physical_cores'] or '-'}")
    print(f"  usable by openNVE     {info['usable_cpus']} ({info['usable_cpus_source']})")
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

    command = commands.add_parser("system", help="show the processor and the CPU cores used for automatic threads")
    command.add_argument("--json", action="store_true", help="output as JSON")
    command.set_defaults(function=_system)

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

    for command_name in ("recording", "raw"):
        command = commands.add_parser(
            command_name, help="show the information and statistics of a recording (.raw or HDF5 .h5 / .hdf5)"
        )
        _recording_arguments(command)

    command = commands.add_parser("models", help="list supported camera models")
    command.set_defaults(function=_models)

    args = parser.parse_args(argv)
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
