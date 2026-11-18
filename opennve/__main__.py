"""Command line interface.

    python -m opennve list [--json] [--config cameras.json]
    python -m opennve template SE1-S4 [--serial 05110634] [--name left]
    python -m opennve check cameras.json
    python -m opennve models
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import CameraConfig, load_configs
from .discovery import list_cameras
from .models import MODELS


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


def _check(args) -> int:
    configs = load_configs(args.path)
    for config in configs:
        print(f"{config.name}: {config.model.id} {config.serial or '(first available)'}")
    print(f"{len(configs)} camera(s), configuration is valid")
    return 0


def _models(args) -> int:
    for model in MODELS:
        print(f"{model.id:<16} {model.name}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="opennve", description="openNVE camera tools")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("list", help="list connected cameras")
    command.add_argument("--json", action="store_true", help="output JSON")
    command.add_argument("--config", help="configuration file, adds the camera names")
    command.set_defaults(function=_list)

    command = commands.add_parser("template", help="print a default JSON configuration")
    command.add_argument("model", help="camera model, for instance SE1-S4")
    command.add_argument("--serial")
    command.add_argument("--name")
    command.set_defaults(function=_template)

    command = commands.add_parser("check", help="validate a configuration file")
    command.add_argument("path")
    command.set_defaults(function=_check)

    command = commands.add_parser("models", help="list supported camera models")
    command.set_defaults(function=_models)

    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
