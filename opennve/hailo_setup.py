"""Setup of Hailo accelerators: `opennve hailo` (status) and `opennve hailo install`.

The installation is done by install.sh of openNVE-hailo (github.com/fr3ak2402/openNVE-hailo):
on a Raspberry Pi it enables PCIe (Gen 3), installs driver, firmware and HailoRT (apt), Rust,
and builds opennve_hailo for this Python.
"""

from __future__ import annotations

import os
import pathlib
import platform
import shutil
import subprocess
import sys
import typing

REPOSITORY = "https://github.com/fr3ak2402/openNVE-hailo.git"
PCI_CHIPS = {"0x2864": "Hailo-8 / Hailo-8L", "0x45c4": "Hailo-10H"}


def pci_modules(root: str = "") -> list[dict[str, str]]:
    """Hailo modules on PCIe (Linux), also without driver: [{"address", "chip"}]."""
    modules = []
    for device in sorted(pathlib.Path(root + "/sys/bus/pci/devices").glob("*")):
        try:
            if (device / "vendor").read_text().strip() != "0x1e60":
                continue
            device_id = (device / "device").read_text().strip()
        except OSError:
            continue
        modules.append({"address": device.name, "chip": PCI_CHIPS.get(device_id, f"Hailo ({device_id})")})
    return modules


def status() -> dict[str, typing.Any]:
    from .agents._hailo import hailo_info

    driver = os.path.exists("/dev/hailo0")
    return {"pci": pci_modules() if sys.platform.startswith("linux") else [], "driver": driver, "runtime": hailo_info()}


def print_status() -> int:
    info = status()
    if sys.platform.startswith("linux"):
        modules = ", ".join(f"{module['chip']} ({module['address']})" for module in info["pci"])
        print(f"Hailo module (PCIe)   {modules or 'none found'}")
        print(f"Driver                {'loaded (/dev/hailo0)' if info['driver'] else 'not loaded'}")
    runtime = info["runtime"]
    if runtime.get("installed"):
        devices = runtime.get("devices") or []
        chips = ", ".join(device.get("architecture") or "in use" for device in devices)
        print(f"HailoRT               {runtime['version']}")
        print(f"opennve_hailo         {runtime.get('opennve_hailo')}")
        print(f"Devices               {f'{len(devices)} ({chips})' if devices else 'none found'}")
    else:
        print("opennve_hailo         not installed" + (f" ({runtime['error']})" if runtime.get("error") else ""))
    ready = bool(runtime.get("installed") and runtime.get("devices"))
    if not ready:
        print("\nSet up: opennve hailo install")
    return 0 if ready else 1


def source_directory(ref: str = "main", source: typing.Optional[str] = None) -> pathlib.Path:
    """openNVE-hailo: the given directory, or a clone in ~/.cache/opennve (updated to ref)."""
    if source:
        path = pathlib.Path(source).expanduser().resolve()
        if not (path / "install.sh").is_file():
            raise FileNotFoundError(f"{path} is not an openNVE-hailo directory (install.sh missing)")
        return path
    if shutil.which("git") is None:
        raise RuntimeError("git is missing: sudo apt install git")
    path = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "opennve" / "openNVE-hailo"
    if (path / ".git").is_dir():
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "--depth", "1", "origin", ref], check=True)
        subprocess.run(["git", "-C", str(path), "reset", "--quiet", "--hard", "FETCH_HEAD"], check=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", ref, REPOSITORY, str(path)], check=True)
    return path


def install(yes: bool = False, reboot: bool = False, gen3: bool = True, package: typing.Optional[str] = None,
            system: bool = True, ref: str = "main", source: typing.Optional[str] = None) -> int:
    """Runs install.sh of openNVE-hailo for this Python (sys.executable)."""
    if not sys.platform.startswith("linux"):
        print(
            "opennve hailo install is for Linux (Raspberry Pi). Windows: install HailoRT and the driver with "
            "Hailo's Windows installer (Hailo Developer Zone), then build opennve_hailo with maturin, see "
            "https://github.com/fr3ak2402/openNVE-hailo",
            file=sys.stderr,
        )
        return 1
    try:
        path = source_directory(ref, source)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    command = ["bash", str(path / "install.sh")]
    command += ["--yes"] if yes else []
    command += ["--reboot"] if reboot else []
    command += [] if gen3 else ["--no-gen3"]
    command += [f"--package={package}"] if package else []
    command += [] if system else ["--no-system"]
    environment = dict(os.environ, PYTHON=sys.executable)
    print(f"openNVE-hailo {path} ({platform.machine()}), Python {sys.executable}")
    return subprocess.run(command, env=environment).returncode
