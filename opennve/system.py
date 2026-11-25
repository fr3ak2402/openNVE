"""Hardware information for the automatic thread settings.

    opennve.cpu_count()      CPU cores this process may use (affinity, container limits, OPENNVE_CPUS)
    opennve.system_info()    platform, processor, cores, RAM, GPUs and NPUs (also: opennve system)
    opennve.os_info()        operating system: name, version, build, kernel
    opennve.memory_info()    RAM: total, available, container limit
    opennve.accelerators()   GPUs and NPUs found on the machine

Workflows with "threads": "auto" (the default of the frames and agent steps) and agents with
"threads": "auto" (or null) size their threads from cpu_count(). The environment variable
OPENNVE_CPUS (a positive integer) overrides the detected number, for instance to leave cores free
for other programs.
"""

from __future__ import annotations

import ctypes
import functools
import json
import math
import re
import os
import pathlib
import platform
import subprocess
import sys
import typing

AUTO = "auto"


def _cgroup_limit() -> typing.Optional[int]:
    """CPU limit of a Linux container (cgroup v2 cpu.max or v1 cfs quota), None without a limit."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        text = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            return max(1, math.ceil(int(text[0]) / int(text[1])))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def _usable_cpus() -> tuple[int, str]:
    value = os.environ.get("OPENNVE_CPUS", "").strip()
    if value:
        try:
            number = int(value)
            if number > 0:
                return number, "OPENNVE_CPUS"
        except ValueError:
            pass
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+, respects the affinity
    if process_cpu_count is not None:
        count, source = process_cpu_count() or 1, "process"
    elif hasattr(os, "sched_getaffinity"):
        count, source = len(os.sched_getaffinity(0)) or 1, "process"
    else:
        count, source = os.cpu_count() or 1, "system"
    limit = _cgroup_limit()
    if limit is not None and limit < count:
        return limit, "container limit"
    return count, source


def cpu_count() -> int:
    """Number of CPU cores (logical processors) this process may use: the affinity mask, a
    container CPU limit and the environment variable OPENNVE_CPUS are taken into account."""
    return _usable_cpus()[0]


def _physical_cores() -> typing.Optional[int]:
    try:
        if sys.platform == "darwin":
            output = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=2)
            return int(output.stdout.strip()) or None
        if sys.platform.startswith("linux"):
            cores = set()
            physical = core = None
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "physical id":
                    physical = value.strip()
                elif key == "core id":
                    core = value.strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((physical, core))
                    physical = core = None
            if core is not None:
                cores.add((physical, core))
            return len(cores) or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


# ARM CPU parts (/proc/cpuinfo "CPU part" of implementer 0x41)
_ARM_PARTS = {"0xd03": "Cortex-A53", "0xd04": "Cortex-A35", "0xd05": "Cortex-A55", "0xd07": "Cortex-A57",
              "0xd08": "Cortex-A72", "0xd09": "Cortex-A73", "0xd0a": "Cortex-A75", "0xd0b": "Cortex-A76",
              "0xd0c": "Neoverse-N1", "0xd0d": "Cortex-A77", "0xd40": "Neoverse-V1", "0xd41": "Cortex-A78",
              "0xd44": "Cortex-X1", "0xd46": "Cortex-A510", "0xd47": "Cortex-A710", "0xd48": "Cortex-X2",
              "0xd49": "Neoverse-N2", "0xd4b": "Cortex-A78C", "0xd4d": "Cortex-A715", "0xd4e": "Cortex-X3",
              "0xd4f": "Neoverse-V2", "0xd80": "Cortex-A520", "0xd81": "Cortex-A720", "0xd82": "Cortex-X4",
              "0xd84": "Neoverse-V3", "0xd8e": "Neoverse-N3", "0xd87": "Cortex-A725"}


def _linux_processor() -> str:
    parts: list[str] = []
    implementer = None
    for line in _read("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and value:
            return value
        if key == "CPU implementer":
            implementer = value
        if key == "CPU part" and value.lower() in _ARM_PARTS and implementer == "0x41":
            name = "ARM " + _ARM_PARTS[value.lower()]
            if name not in parts:
                parts.append(name)
    if parts:  # big.LITTLE: all core types
        return " + ".join(parts)
    output = _run(["lscpu"], timeout=2)
    match = re.search(r"^Model name:\s*(.+)$", output, re.M)
    return match.group(1).strip() if match and match.group(1).strip() != "-" else ""


def _processor() -> str:
    name = platform.processor()
    try:
        if sys.platform == "darwin":
            output = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
            name = output.stdout.strip() or name
        elif sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or name
        elif sys.platform.startswith("linux"):
            name = _linux_processor() or name
    except (OSError, subprocess.SubprocessError):
        pass
    return name or platform.machine()


def _run(command: list[str], timeout: float = 5) -> str:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return output.stdout if output.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: typing.Union[str, pathlib.Path]) -> str:
    try:
        return pathlib.Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return ""


def _model() -> typing.Optional[str]:
    """Computer model (Raspberry Pi 5, Mac model, ...), None if unknown."""
    if sys.platform.startswith("linux"):
        return _read("/proc/device-tree/model") or _read("/sys/class/dmi/id/product_name") or None
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "hw.model"]).strip() or None
    if sys.platform == "win32":
        return _powershell("(Get-CimInstance Win32_ComputerSystem).Model").strip() or None
    return None


# ------------------------------------------------------------------ memory


def memory_info() -> dict[str, typing.Optional[int]]:
    """RAM in bytes: total (installed), available (free for new programs) and limit (container
    memory limit on Linux, else None). Values that cannot be read are None."""
    total = available = limit = None
    try:
        if sys.platform.startswith("linux"):
            values = {}
            for line in _read("/proc/meminfo").splitlines():
                key, _, value = line.partition(":")
                parts = value.split()
                if parts:
                    values[key] = int(parts[0]) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable", values.get("MemFree"))
            text = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if text.isdigit() and total and int(text) < total:
                limit = int(text)
        elif sys.platform == "darwin":
            total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0) or None
            output = _run(["vm_stat"])
            page = re.search(r"page size of (\d+) bytes", output)
            if page:
                pages = {key.strip(): int(value.strip(" .")) for key, value in re.findall(r"^(.+?):\s+(\d+)\.?$", output, re.M)}
                free = sum(pages.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
                available = free * int(page.group(1))
        elif sys.platform == "win32":

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total, available = int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (OSError, ValueError, AttributeError):
        pass
    return {"total": total, "available": available, "limit": limit}


# ------------------------------------------------------------------ GPUs and NPUs

_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x1022": "AMD", "0x8086": "Intel", "0x5143": "Qualcomm",
                "0x1e60": "Hailo", "0x1ac1": "Google", "0x1d87": "Rockchip"}

# Linux drivers of NPUs (/sys/class/accel, misc devices): driver -> (vendor, name)
_NPU_DRIVERS = {
    "intel_vpu": ("Intel", "Intel NPU (AI Boost)"),
    "amdxdna": ("AMD", "AMD NPU (Ryzen AI)"),
    "qaic": ("Qualcomm", "Qualcomm Cloud AI"),
    "hailo": ("Hailo", "Hailo AI accelerator"),
    "hailo_pci": ("Hailo", "Hailo AI accelerator"),
    "hailo1x": ("Hailo", "Hailo AI accelerator"),  # Hailo-10H / Hailo-15 (HailoRT 5)
    "hailo1x_pci": ("Hailo", "Hailo AI accelerator"),
    "apex": ("Google", "Coral Edge TPU"),
    "rknpu": ("Rockchip", "Rockchip NPU"),
    "zocl": ("AMD", "AMD DPU (Kria / Zynq UltraScale+, zocl)"),
    "ethosu": ("Arm", "Arm Ethos-U NPU"),
    "ivpu": ("Intel", "Intel NPU (AI Boost)"),
}

# Linux GPU drivers without a PCI vendor (system on chip): driver -> (vendor, name)
_SOC_GPUS = {
    "v3d": ("Broadcom", "VideoCore GPU (V3D)"),
    "vc4": ("Broadcom", "VideoCore GPU"),
    "panfrost": ("Arm", "Mali GPU"),
    "panthor": ("Arm", "Mali GPU"),
    "lima": ("Arm", "Mali GPU"),
    "msm": ("Qualcomm", "Adreno GPU"),
    "etnaviv": ("Vivante", "Vivante GPU"),
    "tegra": ("NVIDIA", "Tegra GPU"),
    "nvgpu": ("NVIDIA", "Jetson GPU"),
}


# display adapters of virtual machines and server management controllers: not listed
_VIRTUAL_GPU_DRIVERS = {"simple-framebuffer", "simpledrm", "vkms", "virtio_gpu", "virtio-pci", "bochs-drm", "bochs",
                        "qxl", "hyperv_drm", "hyperv_fb", "vmwgfx", "vboxvideo", "cirrus", "cirrus-qemu", "ast",
                        "mgag200", "xen-fbfront"}
_VIRTUAL_GPU_NAMES = ("Microsoft Basic", "Remote Display", "Hyper-V", "VMware SVGA", "VirtualBox", "Parallels Display",
                      "Paravirtualized", "Red Hat QXL", "Citrix Indirect")


def _device(kind: str, name: str, vendor: typing.Optional[str] = None, memory: typing.Optional[int] = None,
             driver: typing.Optional[str] = None) -> dict[str, typing.Any]:
    return {"type": kind, "name": name, "vendor": vendor, "memory": memory, "driver": driver}


def _powershell(script: str) -> str:
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=10)


def _uevent(device: pathlib.Path) -> dict[str, str]:
    values = {}
    for line in _read(device / "uevent").splitlines():
        key, _, value = line.partition("=")
        values[key] = value
    return values


def _driver(device: pathlib.Path) -> str:
    driver = _uevent(device).get("DRIVER")
    if not driver and (device / "driver").exists():
        driver = (device / "driver").resolve().name
    return driver or ""


def _nvidia_smi() -> list[dict[str, typing.Any]]:
    devices = []
    output = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in output.splitlines():
        name, _, memory = line.partition(",")
        if name.strip():
            mib = memory.strip()
            devices.append(_device("gpu", name.strip(), "NVIDIA", int(mib) * 1024 * 1024 if mib.isdigit() else None, "nvidia"))
    return devices


def _linux_devices(root: pathlib.Path = pathlib.Path("/")) -> list[dict[str, typing.Any]]:
    devices = _nvidia_smi()
    seen = set()
    for card in sorted((root / "sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:  # connectors (card0-HDMI-A-1)
            continue
        device = (card / "device").resolve()
        if device in seen:
            continue
        seen.add(device)
        driver = _driver(device)
        vendor = _PCI_VENDORS.get(_read(device / "vendor"))
        if driver in _NPU_DRIVERS:
            continue
        if vendor == "NVIDIA" and any(item["vendor"] == "NVIDIA" for item in devices):
            continue  # already listed by nvidia-smi
        if driver in _VIRTUAL_GPU_DRIVERS:
            continue
        if vendor:
            memory = _read(device / "mem_info_vram_total")
            name = f"{vendor} GPU ({driver})" if driver else f"{vendor} GPU"
            devices.append(_device("gpu", name, vendor, int(memory) if memory.isdigit() else None, driver))
        elif driver in _SOC_GPUS:
            vendor, name = _SOC_GPUS[driver]
            devices.append(_device("gpu", name, vendor, None, driver))
        elif driver and driver not in _VIRTUAL_GPU_DRIVERS:
            devices.append(_device("gpu", f"GPU ({driver})", None, None, driver))
    npu_drivers = set()
    for accel in sorted((root / "sys/class/accel").glob("accel*")):
        device = (accel / "device").resolve()
        driver = _driver(device)
        npu_drivers.add(driver)
    for driver, path in (("hailo", "dev/hailo0"), ("apex", "dev/apex_0"), ("rknpu", "sys/class/devfreq/fdab0000.npu")):
        if (root / path).exists():
            npu_drivers.add(driver)
    if "rknpu" not in npu_drivers and "rknpu" in _read(root / "proc/device-tree/compatible"):
        npu_drivers.add("rknpu")
    zocl = root / "sys/bus/platform/drivers/zocl"
    if zocl.is_dir() and any(entry.is_symlink() and entry.name != "module" for entry in zocl.iterdir()):
        npu_drivers.add("zocl")  # XRT driver of AMD Kria / Zynq UltraScale+ boards: a DPU design is loaded
    for driver in sorted(npu_drivers):
        vendor, name = _NPU_DRIVERS.get(driver, (None, f"NPU ({driver})"))
        devices.append(_device("npu", name, vendor, None, driver))
    return devices


def _macos_devices() -> list[dict[str, typing.Any]]:
    devices = []
    try:
        data = json.loads(_run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=10) or "{}")
    except ValueError:
        data = {}
    for item in data.get("SPDisplaysDataType", []):
        name = item.get("sppci_model") or item.get("_name") or "GPU"
        if any(virtual in name for virtual in _VIRTUAL_GPU_NAMES):
            continue
        cores = item.get("sppci_cores")
        vendor = str(item.get("spdisplays_vendor", "")).replace("sppci_vendor_", "").capitalize() or None
        memory = None
        vram = str(item.get("spdisplays_vram", "") or item.get("spdisplays_vram_shared", ""))
        match = re.match(r"(\d+)\s*(GB|MB)", vram)
        if match:
            memory = int(match.group(1)) * (1024**3 if match.group(2) == "GB" else 1024**2)
        devices.append(_device("gpu", f"{name} ({cores} cores)" if cores else name, vendor, memory, "Metal"))
    if platform.machine() == "arm64":  # every Apple silicon chip has a Neural Engine
        devices.append(_device("npu", "Apple Neural Engine", "Apple", None, "Core ML"))
    return devices


def _windows_devices() -> list[dict[str, typing.Any]]:
    devices = []
    script = (
        "$g = @(Get-CimInstance Win32_VideoController | Select-Object Name, AdapterCompatibility, AdapterRAM);"
        "$n = @(Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'ComputeAccelerator' }"
        " | Select-Object Name, Manufacturer);"
        "@{gpus = $g; npus = $n} | ConvertTo-Json -Depth 3 -Compress"
    )
    try:
        data = json.loads(_powershell(script) or "{}")
    except ValueError:
        data = {}

    def items(value):
        return value if isinstance(value, list) else [value] if value else []

    nvidia = _nvidia_smi()
    devices.extend(nvidia)
    for item in items(data.get("gpus")):
        name = item.get("Name") or "GPU"
        if any(virtual in name for virtual in _VIRTUAL_GPU_NAMES):
            continue
        if nvidia and "NVIDIA" in name:
            continue
        ram = item.get("AdapterRAM")
        devices.append(_device("gpu", name, item.get("AdapterCompatibility"), ram if isinstance(ram, int) and ram > 0 else None, "DirectX"))
    for item in items(data.get("npus")):
        devices.append(_device("npu", item.get("Name") or "NPU", item.get("Manufacturer"), None, None))
    return devices


@functools.lru_cache(maxsize=1)
def _accelerators() -> tuple:
    try:
        if sys.platform.startswith("linux"):
            devices = _linux_devices()
        elif sys.platform == "darwin":
            devices = _macos_devices()
        elif sys.platform == "win32":
            devices = _windows_devices()
        else:
            devices = []
    except Exception:  # never fail because of an unusual system
        devices = []
    return tuple(json.dumps(device) for device in devices)


def accelerators() -> list[dict[str, typing.Any]]:
    """GPUs and NPUs found on the machine: dicts with type ("gpu" or "npu"), name, vendor,
    memory (bytes, None if unknown or shared with the CPU) and driver.

    Found through the operating system, without extra packages: nvidia-smi and /sys/class/drm,
    /sys/class/accel (Intel / AMD NPUs), Hailo, Coral and Rockchip devices on Linux; system_profiler
    on macOS (Apple silicon: GPU and Neural Engine); Windows Management Instrumentation on Windows
    (graphics adapters and "compute accelerators" such as Intel AI Boost, AMD Ryzen AI, Qualcomm
    Hexagon). The result is cached. Information only: openNVE runs TensorFlow Lite models on the CPU.
    """
    return [json.loads(device) for device in _accelerators()]


_MACOS_NAMES = {"11": "Big Sur", "12": "Monterey", "13": "Ventura", "14": "Sonoma", "15": "Sequoia", "26": "Tahoe"}


def _windows_registry(name: str) -> typing.Optional[str]:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            return str(winreg.QueryValueEx(key, name)[0]).strip() or None
    except (ImportError, OSError):
        return None


def os_info() -> dict[str, typing.Any]:
    """Operating system: name (for example "Windows 11 Pro", "macOS 15.1 Sequoia",
    "Ubuntu 24.04.1 LTS", "Debian GNU/Linux 12 (bookworm)"), version, build (Windows: build
    number, macOS: build, Linux: None), kernel ("Windows NT 10.0.26100", "Darwin 24.1.0",
    "Linux 6.8.0-49-generic") and family ("Windows",
    "macOS", "Linux")."""
    system = platform.system()
    kernel = platform.release()
    if sys.platform == "win32":
        release, version, _, _ = platform.win32_ver()
        build = version.split(".")[-1] if version else None
        number = "11" if build and build.isdigit() and int(build) >= 22000 else (release or "")
        edition = (platform.win32_edition() or "").replace("Professional", "Pro").replace("Core", "Home")
        product = _windows_registry("ProductName") or ""
        if "Server" in product:
            name = product
        else:
            name = f"Windows {number}" + (f" {edition}" if edition and edition not in ("ServerStandard", "ServerDatacenter") else "")
        display = _windows_registry("DisplayVersion")
        if display:
            name += f" {display}"
        revision = _windows_registry("UBR")
        return {"family": "Windows", "name": name, "version": version or None,
                "build": f"{build}.{revision}" if build and revision else build,
                "kernel": f"Windows NT {version}" if version else kernel}
    if sys.platform == "darwin":
        version = platform.mac_ver()[0] or ""
        major = version.split(".")[0]
        name = f"macOS {version}".strip() + (f" {_MACOS_NAMES[major]}" if major in _MACOS_NAMES else "")
        build = _run(["sw_vers", "-buildVersion"], timeout=2).strip() or None
        return {"family": "macOS", "name": name, "version": version or None, "build": build, "kernel": f"Darwin {kernel}"}
    if sys.platform.startswith("linux"):
        release: dict[str, str] = {}
        try:
            release = platform.freedesktop_os_release()
        except (OSError, AttributeError):
            pass
        name = release.get("PRETTY_NAME") or " ".join(filter(None, [release.get("NAME"), release.get("VERSION")])) or "Linux"
        return {"family": "Linux", "name": name, "version": release.get("VERSION_ID"), "build": None, "kernel": f"Linux {kernel}"}
    return {"family": system or None, "name": f"{system} {kernel}".strip(), "version": platform.version() or None,
            "build": None, "kernel": kernel}


def system_info(devices: bool = True) -> dict[str, typing.Any]:
    """Platform, processor, core counts, RAM and accelerators.

    os: os_info() (name, version, build, kernel, family); platform: system and kernel release
    (for example "Linux 6.8.0"); model: computer model (None if unknown); logical_cpus: logical processors of the machine;
    physical_cores: cores without hyper-threading (None if unknown); usable_cpus: what openNVE
    uses for "auto" (cpu_count()); usable_cpus_source: "process" (affinity), "system", "container
    limit" or "OPENNVE_CPUS"; memory: memory_info(); gpus, npus: accelerators() (empty lists with
    devices=False, which skips the slower device search); litert: the GPU and NPU support of the
    installed LiteRT (opennve.litert_accelerators()); pytorch: version and GPU of PyTorch if
    installed ({"installed": False} otherwise); plugins: the system information of the installed
    plugins by name (for instance hailo: HailoRT version and devices).
    """
    usable, source = _usable_cpus()
    found = accelerators() if devices else []
    return {
        "os": os_info(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "model": _model(),
        "processor": _processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "usable_cpus": usable,
        "usable_cpus_source": source,
        "memory": memory_info(),
        "gpus": [device for device in found if device["type"] == "gpu"],
        "npus": [device for device in found if device["type"] == "npu"],
        "litert": _litert_info(),
        "pytorch": _pytorch_info(),
        "plugins": _plugins_info(),
        "python": platform.python_version(),
    }


def _litert_info() -> dict[str, typing.Any]:
    try:
        from .agents._litert import litert_accelerators

        return litert_accelerators()
    except Exception:
        return {"version": None, "compiled_model": False, "gpu": None, "npu": []}


def _pytorch_info() -> dict[str, typing.Any]:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        from .agents._torch import torch_info

        return torch_info()
    except Exception:
        return {"installed": False}


def _plugins_info() -> dict[str, typing.Any]:
    """System information of the installed plugins (their system_info()), by plugin name."""
    try:
        from .plugins import installed_plugins
    except Exception:
        return {}
    result = {}
    for name, installed in installed_plugins().items():
        if installed.plugin is None or installed.plugin.system_info is None:
            continue
        try:
            result[name] = installed.plugin.system_info()
        except Exception as error:
            result[name] = {"error": str(error)}
    return result


def format_bytes(value: typing.Optional[int]) -> str:
    if value is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
        value /= 1024
    return "-"


def is_auto(value) -> bool:
    return value is None or value == AUTO


# ------------------------------------------------------------------ thread plans

# Measured with TensorFlow Lite (XNNPACK) on small models (U-Net, 80 x 80 tiles): several
# interpreters with one thread each in parallel are faster than one interpreter with several
# threads, and more threads than cores (agent threads x interpreter threads) are much slower.
MAX_AUTO_AGENT_THREADS = 16
MAX_AUTO_INTERPRETER_THREADS = 4


def standalone_interpreter_threads(cpus: typing.Optional[int] = None) -> int:
    """Interpreter threads of an agent with "threads": "auto" outside a workflow (one model,
    lowest latency): all usable cores, at most 4 (more threads hardly help small models)."""
    return max(1, min(cpus or cpu_count(), MAX_AUTO_INTERPRETER_THREADS))
