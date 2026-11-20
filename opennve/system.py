"""Hardware information for the automatic thread settings.

    opennve.cpu_count()      CPU cores this process may use (affinity, container limits, OPENNVE_CPUS)
    opennve.system_info()    platform, processor and core counts (also: opennve system)

Workflows with "threads": "auto" (the default of the frames and agent steps) and agents with
"threads": "auto" (or null) size their threads from cpu_count(). The environment variable
OPENNVE_CPUS (a positive integer) overrides the detected number, for instance to leave cores free
for other programs.
"""

from __future__ import annotations

import math
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


def _processor() -> str:
    name = platform.processor()
    try:
        if sys.platform == "darwin":
            output = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
            name = output.stdout.strip() or name
        elif sys.platform.startswith("linux"):
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Model", "Hardware"):
                    name = value.strip()
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    return name or platform.machine()


def system_info() -> dict[str, typing.Any]:
    """Platform, processor and core counts.

    logical_cpus: logical processors of the machine; physical_cores: cores without
    hyper-threading (None if unknown); usable_cpus: what openNVE uses for "auto" (cpu_count());
    usable_cpus_source: "process" (affinity), "system", "container limit" or "OPENNVE_CPUS".
    """
    usable, source = _usable_cpus()
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": _processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "usable_cpus": usable,
        "usable_cpus_source": source,
        "python": platform.python_version(),
    }


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
