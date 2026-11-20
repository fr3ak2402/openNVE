"""Version and build information.

    opennve.__version__        "1.0.0b0"
    opennve.BUILD_NUMBER       "324.2026"  (<day of year>.<year> of the build, UTC)
    opennve.version_info()     everything as a dictionary (also: opennve version on the command line)

The build number is written into the wheel by tools/bundle_wheel.py when the wheel is built.
Without a built wheel (source checkout, editable install) it is the date of the last git commit,
or today if there is no git repository, and BUILD_SOURCE says which.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import pathlib
import platform
import subprocess
import sys
import typing

__version__ = "1.0.0b0"


def format_build_number(date: typing.Union[datetime.date, datetime.datetime]) -> str:
    """Build number of a date: <day of year>.<year>, for instance 20 November 2026 -> "324.2026"."""
    return f"{date.timetuple().tm_yday}.{date.year}"


def parse_build_number(build_number: str) -> datetime.date:
    """Date of a build number ("324.2026" -> 2026-11-20)."""
    day, year = (int(part) for part in build_number.split("."))
    return datetime.date(year, 1, 1) + datetime.timedelta(days=day - 1)


def _git_commit_date() -> typing.Optional[datetime.datetime]:
    root = pathlib.Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return None
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%cI"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return datetime.datetime.fromisoformat(output).astimezone(datetime.timezone.utc)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


try:  # written by tools/bundle_wheel.py
    from ._bundle import BUILD_DATE as _BUILD_DATE
    from ._bundle import NEUROMORPHIC_DRIVERS_VERSION

    BUILD_DATE = datetime.datetime.fromisoformat(_BUILD_DATE)
    BUILD_SOURCE = "wheel"
except ImportError:
    try:
        from ._bundle import NEUROMORPHIC_DRIVERS_VERSION  # older bundles without build date
    except ImportError:
        try:
            NEUROMORPHIC_DRIVERS_VERSION = importlib.metadata.version("neuromorphic_drivers")
        except importlib.metadata.PackageNotFoundError:
            NEUROMORPHIC_DRIVERS_VERSION = None
    _commit = _git_commit_date()
    if _commit is not None:
        BUILD_DATE, BUILD_SOURCE = _commit, "git"
    else:
        BUILD_DATE, BUILD_SOURCE = datetime.datetime.now(datetime.timezone.utc), "today"

BUILD_NUMBER = format_build_number(BUILD_DATE)


def build_number() -> str:
    """The build number: <day of year>.<year>, for instance "324.2026"."""
    return BUILD_NUMBER


def version_info() -> dict[str, typing.Any]:
    """Version, build number, build date and the versions of the main dependencies."""

    def installed(name: str) -> typing.Optional[str]:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "version": __version__,
        "build": BUILD_NUMBER,
        "build_date": BUILD_DATE.isoformat(timespec="seconds"),
        "build_source": BUILD_SOURCE,
        "neuromorphic_drivers": NEUROMORPHIC_DRIVERS_VERSION,
        "python": platform.python_version(),
        "platform": f"{sys.platform} {platform.machine()}",
        "numpy": installed("numpy"),
        "opencv": installed("opencv-python") or installed("opencv-python-headless"),
        "litert": installed("ai-edge-litert"),
        "tensorflow": installed("tf-nightly") or installed("tensorflow") or installed("tensorflow-cpu"),
    }


def version_string() -> str:
    """For instance "openNVE 1.0.0b0 (build 324.2026)"."""
    return f"openNVE {__version__} (build {BUILD_NUMBER})"
