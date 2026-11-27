"""Catalog of the plugins that `opennve plugin install` knows.

Install methods:
    pip      pip install <package> into the Python of openNVE
    script   download the archive of the plugin's repository and run its install script with the
             Python of openNVE (PYTHON=...), for plugins that also set up the system (drivers)
"""

from __future__ import annotations

import dataclasses
import sys
import typing


@dataclasses.dataclass(frozen=True)
class CatalogEntry:
    name: str
    description: str
    distribution: str  # pip name of the installed package, for remove and version
    repository: str
    method: str  # "pip" or "script"
    platforms: tuple[str, ...] = ("linux", "win32", "darwin")  # sys.platform prefixes
    package: str = ""  # pip: requirement to install (default: distribution)
    archive: str = ""  # script: URL of a .tar.gz, {ref} is replaced by the branch or tag
    script: str = ""  # script: path of the install script in the archive
    ref: str = "main"
    backends: tuple[str, ...] = ()  # backends the plugin registers
    suffixes: tuple[str, ...] = ()  # model file suffixes it runs
    commands: tuple[str, ...] = ()  # opennve commands it adds
    notes: str = ""

    @property
    def supported(self) -> bool:
        return any(sys.platform.startswith(platform) for platform in self.platforms)


CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        name="hailo",
        description="Hailo accelerators (Hailo-8, 8L, 10H, 15; Raspberry Pi AI Kit, AI HAT+, AI HAT+ 2): "
        "agents with compiled .hef models",
        distribution="opennve-hailo",
        repository="https://github.com/fr3ak2402/openNVE-hailo",
        method="script",
        platforms=("linux",),
        archive="https://codeload.github.com/fr3ak2402/openNVE-hailo/tar.gz/{ref}",
        script="install.sh",
        backends=("hailo",),
        suffixes=(".hef",),
        commands=("hailo",),
        notes="On a Raspberry Pi it also enables PCIe Gen 3 and installs driver, firmware and HailoRT "
        "(apt, asks first); a reboot may be needed afterwards. Options of install.sh after --: "
        "--yes, --reboot, --no-gen3, --package=hailo-all|hailo-h10-all, --no-system, --build",
    ),
]


def catalog() -> list[CatalogEntry]:
    return list(CATALOG)


def catalog_entry(name: str) -> typing.Optional[CatalogEntry]:
    return next((entry for entry in CATALOG if entry.name == name), None)
