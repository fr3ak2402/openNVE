"""Plugins: separate packages that add model runtimes, commands and system information to openNVE.

A plugin is a Python package with an entry point in the group "opennve.plugins":

    # pyproject.toml of the plugin
    [project.entry-points."opennve.plugins"]
    hailo = "opennve_hailo.opennve_plugin:plugin"

The entry point is a Plugin, or a function returning one (so the plugin imports openNVE only when
openNVE loads it):

    def plugin():
        from opennve.plugins import Plugin
        return Plugin(name="hailo", description="...", register=register, commands=commands)

register() registers backends / tasks (opennve.agents.register_backend, register_task);
commands(subparsers) adds `opennve <command>`; system_info() adds a section to
opennve.system_info() and system_lines(info) its lines to `opennve system`.

    opennve plugin list                # installed and installable plugins
    opennve plugin install hailo       # install (from the catalog)
    opennve plugin remove hailo
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import threading
import typing

from .catalog import CATALOG, CatalogEntry, catalog, catalog_entry

ENTRY_POINT_GROUP = "opennve.plugins"

__all__ = [
    "CATALOG",
    "CatalogEntry",
    "InstalledPlugin",
    "Plugin",
    "catalog",
    "catalog_entry",
    "installed_plugins",
    "load_plugins",
    "plugin_for_backend",
    "plugin_for_suffix",
]


@dataclasses.dataclass
class Plugin:
    """What a plugin adds to openNVE (every part optional)."""

    name: str
    description: str = ""
    version: typing.Optional[str] = None  # default: version of the installed distribution
    register: typing.Optional[typing.Callable[[], None]] = None
    commands: typing.Optional[typing.Callable[[argparse._SubParsersAction], None]] = None
    system_info: typing.Optional[typing.Callable[[], dict]] = None
    system_lines: typing.Optional[typing.Callable[[dict], list[str]]] = None


@dataclasses.dataclass
class InstalledPlugin:
    name: str
    version: typing.Optional[str]
    distribution: typing.Optional[str]
    plugin: typing.Optional[Plugin] = None
    error: typing.Optional[str] = None  # the plugin is installed but could not be loaded

    @property
    def description(self) -> str:
        if self.plugin is not None and self.plugin.description:
            return self.plugin.description
        entry = catalog_entry(self.name)
        return entry.description if entry else ""


_LOCK = threading.RLock()
_INSTALLED: typing.Optional[dict[str, InstalledPlugin]] = None
_REGISTERED: set[str] = set()


def _entry_points() -> list:
    return list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))


def _load(entry_point) -> InstalledPlugin:
    distribution = getattr(getattr(entry_point, "dist", None), "name", None)
    version = getattr(getattr(entry_point, "dist", None), "version", None)
    try:
        value = entry_point.load()
        plugin = value if isinstance(value, Plugin) else value()
        if not isinstance(plugin, Plugin):
            raise TypeError(f"{entry_point.value} returned {type(plugin).__name__}, not opennve.plugins.Plugin")
    except Exception as error:  # a broken plugin must not break openNVE
        message = str(error).splitlines()[0] if str(error) else type(error).__name__
        return InstalledPlugin(entry_point.name, version, distribution, error=message)
    return InstalledPlugin(plugin.name or entry_point.name, plugin.version or version, distribution, plugin)


def installed_plugins(reload: bool = False) -> dict[str, InstalledPlugin]:
    """The installed plugins by name (loaded once; reload=True finds newly installed ones)."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED is None or reload:
            importlib.invalidate_caches()
            _INSTALLED = {}
            for entry_point in _entry_points():
                installed = _load(entry_point)
                _INSTALLED.setdefault(installed.name, installed)
        return dict(_INSTALLED)


def load_plugins(reload: bool = False) -> dict[str, InstalledPlugin]:
    """Loads the installed plugins and lets them register their backends and tasks (once each)."""
    with _LOCK:
        plugins = installed_plugins(reload)
        for name, installed in plugins.items():
            if installed.plugin is None or name in _REGISTERED:
                continue
            _REGISTERED.add(name)
            if installed.plugin.register is not None:
                try:
                    installed.plugin.register()
                except Exception as error:
                    installed.error = f"register failed: {error}"
        return plugins


def plugin_for_suffix(suffix: str) -> typing.Optional[CatalogEntry]:
    """The catalog plugin that runs models with this file suffix (for hints), for instance .hef."""
    return next((entry for entry in catalog() if suffix.lower() in entry.suffixes), None)


def plugin_for_backend(backend: str) -> typing.Optional[CatalogEntry]:
    return next((entry for entry in catalog() if backend in entry.backends), None)


def _reset() -> None:
    """Forgets loaded plugins (tests)."""
    global _INSTALLED
    with _LOCK:
        _INSTALLED = None
        _REGISTERED.clear()
