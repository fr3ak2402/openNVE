"""Install and remove plugins (opennve plugin install / remove)."""

from __future__ import annotations

import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import typing
import urllib.request

from . import catalog_entry, installed_plugins
from .catalog import CatalogEntry


def cache_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "opennve" / "plugins"


def download_archive(url: str, target: pathlib.Path) -> pathlib.Path:
    """Downloads a .tar.gz (GitHub archive) and extracts it without its top folder."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = pathlib.PurePosixPath(member.name).parts[1:]
            if not parts or ".." in parts or member.issym() or member.islnk():
                continue
            member.name = str(pathlib.PurePosixPath(*parts))
            archive.extract(member, target, filter="data")
    return target


def _entry(name: str) -> CatalogEntry:
    entry = catalog_entry(name)
    if entry is None:
        from . import catalog

        raise KeyError(f"unknown plugin {name!r} (known: {', '.join(e.name for e in catalog())})")
    return entry


def install(name: str, args: typing.Sequence[str] = (), ref: typing.Optional[str] = None,
            source: typing.Optional[str] = None, python: str = sys.executable) -> int:
    """Installs a catalog plugin into the Python of openNVE. args: passed to the install script
    (method script) or to pip (method pip). source: a local copy of the plugin (script: its
    directory, pip: a path or requirement)."""
    entry = _entry(name)
    if not entry.supported:
        print(f"the plugin {name} is not available on {sys.platform} (only {', '.join(entry.platforms)})",
              file=sys.stderr)
        return 1
    if entry.method == "pip":
        command = [python, "-m", "pip", "install", "--upgrade", source or entry.package or entry.distribution, *args]
        print(" ".join(command))
        code = subprocess.run(command).returncode
    elif entry.method == "script":
        if source:
            folder = pathlib.Path(source).expanduser().resolve()
        else:
            url = entry.archive.format(ref=ref or entry.ref)
            print(f"downloading {url}")
            try:
                folder = download_archive(url, cache_dir() / entry.name)
            except OSError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
        script = folder / entry.script
        if not script.is_file():
            print(f"error: {script} not found", file=sys.stderr)
            return 1
        interpreter = ["bash"] if script.suffix == ".sh" else [python]
        environment = dict(os.environ, PYTHON=python)
        code = subprocess.run([*interpreter, str(script), *args], env=environment, cwd=folder).returncode
    else:
        print(f"error: unknown install method {entry.method!r}", file=sys.stderr)
        return 1
    if code == 0:
        plugin = installed_plugins(reload=True).get(name)
        if plugin is None:
            print(f"warning: the plugin {name} was installed, but openNVE does not find it", file=sys.stderr)
        elif plugin.error:
            print(f"warning: the plugin {name} is installed, but cannot be loaded: {plugin.error}", file=sys.stderr)
        else:
            print(f"plugin {name} {plugin.version or ''} installed")
    return code


def remove(name: str, python: str = sys.executable) -> int:
    """Removes an installed plugin (pip uninstall of its distribution). System parts that a
    script installed (drivers, packages) stay."""
    installed = installed_plugins(reload=True).get(name)
    entry = catalog_entry(name)
    distribution = (installed.distribution if installed else None) or (entry.distribution if entry else None)
    if installed is None and distribution is None:
        print(f"the plugin {name} is not installed", file=sys.stderr)
        return 1
    code = subprocess.run([python, "-m", "pip", "uninstall", "--yes", distribution]).returncode
    installed_plugins(reload=True)
    return code
