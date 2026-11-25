"""Tests for plugins: discovery through entry points, backends of plugins, CLI, install / remove."""

import importlib.metadata
import subprocess
import sys
import types

import numpy as np
import pytest

import opennve
from opennve import plugins
from opennve.agents.backends import Backend, ModelInput
from opennve.plugins import Plugin, manager


class DoubleBackend(Backend):
    """Model of the fake plugin: output = 2 x input."""

    device = "npu"
    device_label = "Fake"

    def __init__(self, config):
        self.input = ModelInput(shape=(None, 16, 16, 2), dtype=np.dtype(np.float32))

    def predict(self, batch):
        return [batch * 2]


def _fake_module(name="fake_opennve_plugin", broken=False):
    module = types.ModuleType(name)
    module.calls = {"register": 0, "commands": 0}

    def register():
        module.calls["register"] += 1
        opennve.agents.register_backend("double", DoubleBackend, [".dbl"], threads="planned", accelerator=True)

    def commands(subparsers):
        module.calls["commands"] += 1
        command = subparsers.add_parser("double", help="fake plugin command")
        command.set_defaults(function=lambda args: print("double says hi") or 0)

    def plugin():
        if broken:
            raise ImportError("libdouble.so not found")
        return Plugin(name="double", description="doubles its input", version="1.2.3", register=register,
                      commands=commands, system_info=lambda: {"devices": 1},
                      system_lines=lambda info: [f"Double                {info['devices']} device"])

    module.plugin = plugin
    return module


@pytest.fixture
def fake_plugin(monkeypatch):
    module = _fake_module()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entry_point = importlib.metadata.EntryPoint("double", f"{module.__name__}:plugin", plugins.ENTRY_POINT_GROUP)
    monkeypatch.setattr(plugins, "_entry_points", lambda: [entry_point])
    plugins._reset()
    yield module
    plugins._reset()
    from opennve.agents import registry

    for table in (registry._BACKENDS, registry._BACKEND_OPTIONS):
        table.pop("double", None)
    registry._BACKEND_SUFFIXES.pop(".dbl", None)


def test_discovery_and_backend(fake_plugin, tmp_path):
    installed = plugins.installed_plugins()
    assert list(installed) == ["double"] and installed["double"].version == "1.2.3"
    assert installed["double"].description == "doubles its input" and installed["double"].error is None
    model = tmp_path / "model.dbl"
    model.write_bytes(b"x")
    config = opennve.AgentConfig(model=str(model))
    assert config.resolved_backend == "double"  # the unknown suffix loads the plugins
    assert fake_plugin.calls["register"] == 1
    plugins.load_plugins()
    assert fake_plugin.calls["register"] == 1  # registered once
    assert "double" in opennve.agents.list_backends()
    agent = opennve.Agent(config)
    assert agent.device_description().startswith("NPU Fake")
    batch = np.ones((3, 16, 16, 2), np.float32)
    np.testing.assert_array_equal(agent.predict_batch(batch)[0], batch * 2)
    from opennve.workflows.steps.processing import AgentStep

    assert AgentStep._accelerated(config)
    # backend by name
    assert opennve.AgentConfig(model=str(model), backend="double").resolved_backend == "double"


def test_system_info_and_cli(fake_plugin, capsys):
    from opennve.cli import main

    assert opennve.system_info()["plugins"] == {"double": {"devices": 1}}
    assert main(["plugin", "list"]) == 0
    out = capsys.readouterr().out
    assert "double       1.2.3" in out and "hailo" in out  # installed, and the catalog
    assert main(["double"]) == 0  # command of the plugin
    assert "double says hi" in capsys.readouterr().out
    assert main(["system"]) == 0
    assert "Double                1 device" in capsys.readouterr().out
    assert main(["version", "--json"]) == 0
    assert '"double": "1.2.3"' in capsys.readouterr().out


def test_broken_plugin(monkeypatch, capsys):
    module = _fake_module("broken_opennve_plugin", broken=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entry_point = importlib.metadata.EntryPoint("double", f"{module.__name__}:plugin", plugins.ENTRY_POINT_GROUP)
    monkeypatch.setattr(plugins, "_entry_points", lambda: [entry_point])
    plugins._reset()
    try:
        installed = plugins.load_plugins()
        assert installed["double"].plugin is None and "libdouble.so" in installed["double"].error
        from opennve.cli import main

        assert main(["plugin", "list"]) == 0  # openNVE keeps working
        assert "cannot be loaded: libdouble.so not found" in capsys.readouterr().out
    finally:
        plugins._reset()


def test_hint_for_plugin_models(monkeypatch, tmp_path):
    monkeypatch.setattr(plugins, "_entry_points", lambda: [])
    plugins._reset()
    (tmp_path / "unet.hef").write_bytes(b"x")
    with pytest.raises(ValueError, match="opennve plugin install hailo"):
        opennve.AgentConfig(model=str(tmp_path / "unet.hef")).resolved_backend
    with pytest.raises(ValueError, match="opennve plugin install hailo"):
        opennve.AgentConfig(model=str(tmp_path / "unet.hef"), backend="hailo")
    assert plugins.plugin_for_suffix(".HEF").name == "hailo"
    (tmp_path / "unet.xmodel").write_bytes(b"x")
    with pytest.raises(ValueError, match="opennve plugin install vitis"):
        opennve.AgentConfig(model=str(tmp_path / "unet.xmodel")).resolved_backend
    assert plugins.plugin_for_backend("vitis").name == "vitis"
    plugins._reset()


def test_catalog_and_info(monkeypatch, capsys):
    from opennve.cli import main

    monkeypatch.setattr(plugins, "_entry_points", lambda: [])
    plugins._reset()
    entry = plugins.catalog_entry("hailo")
    assert entry.method == "script" and entry.backends == ("hailo",) and entry.platforms == ("linux",)
    assert main(["plugin", "info", "hailo"]) == 0
    out = capsys.readouterr().out
    assert "not installed" in out and "github.com/fr3ak2402/openNVE-hailo" in out
    assert main(["plugin", "info", "nothing"]) == 1
    entry = plugins.catalog_entry("vitis")
    assert entry.method == "script" and entry.suffixes == (".xmodel",) and entry.commands == ("vitis",)
    assert "codeload.github.com/fr3ak2402/openNVE-vitis" in entry.archive
    plugins._reset()


@pytest.mark.skipif(sys.platform == "win32", reason="runs an install.sh with bash (plugins with install scripts are for Linux)")
def test_install_script_plugin(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(plugins, "_entry_points", lambda: [])
    plugins._reset()
    (tmp_path / "install.sh").write_text('echo "$PYTHON $*" > "$(pwd)/called"\n')
    assert manager.install("hailo", ["--yes", "--no-gen3"], source=str(tmp_path)) == 0
    assert (tmp_path / "called").read_text().split() == [sys.executable, "--yes", "--no-gen3"]
    assert manager.install("hailo", source=str(tmp_path / "missing")) == 1
    monkeypatch.setattr(sys, "platform", "win32")
    assert manager.install("hailo", source=str(tmp_path)) == 1  # Linux only
    with pytest.raises(KeyError):
        manager.install("nothing")
    plugins._reset()


def test_install_pip_plugin_and_remove(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(plugins, "_entry_points", lambda: [])
    entry = plugins.CatalogEntry(name="pipplugin", description="d", distribution="opennve-pipplugin",
                                 repository="https://example.org", method="pip", package="opennve-pipplugin>=1")
    catalog_module = sys.modules["opennve.plugins.catalog"]
    monkeypatch.setattr(catalog_module, "CATALOG", catalog_module.CATALOG + [entry])
    plugins._reset()
    assert manager.install("pipplugin") == 0
    assert calls[-1] == [sys.executable, "-m", "pip", "install", "--upgrade", "opennve-pipplugin>=1"]
    assert manager.remove("pipplugin") == 0
    assert calls[-1] == [sys.executable, "-m", "pip", "uninstall", "--yes", "opennve-pipplugin"]
    plugins._reset()


def test_system_hint_for_kria(monkeypatch, capsys):
    """opennve system on a Kria board (DPU design loaded, zocl) without the plugin vitis."""
    from opennve import system
    from opennve.cli import main

    monkeypatch.setattr(plugins, "_entry_points", lambda: [])
    plugins._reset()
    info = system.system_info()
    info["npus"] = [{"type": "npu", "name": "AMD DPU (Kria / Zynq UltraScale+, zocl)", "vendor": "AMD",
                     "memory": None, "driver": "zocl"}]
    monkeypatch.setattr(system, "system_info", lambda devices=True: info)
    assert main(["system"]) == 0
    assert "plugin not installed: opennve plugin install vitis" in capsys.readouterr().out
    plugins._reset()
