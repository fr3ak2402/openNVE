"""Tests for Hailo agents (.hef) with a fake opennve_hailo (no Hailo hardware needed).

The real package (github.com/fr3ak2402/openNVE-hailo) is tested there against libhailort 4.x and 5.x.
"""

import sys
import threading
import types
import typing

import numpy as np
import pytest

import opennve
from opennve.agents import _hailo

WEIGHTS = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], np.float32)  # 1 x 1 convolution, 2 -> 3 classes
BIAS = np.array([1.0, 0.0, 0.0], np.float32)


def _fake_hailo(shape=(16, 16, 2), chip="Hailo-8L", version="4.23.0"):
    """Stand-in for opennve_hailo: the .hef "model" is a 1 x 1 convolution."""
    module = types.ModuleType("opennve_hailo")
    module.__version__ = "0.1.0"
    module.calls = {"runs": 0, "batches": [], "threads": set(), "closed": 0, "models": []}

    class Stream(typing.NamedTuple):
        name: str
        shape: tuple
        size: int

    class Model:
        def __init__(self, path, batch_size=8, timeout_ms=10_000):
            if str(path).endswith("other_chip.hef"):
                raise RuntimeError("model.hef does not fit the Hailo device: a .hef is compiled for one chip")
            self.path, self.batch_size = str(path), batch_size
            self.input = Stream("unet/input_layer1", shape, int(np.prod(shape)))
            out = shape[:2] + (3,)
            self.outputs = [Stream("unet/conv_out", out, int(np.prod(out)))]
            module.calls["models"].append(self)

        def run(self, batch):
            assert batch.dtype == np.float32 and batch.flags["C_CONTIGUOUS"] and batch.shape[1:] == shape
            module.calls["runs"] += 1
            module.calls["batches"].append(len(batch))
            module.calls["threads"].add(threading.get_ident())
            return [batch @ WEIGHTS + BIAS]

        def close(self):
            module.calls["closed"] += 1

    def info(identify=True):
        return {"version": "0.1.0", "hailort": version, "compiled_with": version,
                "devices": [{"id": "0001:01:00.0", "architecture": chip if identify else None}]}

    module.Model, module.info = Model, info
    return module


@pytest.fixture
def hailo(monkeypatch):
    module = _fake_hailo()
    monkeypatch.setitem(sys.modules, "opennve_hailo", module)
    return module


EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


def _frame(seed=0, width=40, height=24):
    rng = np.random.default_rng(seed)
    events = np.zeros(500, dtype=EVENT_DTYPE)
    events["t"] = np.sort(rng.integers(0, 40_000, 500))
    events["x"] = rng.integers(0, width, 500)
    events["y"] = rng.integers(0, height, 500)
    events["on"] = rng.random(500) < 0.6
    window = opennve.EventWindow(camera="left", start_t=0, end_t=40_000, width=width, height=height, polarity_events=events)
    return window.frame({"mode": "count", "tiles": {"size": 16, "min_events": 0}})


PREDICTION = {"task": "segmentation", "activation": "softmax", "target_classes": [1, 2]}


def test_hef_agent_matches_the_same_model_on_the_cpu(hailo, tmp_path):
    (tmp_path / "unet.hef").write_bytes(b"fake")
    config = opennve.AgentConfig(model="unet.hef", base_dir=tmp_path, batch_size=4, prediction=PREDICTION)
    assert config.resolved_backend == "hailo"
    agent = opennve.Agent(config)
    assert agent.layout == "nhwc" and agent.device == "npu"
    assert agent.input.shape == (None, 16, 16, 2)
    assert agent.device_description() == "NPU Hailo (batch 4)"
    assert hailo.calls["models"][0].batch_size == 4
    frame = _frame()
    result = agent.predict(frame)  # 6 tiles, 2 of them smaller (padded)
    reference = opennve.Agent.from_model(lambda batch: batch @ WEIGHTS + BIAS, prediction=PREDICTION).predict(frame)
    assert len(result.predictions) == 6
    for a, b in zip(result.predictions, reference.predictions):
        np.testing.assert_array_equal(a.class_map, b.class_map)
        assert a.selected == b.selected
    np.testing.assert_allclose(result.outputs[0], reference.outputs[0], rtol=1e-6)
    agent.close()
    assert hailo.calls["closed"] == 1


def test_classification_head(monkeypatch, tmp_path):
    """1 x 1 x classes outputs are given as (inputs, classes)."""
    monkeypatch.setitem(sys.modules, "opennve_hailo", _fake_hailo(shape=(1, 1, 2)))
    (tmp_path / "c.hef").write_bytes(b"fake")
    backend = _hailo.HailoBackend(tmp_path / "c.hef")
    assert backend.predict(np.ones((5, 1, 1, 2), np.float32))[0].shape == (5, 3)


def test_devices_and_errors(hailo, tmp_path, monkeypatch):
    (tmp_path / "a.hef").write_bytes(b"fake")
    for device in ("npu", "auto", None):
        assert opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"), device=device)).device == "npu"
    for device in ("cpu", "gpu", ["npu", "cpu"]):
        with pytest.raises(ValueError, match="only on a Hailo"):
            opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"), device=device)).load()
    (tmp_path / "other_chip.hef").write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="compiled for one chip"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "other_chip.hef"))).load()
    with pytest.raises(FileNotFoundError):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "missing.hef"))).load()
    monkeypatch.setitem(sys.modules, "opennve_hailo", None)  # not installed
    with pytest.raises(ImportError, match="openNVE-hailo"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"))).load()


@pytest.mark.parametrize("chip, version", [("Hailo-8L", "4.23.0"), ("Hailo-8", "4.23.0"),
                                           ("Hailo-10H", "5.5.0"), ("Hailo-15H", "5.5.0")])
def test_hailo_info(monkeypatch, chip, version):
    monkeypatch.setitem(sys.modules, "opennve_hailo", _fake_hailo(chip=chip, version=version))
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *args: object())
    info = opennve.agents.hailo_info()
    assert info == {"installed": True, "version": version, "opennve_hailo": "0.1.0",
                    "devices": [{"id": "0001:01:00.0", "architecture": chip}]}
    assert opennve.agents.hailo_info(identify=False)["devices"][0]["architecture"] is None


def test_workflow(hailo, tmp_path):
    """Hailo agents get 2 workflow threads like other accelerators."""
    from test_raw import _header, _random_events, encode_evt3

    raw = tmp_path / "rec.raw"
    raw.write_bytes(_header("EVT3;height=32;width=64") + encode_evt3(_random_events(11, count=20_000, span=1_000_000, width=64, height=32)))
    hef = tmp_path / "unet.hef"
    hef.write_bytes(b"fake")
    tiles = {"name": "t16", "mode": "count", "tiles": {"size": 16, "min_events": 1}}
    agent = {"name": "unet", "model": str(hef), "frame": "t16", "prediction": PREDICTION}
    steps = [{"type": "raw", "path": str(raw), "block_ms": 250}, {"type": "windows", "count": 1},
             {"type": "frames", "frame": "t16"}, {"type": "agent", "agent": "unet"}, {"type": "collect"}]
    workflow = opennve.Workflow.from_steps("hailo", steps, frames=[tiles], agents=[agent], cpus=8)
    plan = {step["name"]: step for step in workflow.thread_plan()["steps"]}["agent"]
    assert plan["threads"] == 2
    run = workflow.run()
    assert len(run.results) == 4 and hailo.calls["runs"] > 0
