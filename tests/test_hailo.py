"""Tests for Hailo agents (.hef) with a fake HailoRT Python API (no Hailo hardware needed)."""

import sys
import threading
import types

import numpy as np
import pytest

import opennve
from opennve.agents import _hailo

WEIGHTS = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], np.float32)  # 1 x 1 convolution, 2 -> 3 classes
BIAS = np.array([1.0, 0.0, 0.0], np.float32)


def _fake_hailo(shape=(16, 16, 2), multiple_inputs=False, architecture="HAILO8L", version="4.23.0"):
    """A minimal stand-in for hailo_platform: the .hef "model" is a 1 x 1 convolution."""
    module = types.ModuleType("hailo_platform")
    module.__version__ = version
    module.calls = {"vdevices": 0, "runs": 0, "batches": [], "threads": set(), "shutdown": 0}

    class FormatType:
        FLOAT32 = "float32"
        UINT8 = "uint8"

    class HailoSchedulingAlgorithm:
        ROUND_ROBIN = "round_robin"

    class Stream:
        def __init__(self, name, stream_shape):
            self.name, self.shape, self.format_type = name, list(stream_shape), "uint8"

        def set_format_type(self, value):
            self.format_type = value

    class Buffer:
        def __init__(self):
            self.array = None

        def set_buffer(self, array):
            assert array.flags["C_CONTIGUOUS"]
            self.array = array

    class Bindings:
        def __init__(self, outputs):
            self._input, self._outputs = Buffer(), outputs

        def input(self, name=""):
            return self._input

    class Configured:
        def __init__(self, model):
            self.model = model

        def create_bindings(self, input_buffers=None, output_buffers=None):
            return Bindings(output_buffers)

        def run(self, bindings, timeout):
            module.calls["runs"] += 1
            module.calls["batches"].append(len(bindings))
            module.calls["threads"].add(threading.get_ident())
            assert self.model.inputs[0].format_type == FormatType.FLOAT32
            for binding in bindings:
                image = binding._input.array
                assert image.shape == tuple(self.model.inputs[0].shape) and image.dtype == np.float32
                binding._outputs["unet/conv_out"][...] = image @ WEIGHTS + BIAS

        def shutdown(self):
            module.calls["shutdown"] += 1

    class InferModel:
        def __init__(self, path):
            self.path = path
            self.inputs = [Stream("unet/input_layer1", shape)] + ([Stream("second", shape)] if multiple_inputs else [])
            self.outputs = [Stream("unet/conv_out", shape[:2] + (3,))]
            self.batch_size = None

        @property
        def input_names(self):
            return [stream.name for stream in self.inputs]

        @property
        def output_names(self):
            return [stream.name for stream in self.outputs]

        def input(self, name=""):
            return self.inputs[0]

        def set_batch_size(self, value):
            self.batch_size = value

        def configure(self):
            return Configured(self)

    class Params:
        scheduling_algorithm = None

    class VDevice:
        def __init__(self, params=None):
            module.calls["vdevices"] += 1
            self.params = params

        @staticmethod
        def create_params():
            return Params()

        def create_infer_model(self, path, name=""):
            if str(path).endswith("other_chip.hef"):
                raise RuntimeError("libhailort failed with error: 93 (HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE)")
            return InferModel(path)

        def get_physical_devices_ids(self):
            return ["0001:01:00.0"]

        def release(self):
            pass

    class Identity:
        device_architecture = f"DeviceArchitectureTypes.{architecture}"

    class Control:
        def identify(self):
            return Identity()

    class Device:
        def __init__(self, device_id=None):
            self.device_id, self.control = device_id, Control()

        @staticmethod
        def scan():
            return ["0001:01:00.0"]

        def release(self):
            pass

    module.FormatType, module.HailoSchedulingAlgorithm = FormatType, HailoSchedulingAlgorithm
    module.VDevice, module.Device = VDevice, Device
    return module


@pytest.fixture
def hailo(monkeypatch):
    module = _fake_hailo()
    monkeypatch.setitem(sys.modules, "hailo_platform", module)
    monkeypatch.setattr(_hailo, "_VDEVICE", None)
    yield module
    _hailo._VDEVICE = None


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
    frame = _frame()
    result = agent.predict(frame)  # 6 tiles, 2 of them smaller (padded)
    reference = opennve.Agent.from_model(lambda batch: batch @ WEIGHTS + BIAS, prediction=PREDICTION).predict(frame)
    assert len(result.predictions) == 6
    for a, b in zip(result.predictions, reference.predictions):
        np.testing.assert_array_equal(a.class_map, b.class_map)
        assert a.selected == b.selected
    np.testing.assert_allclose(result.outputs[0], reference.outputs[0], rtol=1e-6)
    model = agent.backend._model
    assert model.batch_size == 4 and model.outputs[0].format_type == "float32"
    assert sum(hailo.calls["batches"]) == 6 and max(hailo.calls["batches"]) <= 4
    agent.close()
    assert hailo.calls["shutdown"] == 1


def test_one_device_for_all_agents(hailo, tmp_path):
    (tmp_path / "a.hef").write_bytes(b"fake")
    agents = [opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"), name=f"a{i}")) for i in range(3)]
    for agent in agents:
        agent.load()
    assert hailo.calls["vdevices"] == 1
    assert _hailo._VDEVICE.params.scheduling_algorithm == "round_robin"
    assert agents[0].backend.batch_size == _hailo.DEFAULT_BATCH_SIZE


def test_devices_and_errors(hailo, tmp_path, monkeypatch):
    (tmp_path / "a.hef").write_bytes(b"fake")
    for device in ("npu", "auto", None):
        assert opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"), device=device)).device == "npu"
    for device in ("cpu", "gpu", ["npu", "cpu"]):
        with pytest.raises(ValueError, match="only on a Hailo"):
            opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"), device=device)).load()
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_hailo(multiple_inputs=True))
    monkeypatch.setattr(_hailo, "_VDEVICE", None)
    with pytest.raises(ValueError, match="2 inputs"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"))).load()
    module = _fake_hailo()

    class NoDevice:
        def __init__(self, params=None):
            raise RuntimeError("libhailort failed with error: 74 (HAILO_OUT_OF_PHYSICAL_DEVICES)")

    module.VDevice.__init__ = NoDevice.__init__
    monkeypatch.setitem(sys.modules, "hailo_platform", module)
    monkeypatch.setattr(_hailo, "_VDEVICE", None)
    with pytest.raises(RuntimeError, match="no free Hailo device"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"))).load()
    monkeypatch.setitem(sys.modules, "hailo_platform", None)  # not installed
    with pytest.raises(ImportError, match="build_hailort_python.sh"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.hef"))).load()


@pytest.mark.parametrize("architecture, version, chip", [
    ("HAILO8L", "4.23.0", "Hailo-8L"), ("HAILO8", "4.23.0", "Hailo-8"),
    ("HAILO10H", "5.5.0", "Hailo-10H"), ("HAILO15H", "5.5.0", "Hailo-15H")])
def test_hailo_info(monkeypatch, architecture, version, chip):
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_hailo(architecture=architecture, version=version))
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *args: object())
    info = opennve.agents.hailo_info()
    assert info == {"installed": True, "version": version, "devices": [{"id": "0001:01:00.0", "architecture": chip}]}
    assert opennve.agents.hailo_info(identify=False)["devices"][0]["architecture"] is None


def test_hef_for_another_chip(hailo, tmp_path):
    (tmp_path / "other_chip.hef").write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="compiled for one chip"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "other_chip.hef"))).load()


def test_workflow(hailo, tmp_path):
    """Hailo agents get 2 workflow threads like other accelerators; they share one device."""
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
    assert len(run.results) == 4 and hailo.calls["vdevices"] == 1 and hailo.calls["runs"] > 0
