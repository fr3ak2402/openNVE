"""Tests for PyTorch agents (TorchScript, torch.export, complete models, nn.Module objects).

Skipped if PyTorch is not installed.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

import opennve

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")

if importlib.util.find_spec("torch") is not None:
    import torch

    TILE_WEIGHTS = torch.tensor([[4.0, -2.0], [-3.0, 5.0]])
    TILE_BIAS = torch.tensor([0.1, -0.2])

    class MeanNet(torch.nn.Module):
        """Mean per channel -> dense -> softmax (like tests/data/agent_tiles.tflite), channels first."""

        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(2, 2)
            with torch.no_grad():
                self.dense.weight.copy_(TILE_WEIGHTS.T)
                self.dense.bias.copy_(TILE_BIAS)

        def forward(self, x):
            return torch.softmax(self.dense(x.mean(dim=(2, 3))), dim=1)

    class SegNet(torch.nn.Module):
        """1 x 1 convolution: 2 input channels -> 3 class logits per pixel."""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 3, 1)
            with torch.no_grad():
                self.conv.weight.copy_(torch.tensor([[[[0.0]], [[0.0]]], [[[5.0]], [[0.0]]], [[[0.0]], [[5.0]]]]))
                self.conv.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

        def forward(self, x):
            return self.conv(x)


EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
TILES16 = opennve.FrameConfig(name="tiles16", tiles=opennve.TileConfig(size=16, min_events=0))


def _random_window(seed=0, count=400, width=40, height=24):
    rng = np.random.default_rng(seed)
    events = np.zeros(count, dtype=EVENT_DTYPE)
    events["t"] = np.sort(rng.integers(0, 40_000, count))
    events["x"] = rng.integers(0, width, count)
    events["y"] = rng.integers(0, height, count)
    events["on"] = rng.random(count) < 0.6
    return opennve.EventWindow(camera="left", start_t=0, end_t=40_000, width=width, height=height, polarity_events=events)


def _expected(frame):
    """Output of MeanNet for the padded tiles of a frame, computed with numpy."""
    means = []
    for tile in frame.tiles:
        image = np.zeros((16, 16, 2), np.float32)
        image[: tile.height, : tile.width] = tile.data
        means.append(image.reshape(-1, 2).mean(axis=0))
    logits = np.stack(means) @ TILE_WEIGHTS.numpy() + TILE_BIAS.numpy()
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def test_module_object_channels_first():
    agent = opennve.Agent.from_model(MeanNet(), name="mean", batch_size=4)
    assert agent.backend.__class__.__name__ == "TorchBackend" and agent.layout == "nchw"
    assert agent.device == "cpu"
    frame = _random_window().frame(TILES16)
    result = agent.predict(frame)
    np.testing.assert_allclose(result.outputs[0], _expected(frame), rtol=1e-5, atol=1e-6)
    assert result.outputs[0].dtype == np.float32
    # "layout": "nhwc" set explicitly is kept
    assert opennve.Agent.from_model(MeanNet(), preprocess={"layout": "nhwc"}).layout == "nhwc"
    assert opennve.Agent.from_model(lambda batch: batch).layout == "nhwc"  # other objects: nhwc


def test_model_files(tmp_path):
    example = torch.zeros(4, 2, 16, 16)
    net = MeanNet().eval()
    torch.jit.trace(net, example).save(str(tmp_path / "traced.pt"))
    torch.jit.script(net).save(str(tmp_path / "scripted.pth"))
    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.export.save(torch.export.export(net, (example,), dynamic_shapes=({0: batch},)), str(tmp_path / "dynamic.pt2"))
    torch.export.save(torch.export.export(net, (example,)), str(tmp_path / "fixed.pt2"))  # batch size 4

    frame = _random_window(1).frame(TILES16)  # 6 tiles
    expected = _expected(frame)
    for name in ("traced.pt", "scripted.pth", "dynamic.pt2", "fixed.pt2"):
        config = opennve.AgentConfig(model=name, base_dir=tmp_path)
        assert config.resolved_backend == "pytorch"
        agent = opennve.Agent(config)
        np.testing.assert_allclose(agent.predict(frame).outputs[0], expected, rtol=1e-5, atol=1e-6, err_msg=name)
        assert agent.layout == "nchw"
    assert opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "dynamic.pt2"))).input.shape == (None, 2, 16, 16)
    fixed = opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "fixed.pt2")))
    assert fixed.input.shape == (4, 2, 16, 16)  # 6 inputs: chunks of 4, the last one padded
    assert fixed.predict(frame).outputs[0].shape == (6, 2)


def test_weights_only_and_complete_models(tmp_path):
    torch.save(MeanNet().state_dict(), tmp_path / "weights.pth")
    with pytest.raises(ValueError, match="only holds weights"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "weights.pth"))).load()
    torch.save(MeanNet(), tmp_path / "complete.pt")  # pickled: needs safe_mode false
    with pytest.raises(ValueError, match="safe_mode"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "complete.pt"))).load()
    agent = opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "complete.pt"), safe_mode=False))
    frame = _random_window(2).frame(TILES16)
    np.testing.assert_allclose(agent.predict(frame).outputs[0], _expected(frame), rtol=1e-5, atol=1e-6)


def test_segmentation_channels_first():
    """A channels-first segmentation model gives the same result as the same model channels last."""
    net = SegNet()

    def channels_last(batch):  # the same 1 x 1 convolution, numpy, NHWC
        weights = net.conv.weight.detach().numpy()[:, :, 0, 0]
        return batch @ weights.T + net.conv.bias.detach().numpy()

    prediction = {"task": "segmentation", "activation": "softmax", "target_classes": [1, 2]}
    # event counts: integer logits, no rounding differences between the two at near ties
    frame = _random_window(4).frame({"mode": "count", "tiles": {"size": 16, "min_events": 0}})
    torch_result = opennve.Agent.from_model(net, prediction=prediction).predict(frame)
    numpy_result = opennve.Agent.from_model(channels_last, prediction=prediction).predict(frame)
    assert len(torch_result.predictions) == 6
    for a, b in zip(torch_result.predictions, numpy_result.predictions):
        np.testing.assert_array_equal(a.class_map, b.class_map)
        np.testing.assert_array_equal(a.class_counts, b.class_counts)
        assert a.selected == b.selected
        np.testing.assert_allclose(a.output, b.output, rtol=1e-5, atol=1e-6)
    # result.outputs: probabilities along the class axis (axis 1 for channels first)
    np.testing.assert_allclose(torch_result.outputs[0].transpose(0, 2, 3, 1), numpy_result.outputs[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(torch_result.outputs[0].sum(axis=1), 1, rtol=1e-5)


def test_devices(monkeypatch):
    from opennve.agents import _torch

    monkeypatch.setattr(_torch, "torch_gpu", lambda: None)
    agent = opennve.Agent.from_model(MeanNet(), device="gpu")
    assert agent.device == "cpu" and "no GPU" in agent.device_description()
    assert "NPU" in opennve.Agent.from_model(MeanNet(), device="npu").device_description()
    assert opennve.Agent.from_model(MeanNet(), device="auto").backend.device_note is None
    assert opennve.Agent.from_model(MeanNet(), device=["gpu", "cpu"]).device == "cpu"
    with pytest.raises(RuntimeError, match="no GPU"):
        opennve.Agent.from_model(MeanNet(), device="gpu", fallback=False).load()
    info = opennve.agents.torch_info()
    assert info["installed"] and info["version"] == torch.__version__


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or not (
    torch.cuda.is_available() or torch.backends.mps.is_available()), reason="no PyTorch GPU")
def test_gpu_if_available():
    frame = _random_window(5).frame(TILES16)
    agent = opennve.Agent.from_model(MeanNet(), device="gpu", gpu_precision="fp32")
    assert agent.device == "gpu"
    np.testing.assert_allclose(agent.predict(frame).outputs[0], _expected(frame), rtol=1e-4, atol=1e-5)
