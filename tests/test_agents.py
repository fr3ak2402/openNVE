"""Tests for agents (TensorFlow and TensorFlow Lite models on frames and tiles).

The models in tests/data are created by tools/make_test_models.py and have fixed weights,
so the expected outputs are computed with numpy. Tests that need LiteRT or TensorFlow are
skipped if the package is not installed.
"""

import json
import pathlib

import numpy as np
import pytest

import opennve
from opennve.agents.backends import _tflite_interpreter_class

DATA = pathlib.Path(__file__).resolve().parent / "data"
TILE_WEIGHTS = np.array([[4.0, -2.0], [-3.0, 5.0]], dtype=np.float32)
TILE_BIAS = np.array([0.1, -0.2], dtype=np.float32)
FRAME_WEIGHTS = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
FRAME_BIAS = np.array([0.0, 0.3, -0.1], dtype=np.float32)
EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


def _has_tflite() -> bool:
    try:
        _tflite_interpreter_class()
        return True
    except ImportError:
        return False


def _has_tensorflow() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


needs_tflite = pytest.mark.skipif(not _has_tflite(), reason="LiteRT (ai-edge-litert) is not installed")
needs_tensorflow = pytest.mark.skipif(not _has_tensorflow(), reason="TensorFlow is not installed")


def softmax(values):
    exponent = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def expected_tiles(images):
    """Output of the agent_tiles models: mean per channel -> dense -> softmax."""
    means = np.stack([image.reshape(-1, 2).mean(axis=0) for image in images])
    return softmax(means @ TILE_WEIGHTS + TILE_BIAS)


def _window(points, width=40, height=24, end_t=40_000):
    events = np.zeros(len(points), dtype=EVENT_DTYPE)
    for index, point in enumerate(points):
        events[index] = point
    return opennve.EventWindow(camera="left", start_t=0, end_t=end_t, width=width, height=height, polarity_events=events)


def _random_window(seed=0, count=400, width=40, height=24):
    rng = np.random.default_rng(seed)
    points = sorted(
        (int(t), int(x), int(y), bool(on))
        for t, x, y, on in zip(
            rng.integers(0, 40_000, count), rng.integers(0, width, count), rng.integers(0, height, count), rng.random(count) < 0.6
        )
    )
    return _window(points, width, height)


TILES16 = opennve.FrameConfig(name="tiles16", tiles=opennve.TileConfig(size=16, min_events=0))


# ------------------------------------------------------------------ configuration


def test_agent_config():
    config = opennve.AgentConfig.from_dict(
        {
            "name": "insects",
            "model": "models/insects.tflite",
            "frame": "tiles80",
            "batch_size": 64,
            "preprocess": {"channels": "ON", "resize": [32, 32]},
            "prediction": {"labels": ["background", "insect"], "threshold": 0.5, "activation": "Softmax"},
        },
        base_dir="/data/agents",
    )
    assert config.preprocess.channels == "on" and config.preprocess.resize == (32, 32)
    assert config.prediction.activation == "softmax" and config.input == "tiles"
    assert config.model_path == pathlib.Path("/data/agents/models/insects.tflite")
    assert config.resolved_backend == "tflite"
    assert opennve.AgentConfig.from_dict(config.to_dict()) == config
    assert opennve.AgentConfig.from_json(config.to_json()) == config
    assert opennve.AgentConfig(model="net.keras").name == "net"  # default name: file name
    assert opennve.AgentConfig(model="net.keras").resolved_backend == "tensorflow"
    for data, message in (
        ({}, "agent.model is required"),
        ({"model": "a.tflite", "backend": "torch"}, "agent.backend"),
        ({"model": "a.tflite", "input": "events"}, "agent.input"),
        ({"model": "a.tflite", "batch_size": 0}, "agent.batch_size"),
        ({"model": "a.tflite", "preprocess": {"channels": "rgb"}}, "preprocess.channels"),
        ({"model": "a.tflite", "preprocess": {"resize": [32]}}, "preprocess.resize"),
        ({"model": "a.tflite", "preprocess": {"size": 32}}, "unknown agent.preprocess"),
        ({"model": "a.tflite", "prediction": {"activation": "relu"}}, "prediction.activation"),
        ({"model": "a.tflite", "prediction": {"labels": "insect"}}, "prediction.labels"),
        ({"model": "a.tflite", "tiles": 80}, "unknown agent setting"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.AgentConfig.from_dict(data)
    with pytest.raises(ValueError, match="cannot choose a backend"):
        opennve.AgentConfig(model="model.onnx").resolved_backend


def test_agent_config_files(tmp_path):
    (tmp_path / "models").mkdir()
    path = tmp_path / "agents.json"
    path.write_text(
        json.dumps(
            {
                "agents": [
                    {"name": "a", "model": "models/a.tflite", "frame": "tiles80"},
                    {"name": "b", "model": "models/b.keras", "input": "frame"},
                ]
            }
        )
    )
    configs = opennve.load_agent_configs(path)
    assert [c.name for c in configs] == ["a", "b"]
    assert configs[0].model_path == tmp_path / "models" / "a.tflite"  # relative to the file
    assert opennve.load_agent_config(path, "b").input == "frame"
    with pytest.raises(ValueError, match="specify a name"):
        opennve.load_agent_config(path)
    with pytest.raises(ValueError, match="no agent named 'c'"):
        opennve.load_agent_config(path, "c")
    opennve.save_agent_configs(tmp_path / "copy.json", configs)
    assert opennve.load_agent_configs(tmp_path / "copy.json") == configs
    agents = opennve.load_agents(path)
    assert [a.name for a in agents] == ["a", "b"] and not agents[0].is_loaded  # loaded on first use
    with pytest.raises(FileNotFoundError, match="model not found"):
        agents[0].load()
    path.write_text('{"agents": [{"name": "a", "model": "x.tflite"}, {"name": "a", "model": "y.tflite"}]}')
    with pytest.raises(ValueError, match="duplicate agent name"):
        opennve.load_agent_configs(path)
    path.write_text('{"agents": [{"name": "a", "model": "x.tflite", "batch_size": -1}]}')
    with pytest.raises(ValueError, match="agent 0: agent.batch_size"):
        opennve.load_agent_configs(path)


# ------------------------------------------------------------------ models in code


class MeanModel:
    """Model with predict(batch): mean per input and channel (stands in for a Keras model)."""

    input_shape = (None, 16, 16, 2)

    def __init__(self):
        self.batches = []

    def predict(self, batch, verbose=0):
        self.batches.append(batch.shape)
        means = batch.reshape(len(batch), -1, batch.shape[-1]).mean(axis=1)
        return softmax(means @ TILE_WEIGHTS + TILE_BIAS)


def test_model_object_tiles_padding_and_batches():
    model = MeanModel()
    agent = opennve.Agent.from_model(
        model, name="mean", batch_size=4, prediction={"labels": ["off", "on"], "threshold": 0.6}
    )
    window = _random_window()
    frame = window.frame(TILES16)  # 40 x 24 -> 3 x 2 tiles, the right and bottom tiles are smaller
    assert [(t.width, t.height) for t in frame.tiles][:3] == [(16, 16), (16, 16), (8, 16)]
    result = agent.predict(frame)
    assert len(result) == 6 and model.batches == [(4, 16, 16, 2), (2, 16, 16, 2)]  # batch_size 4
    padded = []
    for tile in frame.tiles:
        image = np.zeros((16, 16, 2), np.float32)
        image[: tile.height, : tile.width] = tile.data
        padded.append(image)
    np.testing.assert_allclose(result.outputs[0], expected_tiles(padded), rtol=1e-5)
    first = result.predictions[0]
    assert first.tile is frame.tiles[0] and first.tile_id == 1
    assert first.label == ("on" if first.class_id == 1 else "off") and first.score == pytest.approx(first.output.max())
    assert [p.tile_id for p in result.selected] == [p.tile_id for p in result.predictions if p.score >= 0.6]
    assert result.by_tile()[3].tile.col == 3 and result.frame is frame
    summary = result.to_dict()
    assert summary["agent"] == "mean" and summary["frame"] == "tiles16" and summary["predictions"][0]["tile_id"] == 1
    json.dumps(summary)  # JSON-friendly


def test_frame_input_channels_and_errors():
    seen = []

    def model(batch):  # a plain callable works as well
        seen.append(batch.shape)
        return batch.reshape(len(batch), -1).sum(axis=1, keepdims=True)

    window = _window([(10, 1, 1, True), (20, 2, 2, False), (30, 3, 3, True)], end_t=40)
    frame = window.frame({"mode": "count"})
    for channels, total in (("both", 3), ("on", 2), ("off", 1), ("sum", 3), ("max", 3)):
        agent = opennve.Agent.from_model(model, input="frame", preprocess={"channels": channels})
        result = agent.predict(frame)
        assert result.predictions[0].tile is None and result.predictions[0].score == total
    assert seen[0] == (1, 24, 40, 2) and seen[1] == (1, 24, 40, 1)
    agent = opennve.Agent.from_model(model, input="frame", preprocess={"layout": "nchw", "scale": 2})
    assert agent.predict(frame).predictions[0].score == 6 and seen[-1] == (1, 2, 24, 40)
    agent = opennve.Agent.from_model(model, preprocess={"resize": [8, 8]})
    agent.predict(window.frame(TILES16))
    assert seen[-1] == (6, 8, 8, 2)
    with pytest.raises(ValueError, match="needs tiles"):
        opennve.Agent.from_model(model).predict(frame)  # input "tiles", frame without tiles
    agent = opennve.Agent.from_model(model, preprocess={"pad": False})
    with pytest.raises(ValueError, match="different sizes"):
        agent.predict(window.frame(TILES16))
    # a frame with tiles but none selected: no predictions
    empty = window.frame({"tiles": {"size": 16, "min_events": 100}})
    assert len(opennve.Agent.from_model(model).predict(empty)) == 0


def test_activation_and_scores():
    logits = np.array([[1.0, 3.0, 0.5]], dtype=np.float32)
    agent = opennve.Agent.from_model(lambda batch: logits, input="frame", prediction={"activation": "softmax"})
    prediction = agent.predict(np.zeros((4, 4, 2), np.float32)).predictions[0]
    assert prediction.class_id == 1 and prediction.score == pytest.approx(softmax(logits)[0, 1])
    agent = opennve.Agent.from_model(
        lambda batch: np.array([[0.0]]), input="frame", prediction={"activation": "sigmoid", "threshold": 0.6}
    )
    prediction = agent.predict(np.zeros((4, 4, 2), np.float32)).predictions[0]
    assert prediction.score == pytest.approx(0.5) and prediction.class_id == 1 and not prediction.selected
    # outputs that are not vectors (for instance maps) have no class or score
    agent = opennve.Agent.from_model(lambda batch: np.zeros((len(batch), 4, 4)), input="frame")
    prediction = agent.predict(np.zeros((4, 4, 2), np.float32)).predictions[0]
    assert prediction.score is None and prediction.output.shape == (4, 4) and prediction.selected


# ------------------------------------------------------------------ TensorFlow Lite


@needs_tflite
def test_tflite_tiles():
    agent = opennve.Agent({"name": "tiles", "model": str(DATA / "agent_tiles.tflite"), "frame": "tiles16"})
    assert agent.config.resolved_backend == "tflite"
    assert agent.input.shape == (None, 16, 16, 2) and agent.input.dtype == np.float32
    frame = _random_window(3).frame(TILES16)
    result = agent.predict(frame)
    padded = []
    for tile in frame.tiles:
        image = np.zeros((16, 16, 2), np.float32)
        image[: tile.height, : tile.width] = tile.data
        padded.append(image)
    np.testing.assert_allclose(result.outputs[0], expected_tiles(padded), rtol=1e-4, atol=1e-6)
    # dynamic batch size: also with batch_size 4 (4 + 2 inputs)
    agent_small = opennve.Agent({"model": str(DATA / "agent_tiles.tflite"), "batch_size": 4})
    np.testing.assert_allclose(agent_small.predict(frame).outputs[0], result.outputs[0], rtol=1e-5)


@needs_tflite
def test_tflite_int8_quantized():
    agent = opennve.Agent({"model": str(DATA / "agent_tiles_int8.tflite"), "threads": 1})
    assert agent.input.dtype == np.int8
    frame = _random_window(4).frame(TILES16)
    result = agent.predict(frame)
    padded = []
    for tile in frame.tiles:
        image = np.zeros((16, 16, 2), np.float32)
        image[: tile.height, : tile.width] = tile.data
        padded.append(image)
    assert result.outputs[0].dtype == np.float32  # dequantized
    np.testing.assert_allclose(result.outputs[0], expected_tiles(padded), atol=0.05)  # int8 steps


@needs_tflite
def test_tflite_frame_fixed_batch_and_resize():
    agent = opennve.Agent(
        {
            "model": str(DATA / "agent_frame.tflite"),
            "input": "frame",
            "preprocess": {"channels": "sum"},
            "prediction": {"activation": "softmax", "labels": ["a", "b", "c"]},
        }
    )
    assert agent.input.shape == (1, 24, 40, 1)
    window = _random_window(5, width=80, height=48)  # resized to the model input 40 x 24
    frame = window.frame({"mode": "binary"})
    result = agent.predict(frame)
    import cv2

    image = cv2.resize(frame.data.sum(axis=2), (40, 24), interpolation=cv2.INTER_AREA)
    expected = softmax(np.array([[image.mean()]]) @ FRAME_WEIGHTS + FRAME_BIAS)
    np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-4)
    assert result.predictions[0].label == "abc"[int(np.argmax(expected))]
    # fixed batch size 1: a batch of 3 inputs is run in 3 calls
    batch = np.random.default_rng(0).random((3, 24, 40, 1), dtype=np.float32)
    outputs = agent.predict_batch(batch)[0]
    np.testing.assert_allclose(outputs, batch.reshape(3, -1).mean(axis=1, keepdims=True) @ FRAME_WEIGHTS + FRAME_BIAS, rtol=1e-4)
    with pytest.raises(ValueError, match="expects 1 channel"):
        opennve.Agent({"model": str(DATA / "agent_frame.tflite"), "input": "frame"}).predict(frame)


@needs_tflite
def test_run_agents_on_raw_file(tmp_path):
    from test_raw import _header, _random_events, encode_evt3

    events = _random_events(7, count=3000, span=120_000)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events))
    frame_configs = [
        opennve.FrameConfig(name="tiles16", width=64, height=32, tiles=opennve.TileConfig(size=16)),
        opennve.FrameConfig(name="full"),
    ]
    agents_path = tmp_path / "agents.json"
    agents_path.write_text(
        json.dumps(
            {
                "agents": [
                    {"name": "tiles", "model": str(DATA / "agent_tiles.tflite"), "frame": "tiles16"},
                    {"name": "tiles_again", "model": str(DATA / "agent_tiles.tflite"), "frame": "tiles16"},
                ]
            }
        )
    )
    with opennve.open_raw(path) as raw:
        steps = list(opennve.run_agents(raw, agents_path, frame_configs))
    assert len(steps) == 3
    step = steps[0]
    assert list(step.frames) == ["tiles16"]  # only the frames needed by the agents, built once
    assert step["tiles"].frame is step["tiles_again"].frame is step.frames["tiles16"]
    assert [p.tile_id for p in step["tiles"]] == [t.id for t in step.frames["tiles16"].tiles]
    assert step.window.start_t == step.frames["tiles16"].start_t
    with pytest.raises(ValueError, match="no frame configuration named 'x'"):
        opennve.process_window(step.window, [{"model": "a.tflite", "frame": "x"}], frame_configs)
    with pytest.raises(ValueError, match="set \"frame\""):
        opennve.process_window(step.window, [{"model": "a.tflite"}], frame_configs)


# ------------------------------------------------------------------ TensorFlow


@needs_tensorflow
@pytest.mark.parametrize("model, call", [("agent_tiles.keras", "predict"), ("agent_tiles.keras", "call"), ("agent_tiles_savedmodel", "predict")])
def test_tensorflow_models(model, call):
    agent = opennve.Agent({"model": str(DATA / model), "call": call, "prediction": {"labels": ["a", "b"]}})
    assert agent.config.resolved_backend == "tensorflow"
    assert agent.input.shape[1:] == (16, 16, 2)
    frame = _random_window(6).frame(TILES16)
    result = agent.predict(frame)
    padded = []
    for tile in frame.tiles:
        image = np.zeros((16, 16, 2), np.float32)
        image[: tile.height, : tile.width] = tile.data
        padded.append(image)
    np.testing.assert_allclose(result.outputs[0], expected_tiles(padded), rtol=1e-4, atol=1e-6)
    assert {p.label for p in result} <= {"a", "b"}
    agent.close()
    assert not agent.is_loaded


@needs_tensorflow
def test_keras_model_object():
    import keras

    model = keras.models.load_model(DATA / "agent_tiles.keras", compile=False)
    agent = opennve.Agent.from_model(model, name="keras")
    frame = _random_window(8).frame(TILES16)
    assert len(agent.predict(frame)) == len(frame.tiles)
