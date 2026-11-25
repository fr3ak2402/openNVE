"""int8 quantization: post-training (opennve quantize) and quantization aware training (--qat) of
a small U-Net, with recordings written for the test."""

import json

import numpy as np
import pytest
from test_agents import _has_tensorflow, _has_tflite

import opennve

needs_tensorflow = pytest.mark.skipif(not (_has_tensorflow() and _has_tflite()),
                                      reason="TensorFlow and LiteRT are needed")

SMALL_UNET = {"layer_depth": 2, "filters_orig": 4, "kernel_size": 3, "final_activation": None}
PREDICTION = {"task": "segmentation", "activation": "softmax", "target_classes": [2]}
FRAMES = {"frames": [{"name": "t16", "mode": "count", "tiles": {"size": 16, "min_events": 1}}]}


def _events(seed=0, count=6000, width=64, height=48, duration=400_000):
    rng = np.random.default_rng(seed)
    events = np.zeros(count, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
    events["t"] = np.sort(rng.integers(0, duration, count))
    events["x"] = rng.integers(0, width, count)
    events["y"] = rng.integers(0, height, count)
    events["on"] = rng.random(count) < 0.5
    return events


@pytest.fixture
def setup(tmp_path):
    """A small U-Net with non-trivial batch normalization, a recording, agents.json and frames.json."""
    network = opennve.Network.build("UNet", opennve.NetworkParameters(3, 16, 16, 2), **SMALL_UNET)
    rng = np.random.default_rng(1)
    for layer in network.model._flatten_layers(include_self=False):
        if layer.__class__.__name__ == "BatchNormalization":
            size = layer.gamma.shape[0]
            layer.gamma.assign(rng.uniform(0.5, 1.5, size).astype(np.float32))
            layer.beta.assign(rng.normal(0, 0.2, size).astype(np.float32))
            layer.moving_mean.assign(rng.normal(0, 0.3, size).astype(np.float32))
            layer.moving_variance.assign(rng.uniform(0.5, 2.0, size).astype(np.float32))
    network.save(tmp_path / "unet.keras")
    opennve.write_hdf5(tmp_path / "train.h5", _events(), width=64, height=48)
    (tmp_path / "frames.json").write_text(json.dumps(FRAMES))
    (tmp_path / "agents.json").write_text(json.dumps({"agents": [
        {"name": "unet", "model": "unet.keras", "frame": "t16", "input": "tiles", "prediction": PREDICTION}]}))
    return tmp_path, network


@needs_tensorflow
def test_folding_keeps_the_network(setup):
    from opennve.agents.quantize import _find_unet
    from opennve.networks.qat import QuantizedUNet, _fold

    _, network = setup
    unet = _find_unet(network.model)
    block = unet.encoder_blocks[0]
    x = np.random.default_rng(2).random((2, 16, 16, 2)).astype(np.float32)
    expected = block.bn1(block.conv1(x), training=False).numpy()
    kernel, bias = _fold(block.conv1, block.bn1)
    import tensorflow as tf

    folded = (tf.nn.conv2d(x, kernel, 1, "SAME") + bias).numpy()
    np.testing.assert_allclose(folded, expected, rtol=1e-4, atol=1e-4)

    # the quantized network after calibration gives almost the same classes as the float network
    student = QuantizedUNet.from_unet(unet, (16, 16, 2))
    inputs = np.random.default_rng(3).random((64, 16, 16, 2)).astype(np.float32) * (
        np.random.default_rng(4).random((64, 16, 16, 2)) < 0.2)
    student.calibrate([inputs[i:i + 16] for i in range(0, 64, 16)])
    same = np.mean(student(inputs).numpy().argmax(-1) == network.model.predict(inputs, verbose=0).argmax(-1))
    assert same > 0.8


@needs_tensorflow
def test_distillation_helpers():
    import tensorflow as tf

    from opennve.networks.qat import agreement, classes_of, distillation_loss

    teacher = tf.constant([[[[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]]])
    assert float(distillation_loss(teacher, teacher, "linear", "softmax")) < 0.1  # entropy of the teacher
    worse = float(distillation_loss(teacher, tf.reverse(teacher, [-1]), "linear", "softmax"))
    weighted = float(distillation_loss(teacher, tf.reverse(teacher, [-1]), "linear", "softmax", [2], 10.0))
    assert worse > 1.0 and weighted > 1.0
    result = agreement(np.array([0, 2, 2, 1]), np.array([0, 2, 1, 1]), [2])
    assert result["agreement"] == 0.75 and result["classes"][2] == {"pixels": 2, "iou": 0.5, "recall": 0.5}
    assert classes_of(np.array([[0.1, 0.9]]), "softmax").tolist() == [1]


@needs_tensorflow
def test_post_training_and_qat(setup):
    from opennve.agents import quantize_agent, quantize_report

    path, _ = setup
    config = opennve.load_agent_config(path / "agents.json")
    frame = opennve.load_frame_config(path / "frames.json", "t16")
    report = quantize_agent(config, [path / "train.h5"], frame, samples=40, validation=12, stride=1)
    assert report["method"] == "post-training" and (path / "unet_int8.tflite").is_file()
    assert report["calibration"] == 40 and report["validation"] == 12 and report["agreement"] is not None
    assert "agreement" in quantize_report(report)

    messages = []
    report = quantize_agent(config, [path / "train.h5"], frame, path / "unet_qat.tflite", qat=True, samples=48,
                            validation=16, epochs=2, batch_size=8, stride=1, progress=messages.append)
    assert report["method"] == "qat" and (path / "unet_qat.tflite").is_file()
    assert [entry["epoch"] for entry in report["history"]] == [0, 1, 2]
    assert report["agreement"] is not None and 2 in report["classes"]
    assert any("epoch 2/2" in message for message in messages)
    text = quantize_report(report)
    assert "training" in text and "LiteRT" in text

    # the int8 model runs as an agent
    agent = opennve.Agent(config.copy(model=str(path / "unet_qat.tflite"), backend="tflite"))
    window = opennve.EventWindow(camera="c", start_t=0, end_t=40_000, width=64, height=48, polarity_events=_events(5, 400, duration=40_000))
    assert agent.predict(window.frame(frame)).predictions
    agent.close()


@needs_tensorflow
def test_qat_needs_a_unet(setup):
    from opennve.agents import quantize_agent

    path, _ = setup
    import keras

    inputs = keras.Input((16, 16, 2))
    keras.Model(inputs, keras.layers.Conv2D(3, 1)(inputs)).save(path / "plain.keras")
    config = opennve.load_agent_config(path / "agents.json").copy(model=str(path / "plain.keras"))
    frame = opennve.load_frame_config(path / "frames.json", "t16")
    with pytest.raises(ValueError, match="opennve U-Net"):
        quantize_agent(config, [path / "train.h5"], frame, qat=True, samples=8, validation=0, stride=1)
    with pytest.raises(ValueError, match="recordings"):
        quantize_agent(config, [], frame)


@needs_tensorflow
def test_cli(setup, capsys):
    from opennve.cli import main

    path, _ = setup
    code = main(["quantize", str(path / "agents.json"), "--data", str(path / "*.h5"), "--frames",
                 str(path / "frames.json"), "--qat", "--samples", "32", "--validation", "8", "--epochs", "1",
                 "--batch-size", "8", "--stride", "1", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == "qat" and report["output"].endswith("unet_int8.tflite")
    assert main(["quantize", str(path / "agents.json"), "--data", str(path / "missing.h5"), "--frames",
                 str(path / "frames.json")]) == 2
