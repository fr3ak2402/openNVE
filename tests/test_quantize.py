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


def test_logit_loss():
    """Compares the logits: an unsure teacher (probabilities close together) still gives a loss of
    0 for the same logits; a common offset of all classes does not count (softmax)."""
    import tensorflow as tf

    from opennve.networks.qat import logit_loss

    teacher = tf.constant([[[[0.2, 0.0, 0.1], [0.0, 0.05, 0.3]]]])
    assert float(logit_loss(teacher, teacher, "softmax")) == 0.0
    assert float(logit_loss(teacher, teacher + 7.0, "softmax")) < 1e-10
    assert float(logit_loss(teacher, tf.reverse(teacher, [-1]), "softmax", scale=0.1)) > 1.0
    assert float(logit_loss(teacher, tf.reverse(teacher, [-1]), "softmax", 0.1, [2], 10.0)) > 1.0


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
    assert any("without the 8 bit rounding: 100.000 %" in message for message in messages)
    assert any("highest probability" in message for message in messages)
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


def _functional_unet(depth=2, filters=4, classes=3, size=16):
    """A functional U-Net with the layer names of the HSNR U-Net (enc_conv0-1 ... dec_output)."""
    import keras
    import tensorflow as tf

    def block(x, name, count, bn_name, act):
        x = keras.layers.Conv2D(count, 3, padding="same", name=name)(x)
        x = keras.layers.BatchNormalization(name=bn_name)(x)
        return keras.layers.LeakyReLU(negative_slope=0.3, name=act)(x)

    inputs = keras.Input((size, size, 2))
    x, skips, count = inputs, [], filters
    for i in range(depth):
        x = block(x, f"enc_conv{i}-1", count, f"enc_bn{i}-1", f"enc_act{i}-1")
        x = block(x, f"enc_conv{i}-2", count, f"enc_bn{i}-2", f"enc_act{i}-2")
        skips.append(x)
        x = keras.layers.MaxPooling2D(2)(x)
        count *= 2
    x = block(x, "enc_convbtleneck-1", count, "enc_bnbtleneck-1", "btl_act1")
    x = block(x, "enc_convbtleneck-2", count, "enc_bnbtleneck-2", "btl_act2")
    for i in range(depth):
        count = max(count // 2, filters)
        x = block(x, f"dec_conv{i}-1", count, f"dec_bn{i}-1", f"dec_act{i}-1")
        x = keras.layers.Conv2DTranspose(count, 3, strides=2, padding="same", name=f"dec_conv{i}-2")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn{i}-2")(x)
        x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
        x = keras.layers.Concatenate()([x, skips[-(i + 1)]])
    x = keras.layers.Conv2D(count, 3, padding="same", activation="relu", name="dec_out1")(x)
    x = keras.layers.Conv2D(count, 3, padding="same", activation="relu", name="dec_out2")(x)
    return keras.Model(inputs, keras.layers.Conv2D(classes, 1, name="dec_output")(x))


@needs_tensorflow
def test_qat_of_a_functional_unet(setup):
    """unet_float32.keras of embedded-unet (functional, HSNR layer names) is copied into the opennve
    U-Net automatically."""
    from opennve.agents import quantize_agent

    path, _ = setup
    _functional_unet().save(path / "unet_float32.keras")
    config = opennve.load_agent_config(path / "agents.json").copy(model=str(path / "unet_float32.keras"))
    frame = opennve.load_frame_config(path / "frames.json", "t16")
    messages = []
    report = quantize_agent(config, [path / "train.h5"], frame, qat=True, samples=32, validation=8, epochs=1,
                            batch_size=8, stride=1, progress=messages.append)
    assert (path / "unet_float32_int8.tflite").is_file() and report["agreement"] is not None
    assert any("functional U-Net" in message for message in messages)


def test_window_at_reads_a_window_directly(tmp_path):
    pytest.importorskip("h5py")
    events = _events(count=20000, duration=2_000_000)
    opennve.write_hdf5(tmp_path / "rec.h5", events, width=64, height=48)
    with opennve.open_hdf5(tmp_path / "rec.h5") as recording:
        assert recording.event_count == 20000
        window = recording.window_at(5000, 40)
        start = int(events["t"][5000])
        expected = events[(events["t"] >= start) & (events["t"] < start + 40_000)]
        assert window.start_t == start and window.end_t == start + 40_000
        assert len(window.polarity_events) == len(expected)
        np.testing.assert_array_equal(window.polarity_events["x"], expected["x"])
        assert recording.window_at(19999, 40).polarity_events["t"][0] == events["t"][-1]


@needs_tensorflow
def test_random_and_sequential_sampling(setup):
    from opennve.agents.quantize import collect_prepared

    path, _ = setup
    config = opennve.load_agent_config(path / "agents.json")
    frame = opennve.load_frame_config(path / "frames.json", "t16")
    agent = opennve.Agent(config)
    messages = []
    data, stats = collect_prepared(agent, [path / "train.h5"], frame, 30, progress=messages.append)
    assert data.shape == (30, 16, 16, 2) and data.dtype == np.float16
    assert any("64 x 48" in message for message in messages)
    data, stats = collect_prepared(agent, [path / "train.h5"], frame, 30, sampling="sequential", stride=1)
    assert len(data) == 30 and stats["windows"] == 10
    agent.close()


@needs_tensorflow
def test_qat_of_a_model_with_softmax_around_the_unet(setup):
    """A saved model that wraps the U-Net and applies a softmax after it: the quantized network
    must do the same (else it learns from probabilities as if they were logits)."""
    import keras

    from opennve.agents import quantize_agent
    from opennve.agents.quantize import _find_unet

    path, network = setup
    unet = _find_unet(network.model)
    head = unet.output_layer
    head.kernel.assign(head.kernel.numpy() * 200)  # confident like a trained network (logits of hundreds)
    inputs = keras.Input((16, 16, 2))
    keras.Model(inputs, keras.layers.Softmax()(unet(inputs))).save(path / "wrapped.keras")
    config = opennve.load_agent_config(path / "agents.json").copy(model=str(path / "wrapped.keras"))
    frame = opennve.load_frame_config(path / "frames.json", "t16")
    messages = []
    report = quantize_agent(config, [path / "train.h5"], frame, qat=True, samples=64, validation=16, epochs=2,
                            batch_size=8, progress=messages.append)
    assert any("applies softmax after the U-Net" in message for message in messages)
    losses = [entry["loss"] for entry in report["history"][1:]]
    assert losses[-1] < 0.6  # a mix-up of probabilities and logits gives about ln(3) = 1.1
    assert report["agreement"] > 0.9
