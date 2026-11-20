"""Tests for networks (registry, UNet, NeuralEngine, TensorFlow Lite export) and the segmentation task.

Tests that need TensorFlow or LiteRT are skipped if they are not installed.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import opennve
from test_agents import EVENT_DTYPE, _has_tensorflow, _has_tflite, _window

needs_tensorflow = pytest.mark.skipif(not _has_tensorflow(), reason="TensorFlow is not installed")
needs_tflite = pytest.mark.skipif(not _has_tflite(), reason="LiteRT (ai-edge-litert) is not installed")


# ------------------------------------------------------------------ registry (no TensorFlow needed)


def test_network_parameters_and_registry():
    parameters = opennve.NetworkParameters.from_dict(
        {"NUMBER_OF_CLASSES": 3, "TILE_HEIGHT": 80, "TILE_WIDTH": 80, "CHANNELS": 2}
    )
    assert parameters == opennve.NetworkParameters(3) and parameters.input_shape == (80, 80, 2)
    with pytest.raises(ValueError, match="number_of_classes"):
        opennve.NetworkParameters(0)
    with pytest.raises(ValueError, match="unknown network parameter"):
        opennve.NetworkParameters.from_dict({"classes": 3})
    assert "UNet" in opennve.list_networks()

    @opennve.register_network("Dummy")
    def build_dummy(parameters, scale=1):
        return {"shape": parameters.input_shape, "scale": scale}

    try:
        assert opennve.networks.build_network("Dummy", {"number_of_classes": 2}, scale=3) == {
            "shape": (80, 80, 2),
            "scale": 3,
        }
        assert "Dummy" in opennve.list_networks()
    finally:
        opennve.networks.unregister_network("Dummy")
    with pytest.raises(ValueError, match="unknown network 'Dummy'"):
        opennve.networks.build_network("Dummy", opennve.NetworkParameters(2))


def test_custom_tasks_and_backends():
    @opennve.register_task("count")
    def count(task):
        return [opennve.Prediction(tile=task.tile(i), output=task.outputs[0][i], outputs=[], score=float(task.outputs[0][i].sum())) for i in range(task.count)]

    agent = opennve.Agent.from_model(lambda batch: batch.reshape(len(batch), -1), input="frame", prediction={"task": "count"})
    assert agent.predict(np.ones((2, 3, 2), np.float32)).predictions[0].score == 12
    with pytest.raises(ValueError, match="prediction.task"):
        opennve.AgentConfig.from_dict({"model": "a.tflite", "prediction": {"task": "tracking"}})

    class FakeBackend(opennve.agents.Backend):
        def __init__(self, config):
            self.input = opennve.agents.ModelInput(shape=(None, None, None, 2), dtype=np.dtype(np.float32))

        def predict(self, batch):
            return [np.full((len(batch), 1), 0.75, np.float32)]

    opennve.register_backend("fake", FakeBackend, suffixes=[".fake"])
    assert opennve.AgentConfig(model="model.fake").resolved_backend == "fake"


# ------------------------------------------------------------------ segmentation task (model in code)


def pixel_model(batch):
    """Logits per pixel: class 2 where channel 1 (polarity 1) has events, class 1 where only
    channel 0 has events, else class 0 (batch x height x width x 3)."""
    on = batch[..., 1] > 0
    off = (batch[..., 0] > 0) & ~on
    logits = np.zeros(batch.shape[:3] + (3,), np.float32)
    logits[..., 0] = 1.0
    logits[..., 1] = np.where(off, 4.0, -1.0)
    logits[..., 2] = np.where(on, 5.0, -1.0)
    return logits


SEGMENTATION = {
    "task": "segmentation",
    "activation": "softmax",
    "labels": ["background", "noise", "insect"],
    "target_classes": [2],
}


def test_segmentation_tiles_coordinates_and_statistics():
    # 40 x 24 frame, 16 px tiles: the right column (x 32..39) and the bottom row (y 16..23) are smaller
    points = [(10, 3, 4, True), (20, 35, 20, True), (30, 18, 2, False), (35, 36, 21, True)]
    window = _window(points, end_t=40)
    frame = window.frame({"name": "t16", "mode": "binary", "tiles": {"size": 16, "min_events": 1}})
    agent = opennve.Agent.from_model(pixel_model, name="insects", prediction=SEGMENTATION)
    result = agent.predict(frame)
    assert [p.tile_id for p in result] == [1, 2, 6]
    first, second, edge = result.predictions
    assert first.pixel_list() == [
        {
            "x": 3,
            "y": 4,
            "class_id": 2,
            "score": pytest.approx(first.pixels["score"][0]),
            "scores": {"background": pytest.approx(first.pixel_scores[0][0]), "noise": pytest.approx(first.pixel_scores[0][1]), "insect": pytest.approx(first.pixel_scores[0][2])},
        }
    ]
    assert first.label == "insect" and first.selected and first.class_map.shape == (16, 16)
    assert first.score == pytest.approx(np.exp(5) / (np.exp(1) + np.exp(-1) + np.exp(5)))
    assert second.pixel_count == 0 and not second.selected and second.class_counts.tolist() == [255, 1, 0]
    # edge tile 8 x 8 (padded to 16 x 16 for the model, cropped back), frame coordinates
    assert edge.class_map.shape == (8, 8) and sorted(edge.pixels[["x", "y"]].tolist()) == [(35, 20), (36, 21)]
    assert [p.tile_id for p in result.selected] == [1, 6]
    stats = result.statistics()
    assert stats["pixels"] == 16 * 16 * 2 + 8 * 8 and stats["active_pixels"] == 4
    assert stats["active_class_pixels"] == {"background": 0, "noise": 1, "insect": 3}
    assert stats["target_pixels"] == 3 and stats["target_score"]["min"] > 0.9
    assert "insect 3 (75.00%)" in result.statistics_report()
    summary = result.to_dict(selected_only=True)
    json.dumps(summary)
    assert summary["predictions"][1]["pixels"][0]["scores"]["insect"] > 0.9


def test_segmentation_options():
    points = [(10, 3, 4, True), (30, 18, 2, False)]
    frame = _window(points, end_t=40).frame({"mode": "binary", "tiles": {"size": 16, "min_events": 1}})
    # all classes except 0, min_pixels, threshold
    agent = opennve.Agent.from_model(pixel_model, prediction={**SEGMENTATION, "target_classes": None})
    assert [p.pixel_count for p in agent.predict(frame)] == [1, 1]
    agent = opennve.Agent.from_model(pixel_model, prediction={**SEGMENTATION, "min_pixels": 2})
    assert agent.predict(frame).selected == []
    agent = opennve.Agent.from_model(pixel_model, prediction={**SEGMENTATION, "threshold": 0.99})
    assert agent.predict(frame).predictions[0].pixel_count == 0  # probability 0.98 < 0.99

    # mask_inactive: a model that marks every pixel as class 2
    def everything(batch):
        logits = np.zeros(batch.shape[:3] + (3,), np.float32)
        logits[..., 2] = 3
        return logits

    agent = opennve.Agent.from_model(everything, prediction={**SEGMENTATION, "mask_inactive": True})
    assert [p.pixel_count for p in agent.predict(frame)] == [1, 1]
    agent = opennve.Agent.from_model(everything, prediction=SEGMENTATION)
    assert [p.pixel_count for p in agent.predict(frame)] == [256, 256]

    # output with half resolution is scaled to the tile size; nchw layout
    agent = opennve.Agent.from_model(lambda batch: pixel_model(batch)[:, ::2, ::2], prediction=SEGMENTATION)
    assert agent.predict(frame).predictions[0].class_map.shape == (16, 16)
    agent = opennve.Agent.from_model(
        lambda batch: pixel_model(batch.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2),
        preprocess={"layout": "nchw"},
        prediction=SEGMENTATION,
    )
    assert agent.predict(frame).predictions[0].pixels[["x", "y"]].tolist() == [(3, 4)]
    # whole frame as input
    agent = opennve.Agent.from_model(pixel_model, input="frame", prediction=SEGMENTATION)
    prediction = agent.predict(frame).predictions[0]
    assert prediction.tile is None and prediction.pixels[["x", "y"]].tolist() == [(3, 4)]
    with pytest.raises(ValueError, match="height x width x classes"):
        opennve.Agent.from_model(lambda batch: np.zeros((len(batch), 3)), prediction=SEGMENTATION).predict(frame)


# ------------------------------------------------------------------ TensorFlow


SMALL_UNET = {"layer_depth": 2, "filters_orig": 4}
PARAMETERS = opennve.NetworkParameters(number_of_classes=3, tile_height=16, tile_width=16, channels=2)


def _tiles_frame(seed=0):
    rng = np.random.default_rng(seed)
    count = 300
    points = sorted(
        (int(t), int(x), int(y), bool(on))
        for t, x, y, on in zip(rng.integers(0, 40_000, count), rng.integers(0, 40, count), rng.integers(0, 24, count), rng.random(count) < 0.5)
    )
    events = np.array(points, dtype=EVENT_DTYPE)
    window = opennve.EventWindow(camera="c", start_t=0, end_t=40_000, width=40, height=24, polarity_events=events)
    return window.frame({"name": "t16", "tiles": {"size": 16, "min_events": 0}})


@needs_tensorflow
def test_unet_build_save_load_and_agent(tmp_path):
    network = opennve.Network.build("UNet", PARAMETERS, name="unet", **SMALL_UNET)
    assert network.input_shape == (None, 16, 16, 2) and network.output_shape == (None, 16, 16, 3)
    frame = _tiles_frame()
    outputs = network.predict(frame)  # tiles padded to 16 x 16
    assert outputs.shape == (6, 16, 16, 3)
    path = tmp_path / "unet.keras"
    network.save(path)
    loaded = opennve.Network.load(path)
    assert loaded.compare_weights(network).equal
    np.testing.assert_allclose(loaded.predict(frame), outputs, rtol=1e-5, atol=1e-6)
    network.save_weights(tmp_path / "unet.weights.h5")
    other = opennve.Network.build("UNet", PARAMETERS, **SMALL_UNET)
    assert not other.compare_weights(network).equal
    other.load_weights(tmp_path / "unet.weights.h5")
    assert other.compare_weights(network).equal

    # agent with the saved model (segmentation)
    agent = opennve.Agent({"model": str(path), "prediction": SEGMENTATION})
    result = agent.predict(frame)
    expected = np.exp(outputs) / np.exp(outputs).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-4, atol=1e-6)
    assert result.predictions[0].class_map.shape == (16, 16) and result.predictions[2].class_map.shape == (16, 8)
    # agent directly from the network
    assert len(network.agent(prediction=SEGMENTATION).predict(frame)) == 6


LEGACY_CODE = '''
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class UNetConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=4, batch_norm=True, dropout=False, name_prefix="enc", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters; self.kernel_size = kernel_size; self.batch_norm = batch_norm
        self.dropout = dropout; self.name_prefix = name_prefix
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.act1 = tf.keras.layers.LeakyReLU(negative_slope=0.3)
        self.drop = tf.keras.layers.Dropout(0.2) if dropout else None
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.act2 = tf.keras.layers.LeakyReLU(negative_slope=0.3)
    def call(self, x, training=False):
        x = self.act1(self.bn1(self.conv1(x), training=training))
        return self.act2(self.bn2(self.conv2(x), training=training))
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "batch_norm": self.batch_norm,
                       "dropout": self.dropout, "name_prefix": self.name_prefix})
        return config

@tf.keras.utils.register_keras_serializable()
class UNetDeconvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, batch_norm=True, dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters; self.kernel_size = kernel_size; self.strides = strides
        self.batch_norm = batch_norm; self.dropout = dropout
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.act1 = tf.keras.layers.LeakyReLU(negative_slope=0.3)
        self.drop = tf.keras.layers.Dropout(0.2) if dropout else None
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, kernel_initializer="he_normal", padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.act2 = tf.keras.layers.LeakyReLU(negative_slope=0.3)
    def call(self, x, training=False):
        x = self.act1(self.bn1(self.conv1(x), training=training))
        if self.drop:
            x = self.drop(x, training=training)
        return self.act2(self.bn2(self.deconv(x), training=training))
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
                       "batch_norm": self.batch_norm, "dropout": self.dropout})
        return config

@tf.keras.utils.register_keras_serializable()
class UNet(tf.keras.Model):
    def __init__(self, out_channels=3, layer_depth=4, filters_orig=32, kernel_size=4, batch_norm=True, final_activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.layer_depth = layer_depth; self.filters_orig = filters_orig
        self.encoder_blocks = []; self.pools = []
        filters = filters_orig
        for _ in range(layer_depth):
            self.encoder_blocks.append(UNetConvBlock(filters=filters, kernel_size=kernel_size, batch_norm=batch_norm))
            self.pools.append(tf.keras.layers.MaxPooling2D(pool_size=2))
            filters = min(filters * 2, 512)
        self.bottleneck = UNetConvBlock(filters=filters, kernel_size=kernel_size, batch_norm=batch_norm)
        self.decoder_blocks = []
        for i in range(layer_depth):
            filters = max(filters // 2, filters_orig)
            self.decoder_blocks.append(UNetDeconvBlock(filters=filters, kernel_size=kernel_size, strides=2, batch_norm=batch_norm, dropout=i < (layer_depth - 2)))
        self.dec_out1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")
        self.drop_out1 = tf.keras.layers.Dropout(0.3)
        self.dec_out2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")
        self.drop_out2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, activation=final_activation, padding="same")
    def call(self, x, training=False):
        skips = []
        for block, pool in zip(self.encoder_blocks, self.pools):
            c = block(x, training=training); skips.append(c); x = pool(c)
        x = self.bottleneck(x, training=training)
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, training=training)
            shortcut = skips[-(i + 1)]
            x = tf.image.resize(x, tf.shape(shortcut)[1:3])
            x = tf.concat([x, shortcut], axis=-1)
        x = self.drop_out1(self.dec_out1(x), training=training)
        x = self.drop_out2(self.dec_out2(x), training=training)
        return self.output_layer(x)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.output_layer.filters, "layer_depth": self.layer_depth, "filters_orig": self.filters_orig,
                       "kernel_size": self.encoder_blocks[0].kernel_size, "batch_norm": self.encoder_blocks[0].batch_norm,
                       "final_activation": self.output_layer.activation.__name__})
        return config

inputs = tf.keras.Input(shape=(16, 16, 2))
model = tf.keras.Model(inputs, UNet(out_channels=3, layer_depth=2, filters_orig=4, kernel_size=3, final_activation="linear")(inputs))
model.save(sys.argv[1])
import numpy as np
x = np.random.default_rng(0).random((2, 16, 16, 2), dtype=np.float32)
np.save(sys.argv[2], np.stack([x, model.predict(x, verbose=0)[..., :2]]))
'''


@needs_tensorflow
def test_load_model_saved_with_the_original_classes(tmp_path):
    """Models saved by the original project code (classes registered as Custom>UNet) load with openNVE."""
    script = tmp_path / "legacy.py"
    script.write_text(LEGACY_CODE)
    path = tmp_path / "legacy.keras"
    subprocess.run([sys.executable, str(script), str(path), str(tmp_path / "io.npy")], check=True, cwd=tmp_path)
    data = np.load(tmp_path / "io.npy")
    network = opennve.Network.load(path)
    output = network.predict(data[0])
    np.testing.assert_allclose(output[..., :2], data[1], rtol=1e-4, atol=1e-5)
    agent = opennve.Agent({"model": str(path), "prediction": SEGMENTATION})
    assert agent.input.shape == (None, 16, 16, 2)


def _hsnr_like_unet(layer_depth=2, filters_orig=4, kernel_size=3):
    """Functional U-Net with the layer names of the HSNR U-Net (enc_conv{i}-1, ...)."""
    import keras

    inputs = keras.Input((16, 16, 2))
    x = inputs
    skips = []
    filters = filters_orig
    for i in range(layer_depth):
        for j in (1, 2):
            x = keras.layers.Conv2D(filters, kernel_size, padding="same", name=f"enc_conv{i}-{j}")(x)
            x = keras.layers.BatchNormalization(name=f"enc_bn{i}-{j}")(x)
            x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
        skips.append(x)
        x = keras.layers.MaxPooling2D(2)(x)
        filters = min(filters * 2, 512)
    for j in (1, 2):
        x = keras.layers.Conv2D(filters, kernel_size, padding="same", name=f"enc_convbtleneck-{j}")(x)
        x = keras.layers.BatchNormalization(name=f"enc_bnbtleneck-{j}")(x)
        x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)
        x = keras.layers.Conv2D(filters, kernel_size, padding="same", name=f"dec_conv{i}-1")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn{i}-1")(x)
        x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
        x = keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same", name=f"dec_conv{i}-2")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn{i}-2")(x)
        x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
        x = keras.layers.Concatenate()([x, skips[-(i + 1)]])
    x = keras.layers.Conv2D(filters, kernel_size, activation="relu", padding="same", name="dec_out1")(x)
    x = keras.layers.Conv2D(filters, kernel_size, activation="relu", padding="same", name="dec_out2")(x)
    outputs = keras.layers.Conv2D(3, 1, padding="same", name="dec_output")(x)
    return keras.Model(inputs, outputs)


@needs_tensorflow
def test_neural_engine_transfer_weights_and_tflite(tmp_path):
    engine = opennve.NeuralEngine(model_directory=tmp_path)
    unet = engine.add_network("unet", "UNet", parameters=PARAMETERS, kernel_size=3, **SMALL_UNET)
    source = _hsnr_like_unet()
    # randomize the batch normalization statistics so the transfer is visible
    rng = np.random.default_rng(1)
    source.set_weights([w + rng.normal(0, 0.1, w.shape).astype(w.dtype) if w.ndim == 1 else w for w in source.get_weights()])
    source.get_layer("enc_bn0-1").set_weights([np.abs(w) + 0.5 for w in source.get_layer("enc_bn0-1").get_weights()])
    engine.add(source, name="hsnr")
    assert engine.names == ["unet", "hsnr"] and engine[1] is engine["hsnr"]
    comparison = engine.transfer_weights("unet", "hsnr", layer_depth=2)
    assert comparison.count_a == comparison.count_b
    x = rng.random((3, 16, 16, 2), dtype=np.float32)
    np.testing.assert_allclose(unet.predict(x), source.predict(x, verbose=0), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="unknown weight transfer"):
        engine.transfer_weights("unet", "hsnr", method="other")

    # TensorFlow Lite export with quantization
    frames = [_tiles_frame(seed) for seed in range(4)]
    dataset = opennve.representative_dataset(frames, input_shape=(16, 16, 2))
    paths = {q: engine.quantize("unet", q, representative_dataset=dataset) for q in ("none", "dynamic", "float16", "int8")}
    assert sorted(p.name for p in paths.values()) == ["unet.tflite", "unet_dynamic.tflite", "unet_float16.tflite", "unet_int8.tflite"]
    with pytest.raises(ValueError, match="representative dataset"):
        unet.to_tflite(tmp_path / "x.tflite", "int8")
    if _has_tflite():
        frame = frames[0]
        reference = opennve.Agent.from_model(unet.model, prediction=SEGMENTATION).predict(frame)
        for quantization, path in paths.items():
            result = opennve.Agent({"model": str(path), "prediction": SEGMENTATION}).predict(frame)
            tolerance = {"none": 1e-4, "dynamic": 0.05, "float16": 0.01, "int8": 0.15}[quantization]
            np.testing.assert_allclose(result.outputs[0], reference.outputs[0], atol=tolerance)


@needs_tensorflow
def test_representative_dataset_from_raw(tmp_path):
    from test_raw import _header, _random_events, encode_evt3

    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(_random_events(2, count=4000, span=120_000)))
    with opennve.open_raw(path) as raw:
        dataset = opennve.representative_dataset((raw, {"tiles": {"size": 80, "min_events": 3}}), max_samples=10)
        samples = list(dataset())
    assert len(samples) == 10 and samples[0][0].shape == (1, 80, 80, 2)
    with pytest.raises(ValueError, match="empty"):
        opennve.representative_dataset([])
