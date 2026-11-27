"""Weight transfer between networks of the same layers (opennve transfer) and portable loading of
models with Lambda layers."""

import json
import zipfile

import numpy as np
import pytest
from test_agents import _has_tensorflow

import opennve

needs_tensorflow = pytest.mark.skipif(not _has_tensorflow(), reason="TensorFlow is needed")


def _hsnr_unet(depth=2, filters=4, classes=3, size=16, seed=0):
    """A functional U-Net as the HSNR U-Net: named layers, resize_to_same Lambda layers, and
    weights that look trained (batch normalization and biases not at their initial values)."""
    import keras

    def block(x, name, count, bn_name):
        x = keras.layers.Conv2D(count, 3, padding="same", name=name)(x)
        x = keras.layers.BatchNormalization(name=bn_name)(x)
        return keras.layers.LeakyReLU(negative_slope=0.3)(x)

    inputs = keras.Input((size, size, 2))
    x, skips, count = inputs, [], filters
    for i in range(depth):
        x = block(x, f"enc_conv{i}-1", count, f"enc_bn{i}-1")
        x = block(x, f"enc_conv{i}-2", count, f"enc_bn{i}-2")
        skips.append(x)
        x = keras.layers.MaxPooling2D(2)(x)
        count *= 2
    x = block(x, "enc_convbtleneck-1", count, "enc_bnbtleneck-1")
    x = block(x, "enc_convbtleneck-2", count, "enc_bnbtleneck-2")
    for i in range(depth):
        count = max(count // 2, filters)
        x = block(x, f"dec_conv{i}-1", count, f"dec_bn{i}-1")
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv2DTranspose(count, 3, strides=2, padding="same", name=f"dec_conv{i}-2")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn{i}-2")(x)
        x = keras.layers.LeakyReLU(negative_slope=0.3)(x)
        shortcut = skips[-(i + 1)]
        x = keras.layers.Lambda(lambda t: __import__("tensorflow").image.resize(t[0], __import__("tensorflow").shape(t[1])[1:3]),
                                name=f"resize_to_same{i}")([x, shortcut])
        x = keras.layers.Concatenate()([x, shortcut])
    x = keras.layers.Conv2D(count, 3, padding="same", activation="relu", name="dec_out1")(x)
    x = keras.layers.Conv2D(count, 3, padding="same", activation="relu", name="dec_out2")(x)
    model = keras.Model(inputs, keras.layers.Conv2D(classes, 1, name="dec_output")(x))
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        if layer.__class__.__name__ == "BatchNormalization":
            size = layer.gamma.shape[0]
            layer.gamma.assign(rng.uniform(0.5, 1.5, size).astype(np.float32))
            layer.beta.assign(rng.normal(0, 0.2, size).astype(np.float32))
            layer.moving_mean.assign(rng.normal(0, 0.3, size).astype(np.float32))
            layer.moving_variance.assign(rng.uniform(0.5, 2.0, size).astype(np.float32))
        elif hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.assign(rng.normal(0, 0.1, layer.bias.shape).astype(np.float32))
    return model


@pytest.fixture
def hsnr(tmp_path):
    model = _hsnr_unet()
    model.save(tmp_path / "original.keras")
    model.save_weights(tmp_path / "trained.weights.h5")
    return tmp_path, model


@needs_tensorflow
def test_lambda_layers_load_as_portable_layers(hsnr):
    """The resize_to_same Lambda layers (Python bytecode) are replaced: the model loads in safe
    mode and gives the same outputs."""
    from opennve.networks import load_keras_model

    path, model = hsnr
    with zipfile.ZipFile(path / "original.keras") as archive:
        assert '"Lambda"' in archive.read("config.json").decode()
    loaded = load_keras_model(path / "original.keras")  # safe mode: a Lambda layer would be refused
    assert loaded.opennve_replaced_lambdas == ("resize_to_same0", "resize_to_same1")
    x = np.random.default_rng(1).random((4, 16, 16, 2)).astype(np.float32)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-5)


@needs_tensorflow
def test_unknown_lambda_layers_give_a_clear_error(tmp_path):
    import keras

    from opennve.networks import load_keras_model

    inputs = keras.Input((4,))
    keras.Model(inputs, keras.layers.Lambda(lambda t: t * 2, name="double")(inputs)).save(tmp_path / "m.keras")
    with pytest.raises(ValueError, match="double.*register_lambda"):
        load_keras_model(tmp_path / "m.keras")


@needs_tensorflow
def test_weights_file_into_unet_by_name(hsnr):
    """A Keras 3 .weights.h5 of the functional U-Net into the opennve UNet built by name: the
    options are read from the weights, the result is checked against the original."""
    path, model = hsnr
    report = opennve.transfer_weights(path / "trained.weights.h5", "UNet", reference=path / "original.keras",
                                      output=path / "unet.keras")
    assert report.ok, report.text()
    assert report.layers == 2 * 4 + 4 + 2 * 4 + 3 and report.match == "in order per layer type"
    assert report.same_class == 1.0 and report.max_difference < 1e-4
    assert "layer_depth=2, filters_orig=4, kernel_size=3" in report.destination
    loaded = opennve.Network.load(path / "unet.keras")
    x = np.random.default_rng(2).random((4, 16, 16, 2)).astype(np.float32)
    np.testing.assert_allclose(loaded.model.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-4)


@needs_tensorflow
def test_model_into_model_and_back(hsnr):
    """.keras -> UNet (the source is the reference), and the opennve UNet back into the functional
    structure."""
    path, _ = hsnr
    report = opennve.transfer_weights(path / "original.keras", "UNet", output=path / "unet.keras")
    assert report.ok and report.reference == "original.keras"
    back = opennve.transfer_weights(path / "unet.keras", path / "original.keras")
    assert back.ok and back.same_class == 1.0


@needs_tensorflow
def test_untrained_weights_are_reported(tmp_path):
    """Weights with only initial values (saved before the trained weights were loaded) are
    reported, and the report is not ok."""
    network = opennve.Network.build("UNet", opennve.NetworkParameters(3, 16, 16, 2), layer_depth=2, filters_orig=4)
    network.save(tmp_path / "fresh.keras")
    report = opennve.transfer_weights(tmp_path / "fresh.keras", "UNet")
    assert not report.ok and "untrained" in report.text()


@needs_tensorflow
def test_different_networks_are_refused(hsnr):
    path, _ = hsnr
    with pytest.raises(ValueError, match="different layers"):
        opennve.transfer_weights(path / "trained.weights.h5", "UNet", options={"layer_depth": 3}, input_size=(16, 16))
    with pytest.raises(ValueError, match="different shapes"):
        opennve.transfer_weights(path / "trained.weights.h5", "UNet", options={"filters_orig": 8}, input_size=(16, 16))


@needs_tensorflow
def test_keras2_h5_and_nested_weights(hsnr):
    """A Keras 2 .h5 file is read with its layer names (paired by name); the weights of a
    subclassed model (nested) load into a model of the same classes."""
    from opennve.networks import read_weights

    path, model = hsnr
    model.save(path / "legacy.h5")
    weights = read_weights(path / "legacy.h5")
    assert weights.layer_names and weights.groups[0].name == "enc_conv0-1"
    report = opennve.transfer_weights(path / "legacy.h5", path / "original.keras")
    assert report.ok and report.match == "by name"

    unet = opennve.transfer_weights(path / "original.keras", "UNet", output=path / "unet.keras").model
    unet.save_weights(path / "nested.weights.h5")
    with pytest.raises(ValueError, match="subclassed"):
        read_weights(path / "nested.weights.h5")
    report = opennve.transfer_weights(path / "nested.weights.h5", path / "unet.keras", reference=path / "original.keras")
    assert report.ok, report.text()


@needs_tensorflow
def test_transfer_command(hsnr, capsys):
    from opennve.cli import main

    path, _ = hsnr
    code = main(["transfer", str(path / "trained.weights.h5"), "--to", "UNet", "--reference", str(path / "original.keras"),
                 "--output", str(path / "unet.keras"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["ok"] and data["same_class"] == 1.0 and (path / "unet.keras").is_file()
    assert main(["transfer", str(path / "trained.weights.h5"), "--to", "UNet", "--set", "layer_depth=3",
                 "--input", "16x16"]) == 1
    assert "different layers" in capsys.readouterr().err
