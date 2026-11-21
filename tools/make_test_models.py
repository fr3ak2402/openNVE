"""Creates the small TensorFlow / TensorFlow Lite models used by tests/test_agents.py.

    python tools/make_test_models.py tests/data

Needs TensorFlow (Keras 3). The models have fixed weights, so their outputs can be computed
with numpy in the tests:

    agent_tiles.*:  input (batch, 16, 16, 2) -> mean per channel -> Dense(2) -> softmax
    agent_frame.tflite: input (1, 24, 40, 1), fixed batch size 1 -> mean -> Dense(3) (logits)
    agent_unet.tflite: small U-Net (batch, 16, 16, 2) -> (batch, 16, 16, 3) logits, with
        Conv2DTranspose (output shape computed at run time) and a dynamic batch size: tests the
        static shapes for GPU / NPU (opennve.agents.static_model)
"""

import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

TILE_WEIGHTS = np.array([[4.0, -2.0], [-3.0, 5.0]], dtype=np.float32)  # rows: channel 0, channel 1
TILE_BIAS = np.array([0.1, -0.2], dtype=np.float32)
FRAME_WEIGHTS = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
FRAME_BIAS = np.array([0.0, 0.3, -0.1], dtype=np.float32)


def tile_model() -> keras.Model:
    inputs = keras.Input((16, 16, 2), name="tiles")
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    dense = keras.layers.Dense(2, activation="softmax", name="classes")
    outputs = dense(x)
    model = keras.Model(inputs, outputs)
    dense.set_weights([TILE_WEIGHTS, TILE_BIAS])
    return model


def frame_model() -> keras.Model:
    inputs = keras.Input((24, 40, 1), batch_size=1, name="frame")
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    dense = keras.layers.Dense(3, name="logits")
    outputs = dense(x)
    model = keras.Model(inputs, outputs)
    dense.set_weights([FRAME_WEIGHTS, FRAME_BIAS])
    return model


def unet_model() -> keras.Model:
    keras.utils.set_random_seed(7)
    inputs = keras.Input((16, 16, 2), name="tiles")
    down = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(down)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(4, 2, strides=2, padding="same")(x)
    x = keras.layers.Concatenate()([x, down])
    outputs = keras.layers.Conv2D(3, 1, name="logits")(x)
    return keras.Model(inputs, outputs)


def write_unet(out: pathlib.Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(unet_model())
    (out / "agent_unet.tflite").write_bytes(converter.convert())


def main(directory: str):
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    model = tile_model()
    model.save(out / "agent_tiles.keras")
    model.export(out / "agent_tiles_savedmodel", format="tf_saved_model", verbose=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    (out / "agent_tiles.tflite").write_bytes(converter.convert())

    def representative():
        rng = np.random.default_rng(1)
        for _ in range(50):
            yield [rng.random((1, 16, 16, 2), dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    (out / "agent_tiles_int8.tflite").write_bytes(converter.convert())

    model = frame_model()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # fixed batch size 1 (keras.Input batch_size)
    (out / "agent_frame.tflite").write_bytes(converter.convert())
    write_unet(out)
    for path in sorted(out.glob("agent_*")):
        print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data")
