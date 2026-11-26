"""Replacements for Lambda layers of saved models.

Keras stores the function of a Lambda layer as Python bytecode. Bytecode only loads with the
Python version that saved it (and only with safe_mode=False), so a .keras file with Lambda layers
saved with Python 3.14 does not load with 3.13 or 3.15. load_keras_model therefore replaces Lambda
layers whose name matches a registered pattern with a regular layer that does the same:

    import keras, opennve

    @keras.saving.register_keras_serializable(package="mine")
    class AddOne(keras.layers.Layer):
        def call(self, x):
            return x + 1

    opennve.register_lambda(r"add_one\\d*", AddOne)   # Lambda layers named add_one, add_one2, ...

Built in: resize_to_same<n> of the HSNR U-Net (resize the first input to the height and width of
the second one, as UNet.resize_to_same).
"""

from __future__ import annotations

import re
import typing

_LAMBDAS: list[tuple[re.Pattern, typing.Any]] = []
_BUILT_IN = False


def register_lambda(pattern: str, layer) -> None:
    """Lambda layers whose name matches pattern (a regular expression for the whole name) are
    loaded as layer (a Keras layer class, created with name= and trainable=)."""
    _LAMBDAS.insert(0, (re.compile(pattern), layer))


def unregister_lambda(pattern: str) -> None:
    _LAMBDAS[:] = [(p, layer) for p, layer in _LAMBDAS if p.pattern != pattern]


def lambda_replacement(name: str):
    """The registered layer class for a Lambda layer name, or None."""
    _built_in()
    for pattern, layer in _LAMBDAS:
        if pattern.fullmatch(name):
            return layer
    return None


def _built_in() -> None:
    global _BUILT_IN
    if _BUILT_IN:
        return
    _BUILT_IN = True
    import keras
    import tensorflow as tf

    @keras.saving.register_keras_serializable(package="opennve")
    class ResizeToSame(keras.layers.Layer):
        """Resizes the first input to the height and width of the second (bilinear, as
        tf.image.resize; the HSNR U-Net's resize_to_same Lambda layers)."""

        def call(self, inputs):
            x, reference = inputs
            return tf.image.resize(x, tf.shape(reference)[1:3])

        def compute_output_shape(self, input_shapes):
            x, reference = input_shapes
            return (x[0], reference[1], reference[2], x[3])

    _LAMBDAS.append((re.compile(r"resize_to_same\d*"), ResizeToSame))


def replace_lambdas(config: dict) -> list[str]:
    """Replaces the registered Lambda layers in a model configuration (config.json of a .keras
    file, changed in place); returns the names of the replaced layers. Nested models are searched
    too."""
    replaced = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda" and isinstance(node.get("config"), dict):
                name = node["config"].get("name", "")
                layer = lambda_replacement(name)
                if layer is not None:
                    import keras

                    node["module"] = None
                    node["class_name"] = layer.__name__
                    node["registered_name"] = keras.saving.get_registered_name(layer)
                    node["config"] = {"name": name, "trainable": node["config"].get("trainable", True),
                                      "dtype": node["config"].get("dtype", "float32")}
                    replaced.append(name)
                    return
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config)
    return replaced


def lambda_names(config: dict) -> list[str]:
    """Names of all Lambda layers in a model configuration."""
    names = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda" and isinstance(node.get("config"), dict):
                names.append(node["config"].get("name", "?"))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config)
    return names
