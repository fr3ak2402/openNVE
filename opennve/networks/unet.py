"""U-Net for event frames / tiles (semantic segmentation: one class per pixel).

Registered as the architecture "UNet":

    network = opennve.Network.build("UNet", opennve.NetworkParameters(number_of_classes=3))

The layers are Keras serializable. Models saved with the same classes registered under
another package (for instance "Custom>UNet") load as well, see registry.custom_objects().
Needs TensorFlow (Keras 3).
"""

from __future__ import annotations

import keras
import tensorflow as tf

from .registry import NetworkParameters, register_custom_objects, register_network

PACKAGE = "opennve"


@keras.saving.register_keras_serializable(package=PACKAGE)
class UNetConvBlock(keras.layers.Layer):
    """Two convolutions with batch normalization and LeakyReLU (encoder and bottleneck)."""

    def __init__(self, filters, kernel_size=4, batch_norm=True, dropout=False, name_prefix="enc", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.name_prefix = name_prefix

        self.conv1 = keras.layers.Conv2D(
            filters, kernel_size=kernel_size, activation=None, kernel_initializer="he_normal", padding="same"
        )
        self.bn1 = keras.layers.BatchNormalization() if batch_norm else None
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.3)
        self.drop = keras.layers.Dropout(0.2) if dropout else None

        self.conv2 = keras.layers.Conv2D(
            filters, kernel_size=kernel_size, activation=None, kernel_initializer="he_normal", padding="same"
        )
        self.bn2 = keras.layers.BatchNormalization() if batch_norm else None
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.3)

    def call(self, x, training=False):
        x = self.conv1(x)
        if self.bn1:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.drop:
            x = self.drop(x, training=training)
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x, training=training)
        return self.act2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "batch_norm": self.batch_norm,
                "dropout": self.dropout,
                "name_prefix": self.name_prefix,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package=PACKAGE)
class UNetDeconvBlock(keras.layers.Layer):
    """Convolution and transposed convolution (upsampling) with batch normalization (decoder)."""

    def __init__(self, filters, kernel_size=4, strides=2, batch_norm=True, dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.conv1 = keras.layers.Conv2D(
            filters, kernel_size=kernel_size, activation=None, kernel_initializer="he_normal", padding="same"
        )
        self.bn1 = keras.layers.BatchNormalization() if batch_norm else None
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.3)
        self.drop = keras.layers.Dropout(0.2) if dropout else None

        self.deconv = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=None,
            kernel_initializer="he_normal",
            padding="same",
        )
        self.bn2 = keras.layers.BatchNormalization() if batch_norm else None
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.3)

    def call(self, x, training=False):
        x = self.conv1(x)
        if self.bn1:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.drop:
            x = self.drop(x, training=training)
        x = self.deconv(x)
        if self.bn2:
            x = self.bn2(x, training=training)
        return self.act2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "batch_norm": self.batch_norm,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package=PACKAGE)
class UNet(keras.Model):
    """U-Net: layer_depth encoder blocks (filters doubled per level, at most 512), a bottleneck,
    layer_depth decoder blocks with skip connections and a 1 x 1 output convolution with
    out_channels channels (one per class)."""

    def __init__(
        self,
        out_channels=3,
        layer_depth=4,
        filters_orig=32,
        kernel_size=4,
        batch_norm=True,
        final_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_depth = layer_depth
        self.filters_orig = filters_orig

        self.encoder_blocks = []
        self.pools = []
        filters = filters_orig
        for _ in range(layer_depth):
            self.encoder_blocks.append(
                UNetConvBlock(filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=False)
            )
            self.pools.append(keras.layers.MaxPooling2D(pool_size=2))
            filters = min(filters * 2, 512)

        self.bottleneck = UNetConvBlock(filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=False)

        self.decoder_blocks = []
        for i in range(layer_depth):
            filters = max(filters // 2, filters_orig)
            self.decoder_blocks.append(
                UNetDeconvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=2,
                    batch_norm=batch_norm,
                    dropout=i < (layer_depth - 2),
                )
            )

        self.dec_out1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")
        self.drop_out1 = keras.layers.Dropout(0.3)
        self.dec_out2 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")
        self.drop_out2 = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, activation=final_activation, padding="same"
        )

    def resize_to_same(self, x, shortcut):
        return tf.image.resize(x, tf.shape(shortcut)[1:3])

    def call(self, x, training=False):
        outputs_for_skip = []
        for encoder_block, pool in zip(self.encoder_blocks, self.pools):
            conv_block = encoder_block(x, training=training)
            outputs_for_skip.append(conv_block)
            x = pool(conv_block)

        x = self.bottleneck(x, training=training)

        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, training=training)
            shortcut = outputs_for_skip[-(i + 1)]
            x = self.resize_to_same(x, shortcut)
            x = tf.concat([x, shortcut], axis=-1)

        x = self.dec_out1(x)
        x = self.drop_out1(x, training=training)
        x = self.dec_out2(x)
        x = self.drop_out2(x, training=training)
        return self.output_layer(x)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.output_layer.filters,
                "layer_depth": self.layer_depth,
                "filters_orig": self.filters_orig,
                "kernel_size": self.encoder_blocks[0].kernel_size,
                "batch_norm": self.encoder_blocks[0].batch_norm,
                "final_activation": self.output_layer.activation.__name__,
            }
        )
        return config


# models saved with these classes registered as "Custom>..." (keras default) load as well
register_custom_objects(
    {"UNet": UNet, "UNetConvBlock": UNetConvBlock, "UNetDeconvBlock": UNetDeconvBlock}, package="Custom"
)
register_custom_objects(
    {"UNet": UNet, "UNetConvBlock": UNetConvBlock, "UNetDeconvBlock": UNetDeconvBlock}, package=PACKAGE
)


@register_network("UNet")
def build_unet(
    parameters: NetworkParameters,
    layer_depth: int = 4,
    filters_orig: int = 16,
    kernel_size: int = 3,
    batch_norm: bool = True,
    final_activation: str = "linear",
) -> keras.Model:
    """UNet for tiles of parameters.tile_height x tile_width x channels with
    parameters.number_of_classes output channels (logits by default: final_activation "linear")."""
    inputs = keras.Input(shape=parameters.input_shape)
    unet = UNet(
        out_channels=parameters.number_of_classes,
        layer_depth=layer_depth,
        filters_orig=filters_orig,
        kernel_size=kernel_size,
        batch_norm=batch_norm,
        final_activation=final_activation,
        name="u_net",
    )
    return keras.Model(inputs=inputs, outputs=unet(inputs))


def _find_unet(model, name: str) -> UNet:
    if isinstance(model, UNet):
        return model
    try:
        return model.get_layer(name)
    except ValueError:
        candidates = [layer for layer in model.layers if isinstance(layer, UNet)]
        if len(candidates) != 1:
            raise ValueError(f"the destination model has no UNet layer {name!r}") from None
        return candidates[0]


def transfer_unet_weights(source_model, destination_model, layer_depth: int = 4, batch_norm: bool = True, unet_layer: str = "u_net"):
    """Copies the weights of a functional U-Net with named layers (enc_conv{i}-1, enc_bn{i}-1, ...,
    enc_convbtleneck-1, dec_conv{i}-1, dec_conv{i}-2, dec_out1, dec_out2, dec_output; for instance
    the HSNR U-Net) into a model built with the UNet class (layer unet_layer)."""
    unet = _find_unet(destination_model, unet_layer)

    def copy(target, name):
        target.set_weights(source_model.get_layer(name).get_weights())

    for i in range(layer_depth):
        block = unet.encoder_blocks[i]
        copy(block.conv1, f"enc_conv{i}-1")
        if batch_norm:
            copy(block.bn1, f"enc_bn{i}-1")
        copy(block.conv2, f"enc_conv{i}-2")
        if batch_norm:
            copy(block.bn2, f"enc_bn{i}-2")

    copy(unet.bottleneck.conv1, "enc_convbtleneck-1")
    if batch_norm:
        copy(unet.bottleneck.bn1, "enc_bnbtleneck-1")
    copy(unet.bottleneck.conv2, "enc_convbtleneck-2")
    if batch_norm:
        copy(unet.bottleneck.bn2, "enc_bnbtleneck-2")

    for i in range(layer_depth):
        block = unet.decoder_blocks[i]
        copy(block.conv1, f"dec_conv{i}-1")
        if batch_norm:
            copy(block.bn1, f"dec_bn{i}-1")
        copy(block.deconv, f"dec_conv{i}-2")
        if batch_norm:
            copy(block.bn2, f"dec_bn{i}-2")

    copy(unet.dec_out1, "dec_out1")
    copy(unet.dec_out2, "dec_out2")
    copy(unet.output_layer, "dec_output")
