"""Quantization aware training (QAT) of the U-Net: an int8 TensorFlow Lite model that is as
accurate as the float model.

Post-training quantization (opennve quantize without --qat) rounds a finished float network to
8 bit; for the U-Net the rounding errors add up from layer to layer. QAT instead trains the network
further with the 8 bit rounding in the forward pass (fake quantization, gradients pass straight
through), so the weights learn to work with it. The teacher is the current float network
(distillation): the quantized network learns to give the same classes as the float network on the
recordings, no labels are needed.

    student = QuantizedUNet.from_unet(unet)   # batch normalization folded into the convolutions
    student.calibrate(batches)                # start ranges of the activations
    ... train (opennve.agents.quantize.quantize_agent(..., qat=True) does this) ...
    data = student.to_tflite()                # int8 .tflite, the ranges come from the training

Layers of the quantized network:
* convolutions with the batch normalization folded in (as TensorFlow Lite does it at the export):
  the weights are rounded per output channel to int8 (symmetric, -127..127)
* an ActivationQuantizer after every convolution, activation, concatenation and on the input: its
  range comes from the calibration (min / max over up to 2048 inputs), it rounds to 256 steps
* the output convolution (1 x 1) stays float32: the classes come from small differences of large
  logits, which 8 bit would round away (costs almost nothing)
* no dropout (the network learns to match the teacher, not new data)

Needs TensorFlow (Keras 3, TensorFlow backend).
"""

from __future__ import annotations

import os
import tempfile
import typing

import keras
import numpy as np
import tensorflow as tf

from .unet import UNet

PACKAGE = "opennve"


class ActivationQuantizer(keras.layers.Layer):
    """Rounds a tensor to 8 bit (256 steps between the learned minimum and maximum).

    observe=True: only measures the range (calibration: minimum / maximum over the calibration
    inputs; the tensor is not changed). Otherwise: rounding with that range, which the TensorFlow
    Lite converter turns into the int8 scale. momentum < 1 moves the range while training (moving
    average of the batch minimum / maximum); the default 1 keeps the calibrated range, which trains
    more steadily with sparse event data (the batch ranges jump)."""

    PERCENTILES = (100.0, 99.999, 99.99, 99.9)
    SAMPLE = 200_000  # values per calibration batch for the percentiles

    def __init__(self, momentum: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.observe = False
        self._batch_ranges: list[np.ndarray] = []  # per calibration batch: (low, high) per percentile

    def build(self, input_shape):
        self.minimum = self.add_weight(name="minimum", shape=(), initializer="zeros", trainable=False)
        self.maximum = self.add_weight(name="maximum", shape=(), initializer="zeros", trainable=False)
        self.seen = self.add_weight(name="seen", shape=(), initializer="zeros", trainable=False)
        super().build(input_shape)

    def call(self, x, training=False):
        if self.observe:
            low, high = tf.reduce_min(x), tf.reduce_max(x)
            first = tf.equal(self.seen, 0.0)
            self._set(tf.where(first, low, tf.minimum(self.minimum, low)),
                      tf.where(first, high, tf.maximum(self.maximum, high)))
            self.seen.assign_add(1.0)
            if tf.executing_eagerly():  # percentiles of the batch (calibrate() runs eagerly)
                values = np.asarray(x).ravel()
                if len(values) > self.SAMPLE:
                    values = values[np.random.default_rng(len(self._batch_ranges)).integers(0, len(values), self.SAMPLE)]
                ranges = [(float(np.min(values)), float(np.max(values))) if p >= 100 else
                          (float(np.percentile(values, 100 - p)), float(np.percentile(values, p))) for p in self.PERCENTILES]
                ranges[0] = (float(low), float(high))  # exact minimum / maximum
                self._batch_ranges.append(np.array(ranges))
            return x
        if training and self.momentum < 1.0:
            m = self.momentum
            self._set(m * self.minimum + (1.0 - m) * tf.reduce_min(x), m * self.maximum + (1.0 - m) * tf.reduce_max(x))
        # the converter reads the range from these variables (constants in the exported model)
        return tf.quantization.fake_quant_with_min_max_vars(x, self.minimum, self.maximum, num_bits=8, narrow_range=False)

    def use_percentile(self, percentile: float) -> None:
        """Range from the calibration: minimum / maximum (100) or the mean over the calibration
        batches of that percentile (cuts off rare outliers, finer steps for the usual values)."""
        if not self._batch_ranges:
            return
        ranges = np.stack(self._batch_ranges)  # batch, percentile, (low, high)
        index = self.PERCENTILES.index(percentile)
        if percentile >= 100:
            low, high = ranges[:, index, 0].min(), ranges[:, index, 1].max()
        else:
            low, high = ranges[:, index, 0].mean(), ranges[:, index, 1].mean()
        self._set(tf.constant(low, tf.float32), tf.constant(high, tf.float32))

    def _set(self, low, high):
        low = tf.minimum(low, 0.0)  # zero must be exact (padding, ReLU)
        self.minimum.assign(low)
        self.maximum.assign(tf.maximum(tf.maximum(high, 0.0), low + 1e-3))


class QuantizedConv(keras.layers.Layer):
    """Convolution (or transposed convolution) with int8-rounded weights and the batch normalization
    folded in. relu: fused ReLU (as dec_out1 / dec_out2 of the U-Net)."""

    def __init__(self, kernel: np.ndarray, bias: np.ndarray, strides: int = 1, transpose: bool = False,
                 relu: bool = False, quantize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.quantize = quantize
        self.transpose = transpose
        self.strides = strides
        self.relu = relu
        self._kernel_value = np.asarray(kernel, np.float32)
        self._bias_value = np.asarray(bias, np.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=self._kernel_value.shape,
                                      initializer=keras.initializers.Constant(self._kernel_value))
        self.bias = self.add_weight(name="bias", shape=self._bias_value.shape,
                                    initializer=keras.initializers.Constant(self._bias_value))
        # ranges of the weights: per output channel for convolutions; one range for transposed
        # convolutions (the converter quantizes those per tensor)
        size = 1 if self.transpose else self._kernel_value.shape[3]
        limit = self._limit(self._kernel_value)
        self.high = self.add_weight(name="high", shape=(size,), trainable=False,
                                    initializer=keras.initializers.Constant(limit))
        self.low = self.add_weight(name="low", shape=(size,), trainable=False,
                                   initializer=keras.initializers.Constant(-limit))
        super().build(input_shape)

    def _limit(self, kernel):
        if isinstance(kernel, np.ndarray):
            limit = np.abs(kernel).max() if self.transpose else np.abs(kernel).max(axis=(0, 1, 2))
            return np.maximum(np.atleast_1d(limit), 1e-8).astype(np.float32)
        limit = tf.reshape(tf.reduce_max(tf.abs(kernel)), (1,)) if self.transpose else \
            tf.reduce_max(tf.abs(kernel), axis=[0, 1, 2])
        return tf.maximum(limit, 1e-8)

    def refresh(self):
        """Weight ranges from the current weights (done in every training step)."""
        limit = self._limit(self.kernel)
        self.high.assign(limit)
        self.low.assign(-limit)

    def quantized_kernel(self):
        """The weights rounded to int8 (symmetric, -127..127): per output channel, or per tensor for
        transposed convolutions. The converter reads the ranges from the variables low / high."""
        if self.transpose:
            return tf.quantization.fake_quant_with_min_max_vars(self.kernel, self.low[0], self.high[0], num_bits=8,
                                                                narrow_range=True)
        return tf.quantization.fake_quant_with_min_max_vars_per_channel(self.kernel, self.low, self.high, num_bits=8,
                                                                         narrow_range=True)

    def call(self, x, training=False):
        if training:
            self.refresh()
        kernel = self.quantized_kernel() if self.quantize else self.kernel
        if self.transpose:
            shape = tf.shape(x)
            out = tf.stack([shape[0], shape[1] * self.strides, shape[2] * self.strides, self.kernel.shape[2]])
            y = tf.nn.conv2d_transpose(x, kernel, out, strides=self.strides, padding="SAME")
        else:
            y = tf.nn.conv2d(x, kernel, strides=self.strides, padding="SAME")
        y = tf.nn.bias_add(y, self.bias)
        return tf.nn.relu(y) if self.relu else y


def _fold(conv, batch_norm, transpose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Kernel and bias of a convolution with the batch normalization after it folded in."""
    kernel = np.asarray(conv.kernel.numpy(), np.float64)
    bias = np.asarray(conv.bias.numpy(), np.float64) if conv.use_bias else np.zeros(kernel.shape[2 if transpose else 3])
    if batch_norm is None:
        return kernel.astype(np.float32), bias.astype(np.float32)
    gamma = np.asarray(batch_norm.gamma.numpy(), np.float64) if batch_norm.scale else 1.0
    beta = np.asarray(batch_norm.beta.numpy(), np.float64) if batch_norm.center else 0.0
    mean = np.asarray(batch_norm.moving_mean.numpy(), np.float64)
    variance = np.asarray(batch_norm.moving_variance.numpy(), np.float64)
    scale = gamma / np.sqrt(variance + batch_norm.epsilon)
    if transpose:
        kernel = kernel * scale[np.newaxis, np.newaxis, :, np.newaxis]
    else:
        kernel = kernel * scale
    bias = (bias - mean) * scale + beta
    return kernel.astype(np.float32), bias.astype(np.float32)


def _activation(layer) -> float:
    return float(layer.negative_slope)


class _Stage(keras.layers.Layer):
    """Folded convolution -> range -> LeakyReLU -> range."""

    def __init__(self, conv: QuantizedConv, negative_slope: float, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.pre = ActivationQuantizer()
        self.negative_slope = negative_slope
        self.post = ActivationQuantizer()

    def call(self, x, training=False):
        x = self.pre(self.conv(x, training=training), training=training)
        x = tf.nn.leaky_relu(x, alpha=self.negative_slope)
        return self.post(x, training=training)


class QuantizedUNet(keras.Model):
    """The U-Net with int8 rounding (quantization aware training). Built from a trained UNet with
    from_unet(); to_tflite() gives the int8 TensorFlow Lite model."""

    def __init__(self, unet: UNet, float_head: bool = True, output_activation: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(unet, UNet):
            raise TypeError("quantization aware training supports the opennve U-Net (opennve.networks.UNet)")
        self.input_quantizer = ActivationQuantizer(name="input_range")

        def stage(conv, bn, act, transpose=False, strides=1, name=None):
            kernel, bias = _fold(conv, bn, transpose)
            return _Stage(QuantizedConv(kernel, bias, strides=strides, transpose=transpose), _activation(act),
                          name=name)

        self.encoder_a, self.encoder_b = [], []
        for index, block in enumerate(unet.encoder_blocks):
            self.encoder_a.append(stage(block.conv1, block.bn1, block.act1, name=f"enc{index}_a"))
            self.encoder_b.append(stage(block.conv2, block.bn2, block.act2, name=f"enc{index}_b"))
        self.pool_size = unet.pools[0].pool_size if unet.pools else (2, 2)
        block = unet.bottleneck
        self.middle_a = stage(block.conv1, block.bn1, block.act1, name="mid_a")
        self.middle_b = stage(block.conv2, block.bn2, block.act2, name="mid_b")
        self.decoder_a, self.decoder_b = [], []
        self.concat_quantizers = []
        for index, block in enumerate(unet.decoder_blocks):
            self.decoder_a.append(stage(block.conv1, block.bn1, block.act1, name=f"dec{index}_a"))
            self.decoder_b.append(stage(block.deconv, block.bn2, block.act2, transpose=True, strides=block.strides,
                                        name=f"dec{index}_b"))
            self.concat_quantizers.append(ActivationQuantizer(name=f"concat{index}_range"))

        def plain(conv, relu, name, quantize=True):
            kernel, bias = _fold(conv, None)
            return QuantizedConv(kernel, bias, relu=relu, name=name, quantize=quantize)

        activation = unet.dec_out1.activation.__name__
        self.out1 = plain(unet.dec_out1, activation == "relu", "out1")
        self.out1_range = ActivationQuantizer(name="out1_range")
        self.out2 = plain(unet.dec_out2, unet.dec_out2.activation.__name__ == "relu", "out2")
        self.out2_range = ActivationQuantizer(name="out2_range")
        self.final_activation = unet.output_layer.activation.__name__
        if output_activation and self.final_activation == "linear":
            # the saved model applies it after the U-Net (for instance a softmax layer around it)
            self.final_activation = output_activation
        if self.final_activation not in ("linear", "sigmoid", "softmax"):
            raise ValueError(f"final activation {self.final_activation} is not supported for quantization aware training")
        # the output convolution (1 x 1, few channels) stays float: the classes are decided by small
        # differences of large logits, which 8 bit would round away; it costs almost nothing
        self.float_head = float_head
        self.head = plain(unet.output_layer, False, "head", quantize=not float_head)
        self.head_range = None if float_head else ActivationQuantizer(name="logits_range")
        self.output_range = ActivationQuantizer(name="output_range") if self.final_activation != "linear" and not float_head else None

    @classmethod
    def from_unet(cls, unet: UNet, input_shape: typing.Sequence[int], float_head: bool = True,
                  output_activation: str | None = None) -> QuantizedUNet:
        """Quantized copy of a trained U-Net (input_shape: height, width, channels). float_head:
        the output convolution stays float32 (more accurate classes, almost no extra time).
        output_activation: "softmax" / "sigmoid" if the saved model applies it after the U-Net."""
        model = cls(unet, float_head=float_head, output_activation=output_activation)
        model(tf.zeros((1, *input_shape), tf.float32))  # build
        return model

    def call(self, x, training=False):
        x = self.input_quantizer(x, training=training)
        skips = []
        for first, second in zip(self.encoder_a, self.encoder_b):
            x = second(first(x, training=training), training=training)
            skips.append(x)
            x = tf.nn.max_pool2d(x, self.pool_size, self.pool_size, "VALID")
        x = self.middle_b(self.middle_a(x, training=training), training=training)
        for index, (first, second, concat_range) in enumerate(zip(self.decoder_a, self.decoder_b, self.concat_quantizers)):
            x = second(first(x, training=training), training=training)
            shortcut = skips[-(index + 1)]
            x = tf.image.resize(x, tf.shape(shortcut)[1:3])
            x = concat_range(tf.concat([x, shortcut], axis=-1), training=training)
        x = self.out1_range(self.out1(x, training=training), training=training)
        x = self.out2_range(self.out2(x, training=training), training=training)
        x = self.head(x, training=training)
        if self.head_range is not None:
            x = self.head_range(x, training=training)
        if self.final_activation == "sigmoid":
            x = tf.nn.sigmoid(x)
        elif self.final_activation == "softmax":
            x = tf.nn.softmax(x, axis=-1)
        if self.output_range is not None:
            x = self.output_range(x, training=training)
        return x

    def quantizers(self) -> list[ActivationQuantizer]:
        return [layer for layer in self._flatten_layers(include_self=False) if isinstance(layer, ActivationQuantizer)]

    def calibrate(self, batches: typing.Iterable[np.ndarray]) -> int:
        """Start ranges of all activations: minimum / maximum over the batches (no rounding); the
        percentiles are kept for use_percentile()."""
        quantizers = self.quantizers()
        for quantizer in quantizers:
            quantizer.observe = True
            quantizer.seen.assign(0.0)
            quantizer._batch_ranges = []
        count = 0
        try:
            for batch in batches:
                self(tf.convert_to_tensor(batch, tf.float32), training=False)
                count += len(batch)
        finally:
            for quantizer in quantizers:
                quantizer.observe = False
        return count

    def use_percentile(self, percentile: float) -> None:
        """Ranges of all activations from the calibration at this percentile (see
        ActivationQuantizer.use_percentile)."""
        for quantizer in self.quantizers():
            quantizer.use_percentile(percentile)

    def to_tflite(self, input_shape: typing.Sequence[int], int8_io: bool = False) -> bytes:
        """The int8 TensorFlow Lite model (ranges from the training, no representative data)."""
        for quantizer in self.quantizers():
            quantizer.observe = False
        for layer in self._flatten_layers(include_self=False):
            if isinstance(layer, QuantizedConv):
                layer.refresh()
        signature = [tf.TensorSpec((None, *input_shape), tf.float32)]
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "saved_model")
            self.export(path, format="tf_saved_model", input_signature=signature, verbose=False)
            converter = tf.lite.TFLiteConverter.from_saved_model(path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if int8_io:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            return converter.convert()


# ---------------------------------------------------------------------------- training
def _probabilities(output, final_activation: str, activation: str):
    """(probabilities, log probabilities or None) of a network output; activation: the agent's
    prediction activation ("softmax", "sigmoid", other: no probabilities, the values are compared)."""
    if final_activation == "linear":
        if activation == "softmax":
            return tf.nn.softmax(output, axis=-1), tf.nn.log_softmax(output, axis=-1)
        if activation == "sigmoid":
            return tf.nn.sigmoid(output), None
        return output, None
    return output, tf.math.log(tf.clip_by_value(output, 1e-7, 1.0)) if final_activation == "softmax" else None


def distillation_loss(teacher_output, student_output, final_activation: str, activation: str,
                      target_classes: typing.Sequence[int] = (), target_weight: float = 1.0):
    """How far the quantized network is from the float network: cross entropy with the float
    network's probabilities (softmax), binary cross entropy (sigmoid) or the squared difference.
    Pixels the float network gives one of the target classes count target_weight times (they are
    rare, for instance insects, and matter most)."""
    target, _ = _probabilities(teacher_output, final_activation, activation)
    if activation == "softmax":
        _, log_probabilities = _probabilities(student_output, final_activation, activation)
        per_pixel = -tf.reduce_sum(target * log_probabilities, axis=-1)
        if target_classes and target_weight != 1.0:
            classes = tf.argmax(target, axis=-1, output_type=tf.int32)
            important = tf.reduce_any(tf.equal(classes[..., tf.newaxis], tf.constant(list(target_classes), tf.int32)), -1)
            weights = tf.where(important, target_weight, 1.0)
            return tf.reduce_sum(per_pixel * weights) / tf.reduce_sum(weights)
        return tf.reduce_mean(per_pixel)
    if activation == "sigmoid":
        if final_activation == "linear":
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=student_output))
        probabilities = tf.clip_by_value(student_output, 1e-7, 1.0 - 1e-7)
        return -tf.reduce_mean(target * tf.math.log(probabilities) + (1.0 - target) * tf.math.log(1.0 - probabilities))
    return tf.reduce_mean(tf.square(teacher_output - student_output))


def classes_of(output: np.ndarray, activation: str) -> np.ndarray:
    """Class per pixel: arg max (softmax / several classes), else the sign / 0.5 threshold."""
    output = np.asarray(output, np.float32)
    if output.shape[-1] > 1:
        return output.argmax(axis=-1)
    threshold = 0.5 if activation == "sigmoid" and output.min() >= 0 else 0.0
    return (output[..., 0] > threshold).astype(np.int64)


def agreement(reference: np.ndarray, candidate: np.ndarray, target_classes: typing.Sequence[int] = ()) -> dict:
    """Share of pixels with the same class; IoU and recall of every target class."""
    result = {"agreement": float(np.mean(reference == candidate)) if reference.size else None, "classes": {}}
    for target in target_classes:
        a, b = reference == target, candidate == target
        union = np.logical_or(a, b).sum()
        both = np.logical_and(a, b).sum()
        result["classes"][int(target)] = {
            "pixels": int(a.sum()),
            "iou": float(both / union) if union else None,
            "recall": float(both / a.sum()) if a.sum() else None,
        }
    return result


def train_qat(
    student: QuantizedUNet,
    teacher: keras.Model,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    activation: str = "softmax",
    target_classes: typing.Sequence[int] = (),
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    target_weight: float = 10.0,
    seed: int = 0,
    progress: typing.Callable[[str], None] | None = None,
) -> list[dict]:
    """Trains the quantized network to give the float network's (teacher) outputs on the inputs
    (distillation, no labels). After every epoch the classes of both are compared on the validation
    inputs; the best epoch's weights are kept (by the IoU of the target classes, else the share of
    equal classes; also the starting point: training never makes it worse). Pixels of the target
    classes count target_weight times in the loss. Returns one entry per epoch (0: before the training)."""
    say = progress or (lambda message: None)
    rng = np.random.default_rng(seed)
    final_activation = student.final_activation
    teacher_classes = np.concatenate([
        classes_of(teacher.predict_on_batch(validation[i:i + batch_size].astype(np.float32)), activation)
        for i in range(0, len(validation), batch_size)
    ]) if len(validation) else np.zeros((0,))

    def evaluate() -> dict:
        if not len(validation):
            return {"agreement": None, "classes": {}}
        predicted = np.concatenate([
            classes_of(student(validation[i:i + batch_size].astype(np.float32), training=False).numpy(), activation)
            for i in range(0, len(validation), batch_size)
        ])
        return agreement(teacher_classes, predicted, target_classes)

    steps = max(1, epochs * int(np.ceil(len(train) / batch_size)))
    schedule = keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps=steps, alpha=0.05)
    optimizer = keras.optimizers.Adam(learning_rate=schedule)
    variables = student.trainable_variables

    @tf.function(reduce_retracing=True)
    def step(batch):
        target = teacher(batch, training=False)
        with tf.GradientTape() as tape:
            output = student(batch, training=True)
            loss = distillation_loss(target, output, final_activation, activation, target_classes, target_weight)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    # ranges of the activations: minimum / maximum or a percentile of the calibration, whichever
    # keeps the classes best (rare outliers can make the 8 bit steps too coarse)
    percentile = 100.0
    if len(validation) and any(q._batch_ranges for q in student.quantizers()):
        best_start = None
        for candidate in ActivationQuantizer.PERCENTILES:
            student.use_percentile(candidate)
            entry = evaluate()
            label = "minimum / maximum" if candidate >= 100 else f"percentile {candidate:g}"
            say(f"ranges {label}: {_describe(entry)}")
            if best_start is None or _score(entry) > best_start[0]:
                best_start = (_score(entry), candidate)
        percentile = best_start[1]
        student.use_percentile(percentile)
    history = [{"epoch": 0, "loss": None, "percentile": percentile, **evaluate()}]
    say(f"before the training: {_describe(history[0])}")
    best = (_score(history[0]), [v.numpy() for v in student.weights])
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = train[np.sort(order[start:start + batch_size])].astype(np.float32)
            losses.append(float(step(tf.convert_to_tensor(batch))))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, **evaluate()}
        history.append(entry)
        say(f"epoch {epoch}/{epochs}: loss {entry['loss']:.4f}, {_describe(entry)}")
        if _score(entry) >= best[0]:
            best = (_score(entry), [v.numpy() for v in student.weights])
    for variable, value in zip(student.weights, best[1]):
        variable.assign(value)
    return history


def _score(entry: dict) -> float:
    """Quality of an epoch: mean IoU of the target classes (they matter most), else the share of
    pixels with the same class."""
    ious = [values["iou"] for values in entry.get("classes", {}).values() if values["iou"] is not None]
    if ious:
        return float(np.mean(ious)) + 1e-3 * (entry["agreement"] or 0.0)
    return entry["agreement"] or 0.0


def _describe(entry: dict) -> str:
    if entry.get("agreement") is None:
        return "no validation inputs"
    text = f"{entry['agreement'] * 100:.3f} % same class as the float network"
    for target, values in entry["classes"].items():
        if values["iou"] is not None:
            text += f", class {target} IoU {values['iou'] * 100:.1f} %"
    return text
