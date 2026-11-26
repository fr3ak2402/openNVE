# Neural networks

The `opennve.networks` module builds, loads, saves and converts the neural networks that [agents](agents.md) run. It contains:

- a **registry of architectures** (built in: `UNet`), extendable with your own architectures,
- `Network`: a Keras model with building, loading (including custom layers), saving, weights, `predict` and TensorFlow Lite export with quantization,
- `NeuralEngine`: a collection of named networks, for instance to transfer weights from one network to another,
- `representative_dataset`: sample tiles from frames, RAW files or cameras for int8 quantization.

Networks need TensorFlow (Keras 3). For Python 3.14 this is currently `pip install tf-nightly`, see [Agents – Installation](agents.md#installation). TensorFlow is imported only when a network is built or loaded, `import opennve` works without it.

## Quick start

```python
import opennve

parameters = opennve.NetworkParameters(number_of_classes=3, tile_height=80, tile_width=80, channels=2)

# build, train (with Keras), save
network = opennve.Network.build("UNet", parameters, name="unet")
network.summary(expand_nested=True)
network.model.compile(...)          # network.model is the Keras model
network.model.fit(...)
network.save("models/unet.keras")

# load and use
network = opennve.Network.load("models/unet.keras")
outputs = network.predict(frame)    # tiles of a frame, padded to 80 x 80 -> (tiles, 80, 80, 3)

# TensorFlow Lite for the agents (LiteRT)
tiles80 = opennve.load_frame_config("frames.json", "tiles80")
with opennve.open_raw("recording.raw") as raw:
    dataset = opennve.representative_dataset((raw, tiles80), input_shape=parameters.input_shape)
network.to_tflite("models/unet_int8.tflite", "int8", representative_dataset=dataset)
```

## `NetworkParameters`

| Field | Default | Description |
|---|---|---|
| `number_of_classes` | – | Output channels (classes), for instance 3: background, noise, insect |
| `tile_height`, `tile_width` | `80` | Input size, usually the tile size of the [frame configuration](frames.md) |
| `channels` | `2` | Input channels: 2 for both polarities (frame tensors are height × width × 2) |

`NetworkParameters.from_dict(...)` also accepts the upper-case names `NUMBER_OF_CLASSES`, `TILE_HEIGHT`, `TILE_WIDTH`, `CHANNELS`. `parameters.input_shape` is `(tile_height, tile_width, channels)`.

## Architectures

`opennve.list_networks()` lists the registered architectures. Built in:

### `UNet`

A U-Net for semantic segmentation (one class per pixel): `layer_depth` encoder blocks (two convolutions with batch normalization and LeakyReLU, max pooling; the number of filters doubles per level, at most 512), a bottleneck, `layer_depth` decoder blocks (convolution and transposed convolution, with dropout in the first `layer_depth - 2` blocks) with skip connections, two output convolutions and a 1 × 1 convolution with one channel per class.

```python
network = opennve.Network.build(
    "UNet", parameters,
    layer_depth=4, filters_orig=16, kernel_size=3, batch_norm=True, final_activation="linear",  # defaults
)
```

`final_activation="linear"` returns logits (for losses such as `SparseCategoricalCrossentropy(from_logits=True)`); the agent then applies the softmax (`"activation": "softmax"`, see [segmentation](agents.md#segmentation)).

The layers `UNet`, `UNetConvBlock` and `UNetDeconvBlock` are in `opennve.networks.unet`. They are Keras serializable; models saved with these classes from your own project (registered as `Custom>UNet`, ...) load with openNVE without further changes.

### Adding an architecture

```python
import keras
import opennve

@opennve.register_network("TinyNet")
def build_tiny_net(parameters: opennve.NetworkParameters, filters: int = 8):
    inputs = keras.Input(parameters.input_shape)
    x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    outputs = keras.layers.Conv2D(parameters.number_of_classes, 1)(x)
    return keras.Model(inputs, outputs)

network = opennve.Network.build("TinyNet", parameters, filters=16)
```

Architectures from other code (for instance the HSNR U-Net) are registered the same way:

```python
from resources.hsnr.unet import unet as hsnr_unet

@opennve.register_network("HSNR_UNet")
def build_hsnr_unet(parameters, **options):
    inputs = keras.Input(parameters.input_shape)
    outputs = hsnr_unet(inputs, kernel_size=3, batch_norm=True, out_channels=parameters.number_of_classes,
                        layer_depth=4, filters_orig=16, final_activation="linear")
    return keras.Model(inputs, outputs)
```

### Custom layers

Keras needs the classes of custom layers to load a saved model. Register them once; `Network.load` and the agents pass them to `keras.models.load_model` automatically:

```python
opennve.register_custom_objects({"MyLayer": MyLayer}, package="Custom")
```

`package` is the package the class was registered with in `@keras.saving.register_keras_serializable(package=...)` (`"Custom"` by default). Models with `Lambda` layers need `safe_mode=False` (`Network.load(path, safe_mode=False)`, or `"safe_mode": false` in the agent configuration).

## `Network`

| Method / attribute | Description |
|---|---|
| `Network.build(architecture, parameters, name=None, **options)` | New network of a registered architecture |
| `Network.load(path, name=None, safe_mode=True, custom_objects=None, compile=False)` | Loads a `.keras` / `.h5` model |
| `Network(model, name=None)` | Wraps an existing Keras model |
| `model` | The Keras model (for `compile`, `fit`, ...) |
| `save(path)` | Saves the model (`.keras`) |
| `load_weights(path)` / `save_weights(path)` | Weights only (`.weights.h5`) |
| `summary(expand_nested=False)` | Keras summary |
| `predict(data, batch_size=None)` | `model.predict` on a batch, a list of tiles or a frame (tiles are padded to the input size) |
| `input_shape`, `output_shape` | Shapes including the batch dimension |
| `get_weights()` / `set_weights(weights)` | Weight arrays |
| `compare_weights(other)` | `WeightComparison` (see below) |
| `transfer_weights_from(source, method="hsnr_unet", **options)` | Copies weights from another network |
| `to_tflite(path, quantization="none", representative_dataset=None)` | TensorFlow Lite export |
| `agent(name=None, **settings)` | [Agent](agents.md) that runs this network directly |

## TensorFlow Lite export and quantization

```python
network.to_tflite("unet.tflite")                                    # float32
network.to_tflite("unet_float16.tflite", "float16")
network.to_tflite("unet_int8.tflite", "int8", representative_dataset=dataset)
```

| `quantization` | Result |
|---|---|
| `none` | float32 model without optimization |
| `dynamic` | Dynamic range quantization: int8 weights, float32 inputs and outputs |
| `float32` | Default optimization with float32 types. TensorFlow treats this like `dynamic` (int8 weights) |
| `float16` | float16 weights (half the size, nearly the same results) |
| `float8_e4m3fn`, `float8_e5m2` | float8 weights (experimental TensorFlow types) |
| `int8` | Full integer model: int8 weights, activations, inputs and outputs. Needs a representative dataset. Fastest on CPUs and the Raspberry Pi |

The agents quantize the inputs and dequantize the outputs of int8 models automatically.

### Representative dataset

int8 quantization needs example inputs to measure the value ranges. `opennve.representative_dataset` takes them from real data:

```python
dataset = opennve.representative_dataset((raw, tiles80), input_shape=(80, 80, 2), max_samples=200, min_events=1)
dataset = opennve.representative_dataset(list_of_frames, input_shape=(80, 80, 2))
dataset = opennve.representative_dataset(tiles)              # tiles or arrays
```

- `(source, frame_config)`: frames of a camera or RAW file built with that frame configuration,
- frames: their tiles (or the whole frame if it has no tiles); only tiles with at least `min_events` events,
- tiles or arrays (height × width × channels).

Smaller tiles are padded to `input_shape`. The samples are collected once; the returned function yields them as the converter expects.

## int8 models for the agents: `opennve quantize`

`opennve quantize` makes an int8 `.tflite` of an agent's Keras model from recordings with typical
data, usually the training data (HDF5 event files, `.raw`). It uses the agent's own frame
configuration and preprocessing, so the int8 model sees the same inputs as later in the
workflow. It always reports how close the int8 model is to the original on inputs it did not see.

```sh
# post-training quantization: round the finished network (seconds to minutes)
opennve quantize config/agents.json --agent prototyp --data "data/train/*.h5" --frames config/frames.json

# quantization aware training: train the U-Net further with the int8 rounding (as accurate as the float network)
opennve quantize config/agents.json --agent prototyp --data "data/train/*.h5" --frames config/frames.json \
    --qat --samples 50000 --epochs 10 --output models/unet_int8.tflite
```

```text
training      epoch: loss, same class as the float network (in the training, with simulated int8)
    0: -, 91.741 %, class 2 IoU 12.6 %
    1: 0.2897, 99.952 %, class 2 IoU 79.0 %
    ...
int8 .tflite  (LiteRT, as openNVE runs it) compared with the original on other inputs:
agreement     99.95 % of the pixels / inputs have the same class as with the original model
class 2       IoU 80.7 %, found 92.0 % of 4285 pixels of the original
```

**Post-training quantization** (without `--qat`) measures the value ranges on `--samples` inputs
(default 500) and rounds the float network to int8. This is enough for robust networks. For the
U-Net it is not: its errors add up from layer to layer (on the test data only about 60 % of the
insect pixels stay the same), and `opennve quantize` warns about this.

**Quantization aware training** (`--qat`) works with the opennve `UNet` and with functional
U-Nets that use the layer names `enc_conv0-1` ... `dec_output`, such as the HSNR U-Net
(`unet_float32.keras` of embedded-unet). The latter are copied into the opennve `UNet`
automatically, after checking that both give the same outputs. QAT trains the network further with
the int8 rounding in the forward pass (`opennve.networks.qat`):

- The batch normalization is folded into the convolutions, as the TensorFlow Lite export does it.
- The weights are rounded per output channel.
- Every activation is rounded to 256 steps in the range measured on the training inputs.
- The output convolution stays float32. It decides the class from small differences between
  large values, and costs almost nothing.

The teacher is the current float network (distillation). The int8 network learns to give the same
classes as the float network on the recordings, so no labels are needed and the result is measured
against the network you use today:

- Pixels of the agent's `target_classes` count `--target-weight` times (default 10). They are rare
  and matter most.
- After every epoch the classes are compared on `--validation` other inputs (default 1000). The
  best epoch is kept, judged by the IoU of the target classes.
- The ranges of the activations are fixed from the calibration (up to 2048 inputs). This trains
  more steadily with sparse event data.

| Option | Default | Meaning |
|---|---|---|
| `--samples` | 500 (QAT 20000) | inputs for the calibration / the training, chosen at random from all recordings |
| `--validation` | 200 (QAT 1000) | other inputs for the comparison |
| `--stride` | 5 | use every n-th time window (`--window-ms`, default 40) |
| `--epochs`, `--batch-size`, `--learning-rate` | 5, 32, 1e-4 | QAT: passes, inputs per step, start of the cosine-decaying learning rate |
| `--target-weight` | 10 | QAT: weight of the target-class pixels |
| `--int8-io` | off | int8 inputs / outputs too (default float32; openNVE converts either way) |
| `--output` | `<model>_int8.tflite` | the int8 model |

More inputs and epochs bring the int8 network closer to the float network. Aim for an agreement
of 99.9 % or more and a target-class IoU close to 100 %. The time per input is about that of the
float network on the CPU, times three per epoch; a GPU (TensorFlow with CUDA) is much faster. The
int8 model runs like every `.tflite` (`"model": "models/unet_int8.tflite"`), and
`opennve benchmark` compares it with the float model. In Python:
`opennve.quantize_agent(config, recordings, frame_config, output, qat=True, epochs=10)`.

## `NeuralEngine`

A collection of named networks:

```python
engine = opennve.NeuralEngine(model_directory="models")

engine.add_network("unet", "UNet", parameters=parameters)          # build
engine.add_network("hsnr", path="models/hsnr_unet.keras")          # load
engine.add(keras_model, name="other")                              # existing model

comparison = engine.transfer_weights("unet", "hsnr")               # hsnr -> unet
comparison.print()

engine.save("unet", "models/unet.keras")
engine.quantize("unet", "int8", representative_dataset=dataset)    # models/unet_int8.tflite
engine.summary("unet")
network = engine["unet"]                                           # by name or index (engine[0])
```

| Method | Description |
|---|---|
| `add_network(name, architecture=None, parameters=None, path=None, safe_mode=True, **options)` | Builds (architecture and parameters) or loads (path) a network |
| `add(network_or_model, name=None)` | Adds an existing `Network` or Keras model |
| `get(name_or_index)`, `engine[...]`, `names`, `networks`, `remove(...)` | Access |
| `save`, `load_weights`, `save_weights`, `summary` | As on `Network` |
| `quantize(name, quantization, path=None, representative_dataset=None)` | TensorFlow Lite export; default path `<model_directory>/<name>_<quantization>.tflite` (`<name>.tflite` for `none`) |
| `transfer_weights(destination, source, method="hsnr_unet", **options)` | Copies weights, returns a `WeightComparison` |
| `compare_weights(a, b)` | `WeightComparison` |

### Transferring weights

`method="hsnr_unet"` copies the weights of a functional U-Net with the layer names `enc_conv{i}-1`, `enc_bn{i}-1`, `enc_conv{i}-2`, `enc_bn{i}-2`, `enc_convbtleneck-1/2`, `enc_bnbtleneck-1/2`, `dec_conv{i}-1`, `dec_bn{i}-1`, `dec_conv{i}-2` (transposed convolution), `dec_bn{i}-2`, `dec_out1`, `dec_out2`, `dec_output` (for instance the HSNR U-Net) into a network built with `UNet`. Options: `layer_depth=4`, `batch_norm=True`.

Other mappings:

```python
# layer by layer: {destination layer path: source layer name}
opennve.networks.transfer_weights_by_name(source_model, destination_model, {
    "u_net.encoder_blocks.0.conv1": "enc_conv0-1",
    "u_net.bottleneck.conv2": "enc_convbtleneck-2",
})

# or as a method for transfer_weights
@opennve.register_weight_transfer("my_mapping")
def my_mapping(source_model, destination_model, **options):
    ...
```

`WeightComparison`: `equal`, `max_difference`, `count_a`, `count_b`, `differences` (one `WeightDifference` per array: `index`, `shape_a`, `shape_b`, `max_difference`), `report()`, `print()`. The arrays are compared in the order of `model.get_weights()`, which can differ between architectures even if the weights were transferred correctly; comparing the outputs (`network.predict`) is the reliable check.

## Migration from the previous neural engine

| Previous code | openNVE |
|---|---|
| `NetworkParameters(NUMBER_OF_CLASSES, TILE_HEIGHT, TILE_WIDTH, CHANNELS)` | `opennve.NetworkParameters(number_of_classes, tile_height, tile_width, channels)` (or `from_dict` with the upper-case names) |
| `NetworkBuilder(parameters).build_network("UNet")` | `opennve.Network.build("UNet", parameters).model` |
| `NetworkBuilder(...).build_network("HSNR_UNet")` | Register the builder once (`register_network("HSNR_UNet")`, see above) |
| `NeuralNetwork(model, parameters=...)` / `NeuralNetwork(model, model_path=...)` | `Network.build(...)` / `Network.load(path)` |
| `predict_model(tiles)`, `get_model()`, `get_weights()` | `predict(tiles)`, `model`, `get_weights()` |
| `NeuralEngine.add_network(name, model, parameters)` / `(name, model, model_path=...)` | `engine.add_network(name, model, parameters=...)` / `(name, path=...)` |
| `NeuralEngine.get_network_list()` | `engine.networks` (or `engine.names`) |
| `network_options(index, path, save_model=True)` and the other flags | `engine.save(index, path)`, `load_weights`, `save_weights`, `summary` |
| `quantization_model(index, "int8", dataset)` (fixed output folder) | `engine.quantize(index, "int8", path=None, representative_dataset=dataset)` |
| `transfer_weights(destination, source, layer_depth, batch_norm)` | `engine.transfer_weights(destination, source, layer_depth=..., batch_norm=...)` |
| `compare_weights(old, new)` (prints) | `opennve.compare_weights(a, b).print()` |
| `Agent(model_path).predict(tile_dicts)` with class 2 pixels | Agent with `"task": "segmentation"`, see [Agents – Segmentation](agents.md#segmentation) |
