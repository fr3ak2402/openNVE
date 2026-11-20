# Agents (TensorFlow models)

An **agent** runs a TensorFlow or TensorFlow Lite model on the frames of a camera or a RAW file, for instance to classify every 80 × 80 tile of each 40 ms frame with the model's predict function. Agents are configured in their own JSON file, next to the [camera](configuration.md) and [frame](frames.md) configurations:

```text
cameras.json  ──►  camera / RAW file ──► time window (40 ms)
frames.json   ──►  frame configuration "tiles80" ──► Frame ──► Tiles (80 × 80)
agents.json   ──►  agent "insects" (model.tflite) ──► predict(tiles) ──► one prediction per tile
```

## Installation

The model runtimes are optional, install the one you need:

| Model | Runtime | Install |
|---|---|---|
| TensorFlow Lite (`.tflite`) | [LiteRT](https://ai.google.dev/edge/litert) (TensorFlow Lite runtime) | `pip install ai-edge-litert` or the extra `opennve[tflite]` |
| TensorFlow / Keras (`.keras`, `.h5`, SavedModel directory) | TensorFlow | `pip install tf-nightly` (Python 3.14, see below) |

LiteRT is small (about 15 MB) and has wheels for Python 3.14 on Windows x64, Linux x64 and ARM64 (Raspberry Pi 5) and macOS Apple Silicon. It is the recommended runtime for live cameras.

> **TensorFlow and Python 3.14:** The current stable TensorFlow release (2.21) has wheels for Python 3.10 to 3.13 only. For Python 3.14, which openNVE requires, install the nightly build `pip install tf-nightly` (Windows, Linux, macOS Apple Silicon), or convert the model to TensorFlow Lite and use LiteRT. Once a stable TensorFlow release supports Python 3.14, `pip install tensorflow` works the same way. If LiteRT is not installed, `.tflite` models also run with TensorFlow (`tf.lite`).

Converting a Keras model to TensorFlow Lite (with TensorFlow, on any Python version):

```python
import tensorflow as tf

model = tf.keras.models.load_model("insects.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
open("insects.tflite", "wb").write(converter.convert())
```

## Configuration

`examples/agents.json`:

```json
{
    "agents": [
        {
            "name": "insects",
            "model": "models/insects.tflite",
            "backend": "auto",
            "input": "tiles",
            "frame": "tiles80",
            "batch_size": 64,
            "threads": 2,
            "preprocess": {"channels": "both", "pad": true, "resize": null, "layout": "nhwc", "scale": 1.0},
            "prediction": {"activation": "none", "labels": ["background", "insect"], "threshold": 0.5}
        },
        {
            "name": "scene",
            "model": "models/scene.keras",
            "input": "frame",
            "frame": "binary",
            "call": "predict",
            "preprocess": {"channels": "sum", "resize": [160, 90]},
            "prediction": {"activation": "softmax", "labels": ["empty", "motion", "crowd"]}
        }
    ]
}
```

A file contains one agent (an object), a list, or `{"agents": [...]}`. Names must be unique. Only `model` is required; the name defaults to the model's file name. Check a file with `opennve check agents.json` (`--load` also loads the models and shows their input shape).

| Field | Default | Description |
|---|---|---|
| `name` | model file name | Name of the agent (key of the results) |
| `model` | – | Model path: `.tflite`, `.keras`, `.h5` or a SavedModel directory. Relative paths are relative to the agent file |
| `backend` | `"auto"` | `tflite`, `tensorflow`, or `auto` (by the file name: `.tflite` → `tflite`, otherwise `tensorflow`) |
| `input` | `"tiles"` | `tiles`: one model input per tile of the frame. `frame`: the whole frame is one input |
| `frame` | `null` | Name of the [frame configuration](frames.md) that builds the input (from `frames.json`). May be omitted if only one frame configuration is used |
| `batch_size` | `null` | Inputs per model call. `null` = all tiles of a frame in one call |
| `threads` | `null` | CPU threads of the TensorFlow Lite interpreter |
| `call` | `"predict"` | Keras models: `predict` calls `model.predict(batch)`; `call` calls `model(batch)`, which has less overhead for small batches |
| `signature` | `"serving_default"` | Signature of a SavedModel directory |
| `safe_mode` | `true` | Keras models: `false` allows loading models with `Lambda` layers. Custom layers (for instance the [UNet](networks.md) layers) are found automatically |

### `preprocess` – from frames and tiles to model inputs

Frames and tiles have the shape height × width × 2 (`float32`, channel 0 = polarity 0, channel 1 = polarity 1, see [Frames and tiles](frames.md)). The model input is built from them as follows:

| Field | Default | Description |
|---|---|---|
| `channels` | `"both"` | `both`: 2 channels. `on` / `off`: 1 channel with polarity 1 / 0. `sum` / `max`: 1 channel with both polarities combined |
| `pad` | `true` | The tiles at the right and bottom edges can be smaller than the tile size (for instance 720 = 9 × 80, but 720 = 11 × 64 + 16). `true` pads them with zeros to the full tile size |
| `resize` | `null` | `[width, height]` to resize every input to. `null`: the model's input size, if the model has a fixed size; otherwise unchanged |
| `layout` | `"nhwc"` | `nhwc` (batch, height, width, channels; TensorFlow) or `nchw` |
| `scale` | `1.0` | Factor for all values, for instance `255` for a model trained on 0–255 images |

The batch is converted to the model's input type. Quantized TensorFlow Lite models (int8 / uint8) are quantized and their outputs dequantized automatically. The number of channels must match the model; otherwise the error message names the expected number.

### `prediction` – reading the outputs

| Field | Default | Description |
|---|---|---|
| `task` | `"classification"` | `classification`: one output vector per tile. `segmentation`: one class per pixel (see [Segmentation](#segmentation)) |
| `activation` | `"none"` | `softmax` or `sigmoid` are applied to the first model output (along the class axis), for models that return logits. `none`: the outputs are used as they are |
| `labels` | `null` | Class names: `labels[i]` is the name of class `i` |
| `threshold` | `null` | Classification: a prediction is **selected** if its score is at least the threshold (`null`: all are selected). Segmentation: minimum probability of a target pixel (`null`: the most probable class counts) |
| `target_classes` | `null` | Segmentation: classes to detect, for instance `[2]`. `null`: all classes except 0 |
| `min_pixels` | `1` | Segmentation: a tile is selected if it has at least this many target pixels |
| `mask_inactive` | `false` | Segmentation: only count target pixels where the input has events |
| `activity_threshold` | `0.0` | Segmentation: a pixel is active if one of its input channels is above this value |

For classification, the first model output of every input gives:

- `class_id`: index of the highest value (for an output with one value: `1` if the value is ≥ 0.5, else `0`),
- `score`: the highest value (or the single value),
- `label`: `labels[class_id]`.

Outputs that are not vectors (for instance maps) have no class and score; they are available as `prediction.output`.

## Segmentation

Segmentation models such as the [U-Net](networks.md#unet) return one class per pixel: an output of tile height × tile width × classes. With `"task": "segmentation"`, the agent finds the pixels of the target classes and converts them to frame coordinates:

```json
{
    "name": "insects",
    "model": "models/unet_int8.tflite",
    "frame": "tiles80",
    "prediction": {
        "task": "segmentation",
        "activation": "softmax",
        "labels": ["background", "noise", "insect"],
        "target_classes": [2],
        "min_pixels": 1
    }
}
```

For every tile:

1. the softmax (for logits) gives the probability of each class per pixel,
2. the class map is the most probable class per pixel (padded edge tiles are cropped back to their real size; outputs with another resolution are scaled to the tile size),
3. target pixels are the pixels of `target_classes` (optionally only active pixels, `mask_inactive`, and only with a probability ≥ `threshold`),
4. the tile is selected if it has at least `min_pixels` target pixels.

```python
for step in opennve.run_agents(camera, agents, frame_configs):
    result = step["insects"]
    for prediction in result.selected:                  # tiles with insect pixels
        for pixel in prediction.pixels:                 # numpy structured array: x, y, class_id, score
            print(pixel["x"], pixel["y"], pixel["score"])   # frame coordinates
    result.print_statistics()                           # debugging, see below
```

| `Prediction` (segmentation) | Description |
|---|---|
| `pixels` | Target pixels: structured array with `x`, `y` (frame coordinates), `class_id`, `score` (probability of its class) |
| `pixel_scores` | Probabilities of all classes for each target pixel (pixels × classes) |
| `pixel_list()` | The target pixels as dictionaries `{"x", "y", "class_id", "score", "scores": {"background": ..., "noise": ..., "insect": ...}}` |
| `pixel_count` | Number of target pixels |
| `class_map` | Class per pixel (tile height × width) |
| `output` | Probabilities (tile height × width × classes) |
| `class_counts` | Pixels per class in this tile |
| `activity_counts` | Pixels per class, split into inactive (row 0) and active (row 1) pixels |
| `class_id`, `label`, `score` | Most frequent target class in the tile and the mean probability of its pixels |

`result.statistics()` summarizes all tiles of a frame: pixels per class, active and inactive pixels per class (does the model mark pixels without events as insects?), and the mean, median, minimum and maximum probability of the target pixels. `result.statistics_report()` returns it as text, `result.print_statistics()` prints it. `result.to_dict(selected_only=True)` contains the target pixels of the selected tiles (`pixel_list()`), ready for JSON.

## Usage

### Running agents on a camera or a RAW file

```python
import opennve

agents = opennve.load_agents("agents.json")
frame_configs = opennve.load_frame_configs("frames.json")

with opennve.open_camera(path="cameras.json", name="left") as camera:   # or opennve.open_raw("recording.raw")
    for step in opennve.run_agents(camera, agents, frame_configs):
        for prediction in step.results["insects"].selected:
            tile = prediction.tile
            print(tile.id, tile.row, tile.col, prediction.label, prediction.score)
        scene = step.results["scene"].predictions[0]
        print(scene.label, scene.score)
```

`run_agents(source, agents, frame_configs, duration_ms=None)` loads the models, then yields one `AgentStep` per time window:

- `step.results`: `{agent name: AgentResult}` (also `step["insects"]`),
- `step.frames`: `{frame configuration name: Frame}` – each frame is built once per window, also if several agents use it,
- `step.window`: the time window with its events (`step.window.events`).

`agents` and `frame_configs` can also be file paths: `opennve.run_agents(camera, "agents.json", "frames.json")`. For several cameras read in one loop, use `opennve.process_window(window, agents, frame_configs)` for each window from `camera.read_windows()`.

### One agent, one frame

```python
agent = opennve.load_agent("agents.json", "insects")
tiles80 = opennve.load_frame_config("frames.json", "tiles80")

for frame in camera.frames(tiles80):
    result = agent.predict(frame)        # predict on all tiles of the frame
    for prediction in result.selected:
        print(prediction.tile_id, prediction.label, prediction.score)
```

`agent.predict(...)` accepts:

- a `Frame`: its tiles (`"input": "tiles"`) or the whole frame (`"input": "frame"`),
- a list of tiles, for instance `agent.predict(frame.split(80))` after [processing the frame](frames.md#frame-first-tiles-later),
- a numpy array: one input (height × width × channels) or a batch.

`agent.predict_batch(batch)` runs the model on an already prepared batch and returns the raw outputs; `agent.prepare(images)` builds such a batch.

### Models created in code

A model object with a `predict(batch)` method, for instance a Keras model that is already loaded, or any callable, can be used directly:

```python
model = keras.models.load_model("insects.keras")
agent = opennve.Agent.from_model(model, name="insects", prediction={"labels": ["background", "insect"]})
result = agent.predict(frame)
```

### Results

| `AgentResult` | Description |
|---|---|
| `agent` | Agent name |
| `frame` | The frame the agent ran on (`frame.camera`, `frame.start_t`, `frame.name`, …) |
| `predictions` | One `Prediction` per tile, in tile order (one for `"input": "frame"`). Empty if the frame has no tiles with enough events (`min_events`) |
| `selected` | The predictions with `score >= threshold` |
| `outputs` | The model outputs for all inputs (first dimension = number of tiles) |
| `inference_ms` | Time spent in the model |
| `by_tile()` | `{tile id: Prediction}` |
| `to_dict()` | JSON-friendly summary (without tile data), `to_dict(selected_only=True)` |
| `statistics()`, `statistics_report()`, `print_statistics()` | Segmentation statistics (see [Segmentation](#segmentation)) |

| `Prediction` | Description |
|---|---|
| `tile` | The `Tile` (`id`, `row`, `col`, `x`, `y`, `width`, `height`, `event_count`, `data`); `None` for frame inputs |
| `tile_id` | `tile.id` |
| `output` | First model output for this input (after the activation) |
| `outputs` | All model outputs for this input |
| `class_id`, `label`, `score` | See [prediction](#prediction--reading-the-outputs) |
| `selected` | `score >= threshold` |

## Extending

Tasks (how outputs are read) and backends (model runtimes) are registries, so new ones can be added without changing openNVE:

```python
@opennve.register_task("detection")
def decode_detection(task: opennve.TaskInput) -> list[opennve.Prediction]:
    # task.outputs: model outputs (after the activation), task.tiles / task.frame, task.images (inputs),
    # task.config: the prediction settings; return one Prediction per input
    ...

@opennve.register_backend("onnx", suffixes=[".onnx"])
class OnnxBackend(opennve.agents.Backend):
    def __init__(self, config: opennve.AgentConfig):
        ...  # load config.model_path; set self.input = opennve.agents.ModelInput(shape, dtype)
    def predict(self, batch):
        ...  # return a list of output arrays (first dimension = batch)
```

`"task": "detection"` and `"backend": "onnx"` (or a `.onnx` model with `"backend": "auto"`) then work in `agents.json`. `opennve.list_tasks()` and `opennve.list_backends()` list the registered ones. New network architectures are added with `register_network`, see [Neural networks](networks.md).

## Example

```sh
python examples/agents.py examples/cameras.json --camera left       # live camera
python examples/agents.py recording.raw --agent insects               # RAW file, one agent
```

The example prints the selected tiles of every agent and draws them on the frame. The model files named in `examples/agents.json` (`models/unet_int8.tflite`: a U-Net with segmentation, `models/scene.keras`) are placeholders for your own models; see [Neural networks](networks.md) for building and converting a U-Net.

## Performance

- Load the models before starting the camera (`agent.load()`; `run_agents` does this as well). Importing TensorFlow and loading a model takes seconds.
- TensorFlow Lite with LiteRT has the lowest latency per call. With 40 ms windows, the agents should need less than 40 ms per window, otherwise events pile up in the camera buffer (`packet.status.ring.backlog` grows).
- Tiles with few events can be skipped with `"min_events"` in the frame configuration, so the model only sees active tiles.
- `batch_size: null` runs all tiles of a frame in one call, which is usually fastest. Models with a fixed batch size (for instance 1) are called once per input automatically.
- For Keras models, `"call": "call"` avoids the overhead of `model.predict` for small batches.

## API

| Name | Description |
|---|---|
| `load_agents(path)` / `load_agent(path, name=None)` | Agents of a file (models are loaded on first use) |
| `load_agent_configs(path)` / `load_agent_config(path, name=None)` / `save_agent_configs(path, configs)` | Agent configurations |
| `AgentConfig(model, name=None, backend="auto", input="tiles", frame=None, batch_size=None, threads=None, call="predict", signature="serving_default", preprocess=..., prediction=...)` | One agent (`from_dict`, `to_dict`, `from_json`, `to_json`, `copy`, `model_path`) |
| `PreprocessConfig`, `PredictionConfig` | The `preprocess` and `prediction` sections |
| `Agent(config, model=None)` | `load()`, `predict(frame / tiles / array)`, `predict_batch(batch)`, `prepare(images)`, `input` (model input shape and dtype), `close()`; context manager |
| `Agent.from_model(model, name="model", **settings)` | Agent for a model object with `predict(batch)` |
| `run_agents(source, agents, frame_configs=None, duration_ms=None)` | Yields one `AgentStep` per time window of a camera or RAW file |
| `process_window(window, agents, frame_configs=None)` | One `AgentStep` for one window |
| `AgentStep` | `window`, `frames`, `results`, `step[agent name]` |
| `AgentResult`, `Prediction` | See [Results](#results) and [Segmentation](#segmentation) |
| `register_task(name)`, `list_tasks()`, `TaskInput` | Output decoders (`classification`, `segmentation`) |
| `register_backend(name, suffixes=())`, `list_backends()` | Model runtimes (`tflite`, `tensorflow`) |
