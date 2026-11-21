# Performance

Where the time goes in a typical segmentation workflow (recording → windows → frames → tiles →
agent → labels → CSV, U-Net on 80 × 80 tiles, measured on an x86 CPU):

| Part | Time per tile |
|---|---|
| Reading the events | ~0.01 ms |
| Frame and tiles | ~0.1 ms |
| Preparing the input | ~0.01 ms |
| **Inference (TensorFlow Lite, float16 model)** | **~4 ms** |
| Decoding the segmentation | ~0.2 ms (openNVE 1.0.0: ~0.55 ms) |

The neural network takes more than 90 % of the time. The biggest gains therefore come from the
model and from the number of tiles that go through it.

## 1. Use an int8 model

The quantization of the TensorFlow Lite model decides the inference speed on the CPU:

| Quantization (`Network.to_tflite`) | Time per 80 × 80 tile (U-Net, 2 threads) | |
|---|---|---|
| `int8` | ~2.3 ms | **fastest, ~2.4 × faster than float16** |
| `float16` | ~4–5 ms | runs as float32 on x86 / ARM CPUs |
| `none` (float32) | ~4–5 ms | |
| `dynamic` / `float32` (default optimization) | ~13–17 ms | **~3 × slower**: the weights are converted on every call |

int8 needs a representative dataset (sample tiles) and the trained Keras model or its weights:

```python
import opennve

network = opennve.Network.load("models/unet.keras")          # or build the network and load the weights
tiles80 = opennve.load_frame_config("config/frames.json", "tiles80")
with opennve.open_hdf5("data/recording.h5", window={"duration_ms": 40}) as recording:
    dataset = opennve.representative_dataset((recording, tiles80), input_shape=(80, 80, 3), max_samples=300)
    network.to_tflite("models/unet_int8.tflite", "int8", representative_dataset=dataset)
```

The int8 model has int8 inputs and outputs; the agent quantizes the input and dequantizes the
output automatically, so `agents.json` only needs the new `model` path. Check the results against
the float model once (the class maps of int8 models can differ for pixels close to a decision
boundary).

## 2. Fewer tiles

Every tile that reaches the agent costs one inference:

- **`min_events_per_ms`** in the frames step skips tiles with little activity (noise). Raising it
  from 0.15 to 0.3 can halve the number of tiles in sparse scenes.
- **Overlapping windows** (`overlap_ms`, or `step_ms` < `duration_ms`) multiply the work: with
  `count: 3, overlap_ms: 20` every event is processed in up to three windows. Use overlap only if
  the results need it.
- **Larger tiles** (if the model accepts them) have less border overhead than many small ones.

## 3. Threads

- The `threads` of an agent step run several interpreters in parallel (one model per thread); the
  agent configuration's `threads` sets the CPU threads of each TensorFlow Lite interpreter.
- Keep *agent step threads × interpreter threads* at or below the number of CPU cores. More threads
  than cores make every inference slower.
- The frames step is cheap; 1–2 threads are enough. Log and write steps are ordered and use one
  thread.
- Look at the statistics at the end of a workflow (`agent x3 1200 items, 9.8 ms per item`): the step
  with the highest time per item and thread is the bottleneck.

## 4. Batches

`batch_size` in `agents.json` runs several tiles in one interpreter call. On the CPU this helps only
a little (a few percent); it pays off on accelerators (GPU / NPU delegates).

## Measuring

```python
import time, opennve

agent = opennve.Agent(opennve.load_agent_config("config/agents.json", "prototyp")).load()
frame_config = opennve.load_frame_config("config/frames.json", "tiles80")
with opennve.open_hdf5("data/recording.h5", window={"duration_ms": 40}) as recording:
    frames = [window.frame(frame_config) for window in recording.windows()][:20]
start, tiles = time.perf_counter(), 0
for frame in frames:
    result = agent.predict(frame)
    tiles += len(result.predictions)
print(f"{(time.perf_counter() - start) * 1000 / tiles:.2f} ms per tile")
```
