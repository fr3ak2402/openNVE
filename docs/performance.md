# Performance

Where the time goes in a typical segmentation workflow (recording → windows → frames → tiles →
agent → labels → CSV, U-Net on 80 × 80 tiles, measured on an x86 CPU):

| Part | Time per tile |
|---|---|
| Reading the events | ~0.01 ms |
| Frame and tiles | ~0.01 ms (openNVE 1.0.0: ~0.05 ms) |
| Preparing the input | ~0.01 ms |
| **Inference (LiteRT, float16 model, one thread)** | **~5 ms** (openNVE 1.0.0: ~6.5 ms) |
| Decoding the segmentation | ~0.2 ms (openNVE 1.0.0: ~0.55 ms) |

The neural network takes more than 90 % of the time. The biggest gains therefore come from the
model and from the number of tiles that go through it.

openNVE already makes models with a dynamic batch size (from Keras) static in memory for the CPU
(chunks of `batch_size`, default 8): then XNNPACK runs the whole model, also the up-sampling of a
U-Net, which is about 25 % faster. The example workflow `main_mh` took 31 s instead of 41 s on a
2-core machine (results identical).

## 1. Use an int8 model

The quantization of the TensorFlow Lite model decides the inference speed on the CPU:

| Quantization (`Network.to_tflite`) | Time per 80 × 80 tile (U-Net, 2 threads) | |
|---|---|---|
| `int8` | ~2.3 ms | **fastest, ~2.4 × faster than float16** |
| `float16` | ~4–5 ms | runs as float32 on x86 / ARM CPUs |
| `none` (float32) | ~4–5 ms | |
| `dynamic` / `float32` (default optimization) | ~13–17 ms | **~3 × slower**: the weights are converted on every call |

The simplest way is `opennve quantize`. It takes the inputs from your recordings, prepared as the
agent prepares them. With `--qat` (quantization aware training) the int8 network is trained to
match the float network, which is needed for U-Nets whose classes change with plain rounding to
int8. It ends with a comparison of the int8 model with the original. See
[int8 models for the agents](networks.md#int8-models-for-the-agents-opennve-quantize):

```sh
opennve quantize config/agents.json --agent unet --data data/recording.h5 --frames config/frames.json \
    --qat --output models/unet_int8.tflite
```

By hand, int8 needs a representative dataset (sample tiles) and the trained Keras model or its weights:

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
boundary; `opennve quantize` reports this comparison, and `--qat` reduces the differences).

## 2. Fewer tiles

Every tile that reaches the agent costs one inference:

- **`min_events_per_ms`** in the frames step skips tiles with little activity (noise). Raising it
  from 0.15 to 0.3 can halve the number of tiles in sparse scenes.
- **Overlapping windows** (`overlap_ms`, or `step_ms` < `duration_ms`) multiply the work: with
  `count: 3, overlap_ms: 20` every event is processed in up to three windows. Use overlap only if
  the results need it.
- **Larger tiles** (if the model accepts them) have less border overhead than many small ones.

## 3. Threads

Leave the threads on `"auto"` (the default): openNVE reads the CPU cores of the machine and gives
the agent step one thread per core with one interpreter thread each, the frames step 1–2 threads
(see [Workflows → Multithreading](workflows.md#multithreading)). `opennve system` shows the cores,
`opennve workflow threads workflows.json` the resulting threads.

Measured on 2 cores (U-Net, 80 × 80 tiles, ms per tile):

| Agent threads × interpreter threads | float16 | int8 |
|---|---|---|
| 1 × 1 | 6.9 | 4.0 |
| 1 × 2 | 4.7 | 3.1 |
| **2 × 1 (auto)** | **3.7** | **2.1** |
| 2 × 2 (more threads than cores) | 3.7 | 8.1 |

- Several interpreters in parallel beat one interpreter with several threads.
- *Agent step threads × interpreter threads* above the number of cores makes the inference slower,
  with int8 models up to 4 times.
- `"cpus": 6` in the workflow (or the environment variable `OPENNVE_CPUS=6`) leaves the other cores
  free, for instance for a live view.
- Look at the statistics at the end of a workflow (`agent x3 1200 items, 9.8 ms per item`): the step
  with the highest time per item and thread is the bottleneck.

## 4. GPU and NPU

Agents can run on the GPU or an NPU (`"device": "gpu"`, `"npu"` or `"auto"` in `agents.json`).
Whether that is faster depends on the model and the hardware; `opennve benchmark` measures it on
your computer. See [GPU and NPU](accelerators.md).

## 5. Batches

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
