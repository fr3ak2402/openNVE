# GPU and NPU

Agents with TensorFlow Lite models (`.tflite`) can run on the GPU or an NPU instead of the CPU.
openNVE uses the CompiledModel API of [LiteRT](https://ai.google.dev/edge/litert) 2.4 (installed
with openNVE):

| Device | Windows | Linux | macOS |
|---|---|---|---|
| `gpu` | WebGPU (Direct3D 12): NVIDIA, AMD, Intel, Qualcomm Adreno | WebGPU (Vulkan): NVIDIA, AMD, Intel, Raspberry Pi 5 (V3D), Mali | Metal (Apple silicon) |
| `npu` | Intel Core Ultra (AI Boost, extra `npu-intel`), Qualcomm Snapdragon X (Hexagon, extra `npu-qualcomm`) | Intel Core Ultra (extra `npu-intel`), Qualcomm (extra `npu-qualcomm`) | – (the Apple Neural Engine is not supported by LiteRT) |
| `delegate` | TensorFlow Lite delegate libraries, for instance the Coral Edge TPU (`libedgetpu`) | | |

`opennve system` shows the GPUs and NPUs of the computer and what the installed LiteRT supports
(example):

```text
  GPU                   NVIDIA GeForce RTX 4070 (12.0 GB, nvidia)
  NPU                   Intel NPU (AI Boost) (intel_vpu)
  LiteRT 2.4.0          GPU via WebGPU, NPU Intel OpenVINO (needs pip install opennve[npu-intel])
```

## Configuration

In `agents.json`:

```json
{
    "name": "prototyp",
    "model": "../models/UNet_float16.tflite",
    "device": "gpu",
    "batch_size": 16
}
```

| Setting | Default | Description |
|---|---|---|
| `device` | `null` (CPU) | `cpu`, `gpu`, `npu` or `auto` (the NPU, else the GPU, else the CPU — only devices the operating system reports are tried) |
| `fallback` | `true` | If the device is not available or cannot run the model: `true` runs it on the CPU (with a message), `false` stops with an error |
| `batch_size` | GPU 8, NPU 1 | Inputs per call. The model is compiled for this batch size; larger batches use the GPU better |
| `gpu_precision` | `"fp16"` | `fp16` (faster) or `fp32` (the same results as the CPU) |
| `delegate` | `null` | TensorFlow Lite delegate: `"libedgetpu.so.1"` or `{"library": "...", "options": {"device": "usb"}}` |
| `threads` | `"auto"` | CPU threads for operations that run on the CPU |

When a workflow starts, it reports where each agent runs:

```text
[agent] agent prototyp runs on GPU (batch 8)
[agent] agent prototyp runs on CPU (NPU: not available, or it does not support all operations of UNet_float16.tflite)
```

In Python: `agent.device` (`"cpu"`, `"gpu"`, `"npu"`, `"gpu+cpu"`, `"delegate"`) and
`agent.device_description()`.

## Is it faster? `opennve benchmark`

Whether the GPU or NPU is faster depends on the model and the hardware: small models (like a
U-Net on 80 × 80 tiles) have little work per call, and every call to the GPU costs time. Measure
on your computer, with your recording:

```sh
opennve benchmark config/agents.json --agent prototyp --recording data/val_r_mixed_hard.h5 --frames config/frames.json
```

The output, here from a machine without a real GPU (the "GPU" is a software Vulkan renderer on
the CPU, so it is slow) and without an NPU:

```text
device   runs on    batch  ms/input  inputs/s  agreement     load
cpu      cpu            -      4.41       227   100.00 %     10ms
gpu      gpu            8     59.35        17    99.99 %    812ms
npu      cpu            -      4.88       205   100.00 %     41ms
720 inputs; agreement: equal classes compared with cpu
npu: NPU: not available, or it does not support all operations of UNet_float16.tflite
```

*agreement* is the share of pixels (segmentation) or inputs (classification) with the same class
as on the CPU. `--devices gpu,cpu` changes the order, `--json` gives the numbers as JSON. In Python:
`opennve.benchmark_agent(config, devices=("cpu", "gpu"), frames=frames)`.

The CPU column is one agent with all its threads; in a workflow, `"threads": "auto"` runs one
agent per core in parallel, so compare the GPU with *ms/input of the CPU ÷ cores* as well.

## How it works

- **Static shapes:** accelerators need fixed tensor sizes. Models from Keras have a dynamic batch
  dimension, and `Conv2DTranspose` computes its output size at run time. openNVE rewrites the
  model in memory when it loads it (the file is not changed): fixed batch size, computed shapes
  replaced by constants (`opennve.agents.static_model`).
- **Transposed convolutions** with `"same"` padding are replaced by `"valid"` padding and a crop
  (the same result): the LiteRT GPU computes some sizes wrongly (5 × 5 → 10 × 10 with a 3 × 3 kernel,
  as in a U-Net on 80 × 80 tiles).
- **Check:** after loading on a GPU / NPU, openNVE runs a test input on the accelerator and on the
  CPU. If the results differ clearly, the accelerator is not used (`fallback`) and the reason is
  reported.
- **GPU:** all operations must run on the GPU; if some are not supported, the GPU runs what it can
  and the CPU the rest (`device` `"gpu+cpu"`).
- **NPU:** used only if it runs the whole model. NPUs are usually fastest with int8 models
  (see [Neural networks](networks.md), `to_tflite(..., "int8")`).
- **Threads:** an agent step with GPU / NPU agents uses 2 threads with `"threads": "auto"` (one
  model each), so preparing and decoding the tiles overlaps with the inference.

## NPU runtimes

The NPU needs the runtime of its vendor, installed as an extra:

```sh
# Intel Core Ultra (AI Boost): the OpenVINO SDK comes from the OpenVINO nightly index
pip install "opennve[npu-intel]" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# Qualcomm Snapdragon (Hexagon NPU)
pip install "opennve[npu-qualcomm]"
```

and the NPU driver of the operating system (Windows: device manager → *Neural processors*;
Linux: Intel `intel_vpu` / `/dev/accel`).

## Limits

- **TensorFlow in the same program:** the LiteRT GPU accelerator crashes in a program that has
  loaded TensorFlow. If TensorFlow is loaded (Keras agents, training), GPU agents fall back to the
  CPU. Use `.tflite` models only, or separate programs.
- **Keras / SavedModel agents** (`backend` `tensorflow`) use the GPU through TensorFlow (CUDA on
  Linux, `tensorflow-metal` on macOS) with `"device": "gpu"`; TensorFlow has no NPU support.
- **Hailo** (Raspberry Pi AI Kit / HAT) needs models compiled with Hailo's tools (`.hef`) and is
  not supported. The **Apple Neural Engine** needs Core ML models.
- `fp16` on the GPU can change the class of single pixels close to a decision boundary;
  `"gpu_precision": "fp32"` gives the results of the CPU.
