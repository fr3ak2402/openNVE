# GPU and NPU

Agents with TensorFlow Lite models (`.tflite`) can run on the GPU or an NPU instead of the CPU.
openNVE uses the CompiledModel API of [LiteRT](https://ai.google.dev/edge/litert) 2.4 (installed
with openNVE):

| Device | Windows | Linux | macOS |
|---|---|---|---|
| `gpu` | WebGPU (Direct3D 12): NVIDIA, AMD, Intel | WebGPU (Vulkan): NVIDIA, AMD, Intel, Raspberry Pi 5 (V3D), Mali | Metal (Apple silicon) |
| `npu` | Intel Core Ultra (AI Boost, extra `npu-intel`) | x64: Intel Core Ultra (extra `npu-intel`); ARM64: Qualcomm (Hexagon, extra `npu-qualcomm`) | – (the Apple Neural Engine is not supported by LiteRT) |
| `delegate` | TensorFlow Lite delegate libraries, for instance the Coral Edge TPU (`libedgetpu`) | | |

Hailo accelerators (Hailo-8, Hailo-8L, Hailo-10H, Hailo-15; Raspberry Pi AI Kit, AI HAT+,
AI HAT+ 2) run compiled Hailo models (`.hef`) with HailoRT, see [Hailo](#hailo).

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
| `device` | `null` (CPU) | `cpu`, `gpu`, `npu` or `auto` (the NPU, else the GPU, else the CPU — only devices the operating system reports are tried), or a list such as `["gpu", "cpu"]` (several devices at the same time, see below) |
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

## GPU and CPU together

With a list, a workflow uses several devices at the same time:

```json
"device": ["gpu", "cpu"]
```

The agent step then gets 2 threads per accelerator and one thread per remaining CPU core (one core
per accelerator stays free to prepare its inputs). Every thread takes the next time window as soon
as it is free, so a fast GPU gets more windows than a slow one — the step is at least as fast as the
better device alone. For example on 8 cores:

```text
  agent            x9 (auto), interpreter threads prototyp 1, prototyp: gpu x2 + cpu x7
```

At the end the statistics show how many inputs (tiles) each device ran, here from a machine with
2 cores and a software GPU (which is why it is slow):

```text
  agent            x3  26 items, 25955.7 ms per item (gpu 6485 inputs, cpu 4320 inputs)
```

- `["npu", "gpu", "cpu"]` uses all three; `["gpu", "npu"]` only the accelerators.
- With a fixed number of threads (`"threads": 4` in the step) the first threads go to the
  accelerators (2 each), at least one to the CPU.
- If an accelerator is not available, its threads run on the CPU (with a message).
- Use `"gpu_precision": "fp32"` if the results must not depend on which device ran a window.
- A single agent outside a workflow uses the first device of the list that works.

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
# Qualcomm (Hexagon NPU, Linux ARM64)
pip install "opennve[npu-qualcomm]"
```

and the NPU driver of the operating system (Windows: device manager → *Neural processors*;
Linux: Intel `intel_vpu` / `/dev/accel`).

## Hailo

Hailo accelerators run agents with compiled Hailo models (`.hef`). openNVE runs them with
HailoRT, Hailo's runtime; LiteRT and PyTorch cannot use Hailo chips. Hailo has two software
generations, and only one of them can be installed on a computer:

| Chip | Boards | HailoRT | Raspberry Pi OS package | `.hef` compiled with |
|---|---|---|---|---|
| Hailo-8L (13 TOPS) | Raspberry Pi AI Kit, AI HAT+ 13 TOPS, M.2 modules | 4.x | `hailo-all` | Dataflow Compiler 3.x, target `hailo8l` |
| Hailo-8 (26 TOPS) | Raspberry Pi AI HAT+ 26 TOPS, M.2 / PCIe cards | 4.x | `hailo-all` | Dataflow Compiler 3.x, target `hailo8` (or `hailo8l`) |
| Hailo-10H (40 TOPS int4) | Raspberry Pi AI HAT+ 2, M.2 modules | 5.x | `hailo-h10-all` | Dataflow Compiler 5.x, target `hailo10h` |
| Hailo-15H / 15M / 15L | Hailo-15 vision processors (camera SoC with its own Linux) | 5.x, on the Hailo-15 | – | Dataflow Compiler 5.x, target `hailo15h` / ... |

A `.hef` runs only on the chip it was compiled for (a `hailo8l` model also runs on the Hailo-8;
Hailo-15H models run on the Hailo-10H). openNVE uses the same code for all of them.

```json
{"name": "prototyp", "model": "../models/unet_hailo8l.hef", "frame": "tiles80",
 "prediction": {"task": "segmentation", "activation": "softmax", "target_classes": [2]}}
```

`"backend": "auto"` picks the `hailo` backend for `.hef` files; `"device"` can be left out (or
`"npu"`): a `.hef` runs only on the Hailo accelerator. `batch_size` (default 8) is the Hailo batch.

**The model.** A `.hef` is compiled from a `.tflite` or ONNX model with the Hailo Dataflow
Compiler (Hailo AI Software Suite, Linux x86 only, free account at the Hailo Developer Zone; the
version must match the chip, see the table). The compiler quantizes the model (8 bit, single
layers 16 bit; the Hailo-10H also 4 bit) with calibration data: use tiles of your recordings
(for instance `agent.prepare(frame.tiles)` of a few hundred windows, saved with `numpy.save`), and
compare the results with the `.tflite` model (`opennve benchmark`, or the CSV of a workflow) –
quantization can change results. HailoRT converts the float32 inputs and outputs of openNVE to
the quantized values and back, so the `preprocess` and `prediction` settings stay the same as for
the `.tflite` model. Inputs and outputs are channels last (`nhwc`).

**Installation.** openNVE runs `.hef` models with
[openNVE-hailo](https://github.com/fr3ak2402/openNVE-hailo) (package `opennve_hailo`): HailoRT in
Rust with a Python module, like the camera drivers in neuromorphic-drivers. It uses the
asynchronous HailoRT API, which works on all Hailo chips, and is compiled against the HailoRT
installed on the computer (its `install.sh` downloads the matching HailoRT headers); one build
works for every Python from 3.11. On the Raspberry Pi 5:

```sh
sudo apt install dkms hailo-all                # AI Kit / AI HAT+ (AI HAT+ 2: hailo-h10-all); reboot afterwards
sudo apt install git g++                       # C++ compiler
curl https://sh.rustup.rs -sSf | sh            # Rust
source ~/opennve/.venv/bin/activate            # the environment openNVE is installed in
git clone https://github.com/fr3ak2402/openNVE-hailo
bash openNVE-hailo/install.sh
opennve system
```

```text
  NPU                   Hailo AI accelerator (hailo_pci)
  HailoRT 4.23.0        1 Hailo device (Hailo-8L) (.hef models)
```

Run `install.sh` again after a HailoRT update: openNVE-hailo is compiled for one HailoRT version.
Other Linux computers (x86_64, ARM64) with a Hailo M.2 / PCIe module work the same way with the
HailoRT packages of the Hailo Developer Zone; Windows (HailoRT Windows installer) is not tested.

**Hailo-15.** The Hailo-15 is a camera processor with its own ARM cores and Linux; HailoRT runs
on the Hailo-15 itself. openNVE would have to run there as well (with Python 3.14, which the
Hailo-15 Linux images do not include) – not tested.

**Workflows.** Hailo agents get 2 workflow threads (like GPU agents); all Hailo agents and threads
of a program share the device, HailoRT switches between them. `opennve benchmark agents.json
--devices npu` measures the model on the Hailo.

## Limits

- **TensorFlow in the same program:** the LiteRT GPU accelerator crashes in a program that has
  loaded TensorFlow. If TensorFlow is loaded (Keras agents, training), GPU agents fall back to the
  CPU. Use `.tflite` models only, or separate programs.
- **Keras / SavedModel agents** (`backend` `tensorflow`) use the GPU through TensorFlow (CUDA on
  Linux, `tensorflow-metal` on macOS) with `"device": "gpu"`; TensorFlow has no NPU support.
- **PyTorch agents** (`backend` `pytorch`) use the GPU through PyTorch with `"device": "gpu"`:
  CUDA (NVIDIA; AMD with the ROCm build), Apple MPS or Intel XPU, whichever the installed PyTorch
  supports (`opennve system` shows it). `"gpu_precision": "fp16"` runs with `torch.autocast`.
  PyTorch has no NPU support. See [Agents](agents.md#pytorch-models).
- **Hailo**: only compiled `.hef` models; openNVE does not compile them (Hailo Dataflow
  Compiler, see [Hailo](#hailo)). Not tested on Hailo hardware yet; Hailo-15 only with openNVE
  running on the Hailo-15. The
  **Apple Neural Engine** needs Core ML models.
- `fp16` on the GPU can change the class of single pixels close to a decision boundary;
  `"gpu_precision": "fp32"` gives the results of the CPU.
