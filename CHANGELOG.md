# Changelog

All notable changes to openNVE. Versions follow [semantic versioning](https://semver.org/).

## [1.2.0] - unreleased

### PyTorch models

- New backend `pytorch` (extra `opennve[pytorch]`): agents run PyTorch models from `.pt2`
  (torch.export, recommended), `.pt` / `.pth` (TorchScript, or complete models saved with
  `torch.save(model)` with `"safe_mode": false`) and `nn.Module` objects
  (`Agent.from_model(module)`); `"backend": "auto"` picks it by the file name
- Devices: CPU, GPU via CUDA / ROCm, Apple MPS or Intel XPU (`"device": "gpu"` / `"auto"`,
  `gpu_precision` fp16 with autocast), fallback to the CPU; channels-last memory format (about
  30 % faster convolutions on the CPU)
- Workflows plan the threads of PyTorch agents like those of TensorFlow Lite agents (several
  agent threads, CPU threads split between them, mixed devices)
- `"preprocess": {"layout": "auto"}` is the new default: `nchw` for PyTorch models, `nhwc`
  otherwise (unchanged for existing configurations)
- `opennve system` and `opennve version` show PyTorch and its GPU; `opennve.agents.torch_info()`

### Plugins

- Plugins: separate packages that add backends, commands and system information, found through
  the entry point group `opennve.plugins` (`opennve.plugins.Plugin`); a catalog of installable
  plugins (`opennve/plugins/catalog.py`)
- `opennve plugin list | info | install | remove`; `opennve version` lists the installed plugins,
  `opennve system` shows their information; broken plugins are reported and skipped
- `register_backend(..., threads="planned" | "self", accelerator=...)`, `backend_options(name)`:
  workflows plan the threads of plugin backends like the built-in ones
- A model file of a plugin that is not installed (for instance `.hef`) says which plugin to install

### Hailo (plugin hailo: Raspberry Pi AI Kit, AI HAT+, AI HAT+ 2)

- Backend `hailo` (from the plugin): agents run compiled Hailo models (`.hef`) on Hailo-8, Hailo-8L (HailoRT
  4.x), Hailo-10H and Hailo-15 (HailoRT 5.x); `"backend": "auto"` picks it for `.hef` files; a
  `.hef` compiled for another chip gives a clear error. HailoRT converts the float32 inputs and
  outputs, so `preprocess` / `prediction` stay as for the `.tflite` model. All Hailo agents and
  workflow threads share one device (HailoRT scheduler); workflows give Hailo agents 2 threads
- The Hailo runtime is a separate project, [openNVE-hailo](https://github.com/fr3ak2402/openNVE-hailo)
  (package `opennve_hailo`): HailoRT in Rust with a Python module, built against the installed
  HailoRT 4.x or 5.x, for every Python from 3.11
- `opennve plugin install hailo` sets up the Raspberry Pi in one step (openNVE-hailo `install.sh`):
  PCIe Gen 3, driver, firmware and HailoRT (`hailo-all` / `hailo-h10-all`) and the ready-made
  `opennve_hailo` package for the installed HailoRT version (no Rust or git needed);
  `opennve hailo` (command of the plugin) shows module, driver, HailoRT and devices
- `opennve system` shows HailoRT and its devices with their chip (Hailo-8L, Hailo-10H, ...)

### AMD Kria DPU (plugin vitis: KV260, KR260, K26 / K24)

- Plugin `vitis` in the catalog: agents run compiled Vitis AI models (`.xmodel`) on the DPU
  (DPUCZDX8G) of AMD Kria boards; `"backend": "auto"` picks it for `.xmodel` files. The runtime
  is a separate project, [openNVE-vitis](https://github.com/fr3ak2402/openNVE-vitis) (package
  `opennve_vitis`): the Vitis AI Runtime in Rust with a Python module, with its own Vitis AI
  Runtime inside (independent of the Ubuntu version and AMD's Python packages)
- `opennve plugin install vitis` sets up the board: XRT and the zocl driver, the DPU design
  (KV260: `kv260-benchmark-b4096`), `/etc/vart.conf`, and the ready-made package; `opennve vitis`
  (command of the plugin) shows XRT, the DPU design and the DPU cores
- `opennve system` shows a loaded DPU design (zocl driver) as NPU and suggests the plugin on a Kria
  board without it

### int8 models (opennve quantize)

- `opennve quantize agents.json --data ... --frames ...` makes an int8 TensorFlow Lite model of an
  agent's Keras model from recordings (the training data). It uses the agent's frame
  configuration and preprocessing, takes inputs at random from all files, and compares the int8
  model with the original on other inputs: share of equal classes, IoU / recall of the target
  classes. It warns if they differ clearly
- `--qat`: quantization aware training of the opennve U-Net (`opennve.networks.qat`). The network
  is trained further with the int8 rounding (batch normalization folded, weights per channel,
  activation ranges from the calibration, float32 output convolution) to give the same classes as
  the float network (distillation, no labels needed). Target-class pixels are weighted and the
  best epoch is kept; the activation ranges (minimum / maximum or a percentile) are chosen by
  how well they keep the target classes. Post-training int8 of the U-Net loses many of the insect pixels; QAT keeps
  them. Functional U-Nets with the HSNR layer names (`enc_conv0-1` ... `dec_output`) are copied
  into the opennve U-Net automatically
- Inputs from HDF5 files are taken from time windows at random places, read directly
  (`Hdf5Reader.window_at`): a 300 GB training file needs seconds to minutes, not a pass over all
  events. Without width / height in the file the size is estimated from samples
  (`open_hdf5(..., estimate_size=True)`, or `--width` / `--height`). Progress every 10 s
- Python: `opennve.quantize_agent(...)`, `opennve.quantize_report(report)`

### Fixes

- Softmax of channels-first (`"layout": "nchw"`) segmentation outputs is taken along the class
  axis (it was taken along the width for tiles at the edges and for `result.outputs`);
  `opennve benchmark` compares the classes of such outputs along the class axis as well

## [1.1.0] - 2026-11-20

Faster processing: vectorized segmentation decoding with lazy softmax, threads sized
automatically to the CPU cores of the machine, and agents on the GPU or an NPU. Configurations of
1.0.0 work unchanged.

### Performance

- Segmentation decoding vectorized over the whole batch (one NumPy pass instead of one per tile):
  the decode takes ~0.2 ms instead of ~0.55 ms per tile, results unchanged
- Softmax for segmentation applied lazily: the class maps come from the logits (same argmax), the
  probabilities are computed only for the selected pixels; `Prediction.output` and
  `AgentResult.outputs` are still the probabilities, computed on first access
- CPU inference about 25 % faster: models with a dynamic batch size are made static in memory
  (chunks of `batch_size`, default 8), so XNNPACK runs the whole model including the up-sampling
  of a U-Net (Conv2DTranspose); results unchanged
- Faster softmax (in place, float32)
- Frames about 4.5 × faster (1.4 instead of 6.2 ms for a 1280 × 720 time surface with 80 × 80
  tiles): the time surface is computed for the events only, not for every pixel, and the selected
  tiles are cut out with one copy; results unchanged
- New guide [docs/performance.md](docs/performance.md): int8 models (~2.4 × faster than float16),
  fewer tiles, threads

### Automatic threads

- `"threads": "auto"` for workflow steps, now the default of `frames`, `tiles` and `agent`: the
  threads are sized to the CPU cores of the machine (agent: one thread per core with one
  TensorFlow Lite interpreter thread each; frames / tiles: 1–2 threads); several agent steps share
  the cores
- Agent `"threads": "auto"` (also `null`, the default): a single agent uses all cores up to 4; in
  workflows the interpreter threads follow the agent step threads (cores / step threads)
- `opennve.cpu_count()` (affinity, container CPU limits, environment variable `OPENNVE_CPUS`) and
  `opennve.system_info()`; command `opennve system`

### GPU and NPU

- Agents with TensorFlow Lite models run on the GPU or an NPU: `"device": "gpu"`, `"npu"` or
  `"auto"` in `agents.json`, with the LiteRT 2.4 CompiledModel API (GPU: WebGPU on Windows /
  Linux, Metal on macOS; NPU: Intel Core Ultra, Qualcomm Snapdragon). Falls back to the CPU with
  a message if the device is missing (`"fallback": false` raises an error)
- Models are rewritten in memory for accelerators (static shapes; "same" transposed convolutions
  as "valid" + crop, which the LiteRT GPU otherwise computes wrongly for U-Nets on 80 x 80 tiles);
  a test input checks the results against the CPU when the model is loaded
- `gpu_precision` (`fp16` / `fp32`), `delegate` (TensorFlow Lite delegates, for instance the Coral
  Edge TPU); Keras / SavedModel agents use `tf.device` for `"device": "gpu"`
- Several devices at the same time: `"device": ["gpu", "cpu"]` gives the agent step 2 threads per
  accelerator and one CPU thread per remaining core; each thread takes the next window when it is
  free. The statistics show the inputs per device
- `opennve benchmark` / `opennve.benchmark_agent()`: speed and agreement of CPU, GPU and NPU
- `opennve system` shows the GPU / NPU support of the installed LiteRT; agent steps with GPU / NPU
  agents use 2 threads with `"threads": "auto"`; workflows report where each agent runs
- Extras `npu-intel`, `npu-qualcomm`; dependency `ai-edge-litert>=2.4`
- Dependencies completed: `flatbuffers` (used directly for GPU / NPU models), extras `hdf5`
  (hdf5plugin for compressed HDF5 files), `dev` (pytest, build, packaging); `all` includes
  hdf5plugin
- New guide [docs/accelerators.md](docs/accelerators.md)

### System information

- `opennve system` / `system_info()` show the operating system (name, version, build, kernel;
  `opennve.os_info()`), the computer model, RAM (total, available, container
  limit) and the GPUs and NPUs (Intel AI Boost, AMD Ryzen AI, Apple Neural Engine, Hailo, Coral,
  Rockchip, Qualcomm Hexagon, ...), found through the operating system without extra packages
- `opennve.memory_info()`, `opennve.accelerators()`
- Workflow setting `cpus` (cores to plan for), `queue_size` automatic by default;
  `workflow.thread_plan()` / `thread_report()`, `opennve workflow threads <file> [--cpus n]`

## [1.0.0] - 2026-11-20

First release of openNVE, the Open Neuromorphic Vision Engine: a Python library for event cameras
(DVS / EVS) with the SENSING SE1-S4 driver built in. Requires Python 3.14 or newer; wheels for
Windows x64, Linux x64 / ARM64 (Raspberry Pi 5) and macOS (Apple Silicon and Intel).

### Cameras

- Finding connected cameras, camera configurations in JSON (name, model, serial number, hardware
  settings, filter, output format, time window), several cameras at once (with clock synchronization)
- SENSING SE1-S4 (embedded neuromorphic_drivers with the SE1-S4 driver), Prophesee EVK4 and EVK3 HD
- Automatic WinUSB driver installation on Windows (`install_driver=True`, `opennve driver --install`)

### Events

- Background activity (noise) filter and polarity filter
- Output format with the fields x, y, p, t and l (label); CSV export
- Time windows (40 ms by default), aligned to the camera time

### Recordings

- `.raw` files (Prophesee / Metavision EVT 2.0, 2.1, 3.0) with the same interface as a live camera,
  optionally in real time
- HDF5 files (`.h5`, `.hdf5`) with the columns x, y, p, t and optional labels l: column, compound,
  matrix and pandas layouts; writing labeled events to HDF5
- `open_recording` chooses the reader by the file type

### Frames and tiles

- Frame engine: linear or exponential time surfaces, binary and count frames (for instance 1280 × 720)
- Tiles (for instance 80 × 80) with a minimum number of events, also after processing the frame
- Named frame configurations in their own file (`frames.json`)

### Agents and neural networks

- Agents run TensorFlow Lite (LiteRT) and TensorFlow / Keras models on the tiles or frames of every
  window: classification and segmentation (pixels per class in frame coordinates, statistics)
- Neural networks: U-Net and own architectures, loading models of earlier projects, training with
  Keras, TensorFlow Lite export with float16 / dynamic / int8 quantization, weight transfer
- Registries for new tasks, model runtimes, architectures and weight transfers

### Workflows

- Freely configurable chains of steps (`workflows.json`): camera / recording → time windows → frames
  → tiles → agent → labeled events → CSV / HDF5 / JSON lines
- Every step in its own threads, ordered output, statistics per step; own steps with `register_step`

### Tools

- Command line `opennve`: `list`, `template`, `frame-template`, `check`, `workflow`, `recording`,
  `driver`, `models`, `version`
- Version and build number (`<day of year>.<year>`): `opennve.__version__`, `opennve.BUILD_NUMBER`,
  `opennve version`
- Examples in `examples/`, documentation in `docs/` and in the wiki
