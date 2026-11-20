# Changelog

All notable changes to openNVE. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

### Performance

- Segmentation decoding vectorized over the whole batch (one NumPy pass instead of one per tile):
  the decode takes ~0.2 ms instead of ~0.55 ms per tile, results unchanged
- Softmax for segmentation applied lazily: the class maps come from the logits (same argmax), the
  probabilities are computed only for the selected pixels; `Prediction.output` and
  `AgentResult.outputs` are still the probabilities, computed on first access
- Faster softmax (in place, float32)
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
