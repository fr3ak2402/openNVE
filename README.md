# openNVE

Open Neuromorphic Vision Engine: a Python library to find, configure and read event cameras.

📖 **Documentation: [Wiki](https://github.com/fr3ak2402/openNVE/wiki) / [docs/](docs/README.md)** – [Installation](docs/installation.md) · [Quick start](docs/quickstart.md) · [Configuration](docs/configuration.md) · [Python API](docs/api.md) · [Time windows and frames](docs/time-windows.md) · [Frames and tiles](docs/frames.md) · [Multiple cameras](docs/multiple-cameras.md) · [RAW files](docs/raw-files.md) · [HDF5 files](docs/hdf5-files.md) · [Neural networks](docs/networks.md) · [Agents](docs/agents.md) · [Workflows](docs/workflows.md) · [Performance](docs/performance.md) · [GPU and NPU](docs/accelerators.md) · [Troubleshooting](docs/troubleshooting.md) · [Changelog](CHANGELOG.md)

## Features

### Cameras
- **Find cameras:** list all connected event cameras with model and serial number (`opennve list`, `opennve.list_cameras()`), as objects or JSON; each gets the name of its configuration.
- **Configure them with JSON:** name, model, serial number, hardware settings and software filters per camera (`cameras.json`); templates for every model (`opennve template <model>`).
- **Open one or several cameras** from the configuration, several cameras with clock synchronization.
- **Read events** live (x, y, polarity, time), with software filters: polarity, background activity (noise) filter — changeable while the camera runs.
- **USB driver:** check and install the WinUSB driver on Windows (`opennve driver --install`), udev rules on Linux.

### Recordings
- **RAW files** (`.raw`, Prophesee EVT 2.0, 2.1, 3.0) and **HDF5 files** (`.h5`, `.hdf5`: columns x, y, p, t and an optional label l; column, compound, matrix and pandas layouts, time units, sorting) — read with the same interface as a live camera.
- **Information and statistics** of a recording: `opennve recording <file>`.
- **Write** events with labels as HDF5 (`write_hdf5`, `Hdf5Writer`).

### Time windows, frames and tiles
- **Time windows:** collect events in windows by duration and step (default 40 ms) or as several overlapping windows per block, aligned to the block or the first event.
- **Frames** per window: time surface (linear or exponential decay), binary or count, per polarity; `frames.json` with several named frame configurations.
- **Tiles:** split frames into tiles (for instance 80 × 80 px), keep only tiles with enough events (`min_events`, `min_events_per_ms`).
- Display frames and tiles as images (`frame.to_image`).

### Neural networks (agents)
- **Run models** on the tiles or the whole frame of every window: TensorFlow Lite (`.tflite`, LiteRT) and TensorFlow / Keras (`.keras`, `.h5`, SavedModel), configured in `agents.json`.
- **Tasks:** classification and segmentation (class map, pixels and events of the target classes, thresholds, labels), own tasks can be registered.
- **Preprocessing:** channels, padding, resizing, input quantization for int8 models — automatic.
- **Devices:** CPU, GPU (WebGPU, Metal), NPU (Intel, Qualcomm), TensorFlow Lite delegates (Coral Edge TPU), several devices at the same time (`"device": ["gpu", "cpu"]`), automatic fallback to the CPU.
- **Benchmark:** compare CPU, GPU and NPU for a model — speed and agreement of the results (`opennve benchmark`).

### Building and converting networks
- **Build networks:** U-Net (configurable depth, filters, batch normalization) and own architectures through a registry.
- **Load, save, transfer weights** between models (by name or U-Net layout), compare weights.
- **Export to TensorFlow Lite** with quantization: none, float16, dynamic, int8 (with representative data from your recordings).

### Workflows
- **Free chains of steps** in `workflows.json`: camera / recording → time windows → frames → tiles → agents → filter → label events → log / callback / write (CSV, HDF5, JSONL) or collect; own steps can be registered.
- **Multithreaded:** every step in its own threads, results in the original order; threads sized automatically to the CPU cores (`"threads": "auto"`), several agent instances in parallel.
- **Run** from Python or the command line (`opennve workflow run workflows.json`), stop with `stop`; statistics per step at the end, thread overview (`opennve workflow threads`).

### System and tools
- **System information:** operating system, processor, CPU cores, RAM, GPUs and NPUs (`opennve system`, `opennve.system_info()`).
- **Version and build number** of openNVE and its dependencies (`opennve version`).
- **Check configuration files** (`opennve check`).
- Wheels for Windows, Linux (x64, ARM64 / Raspberry Pi 5) and macOS with the camera driver included.

## Supported hardware

### Event cameras (EVS / DVS)

All devices of [neuromorphic_drivers](https://github.com/fr3ak2402/neuromorphic-drivers) (USB):

| `model` | Camera | Sensor type | Resolution |
|---|---|---|---|
| `SE1-S4` | SENSING SE1-S4-USB | EVS | 1280 × 720 |
| `EVK4` | Prophesee EVK4 | EVS | 1280 × 720 |
| `EVK3-HD` | Prophesee EVK3 HD | EVS | 1280 × 720 |
| `EVK1-GEN4` | Prophesee EVK1 Gen4 | EVS | 1280 × 720 |
| `TRITON` | Lucid Triton | EVS | 1280 × 720 |
| `CENTURYARKS-VGA` | CenturyArks VGA | EVS | 640 × 480 |
| `DVXPLORER` | iniVation DVXplorer | DVS | 640 × 480 |
| `DAVIS346` | iniVation DAVIS 346 | DVS | 346 × 260 |

EVS = event-based vision sensor (Prophesee / Sony), DVS = dynamic vision sensor (iniVation). openNVE
reads the events (x, y, polarity, time); `opennve models` lists the models, `opennve template
<model>` shows their settings. Recordings work without a camera: `.raw` (EVT 2.0, 2.1, 3.0) and
HDF5 files.

### Operating systems

| Operating system | Wheel | Camera access | Tested in CI |
|---|---|---|---|
| Windows 10 / 11 x64 | `win_amd64` | WinUSB driver: `opennve driver --install` | Windows Server 2025 |
| Linux x64 (glibc 2.28 or newer: Ubuntu 20.04+, Debian 10+, ...) | `manylinux_2_28_x86_64` | udev rules: `neuromorphic-drivers-install-udev-rules` | Ubuntu 24.04, 26.04 |
| Linux ARM64 (Raspberry Pi 5 with Raspberry Pi OS 64-bit, Jetson, ...) | `manylinux_2_28_aarch64` | udev rules | Ubuntu 24.04 ARM64 |
| macOS 11+ Apple silicon | `macosx_11_0_arm64` | no driver needed | macOS 26 |
| macOS Intel | `macosx_10_12_x86_64` | no driver needed | macOS 15 (no LiteRT: `.tflite` models need the `tensorflow` extra) |

Python 3.14 and 3.15 (LiteRT, and with it GPU / NPU agents, is not yet available for 3.15).
`opennve system` shows the operating system, processor, RAM, GPUs and NPUs of a computer.

### GPU and NPU

Agents with TensorFlow Lite models run on the CPU (default), the GPU or an NPU (`"device"` in
`agents.json`, see [GPU and NPU](docs/accelerators.md)):

| | Windows x64 | Linux x64 | Linux ARM64 | macOS Apple silicon |
|---|---|---|---|---|
| CPU | ✓ | ✓ | ✓ | ✓ |
| GPU | WebGPU (Direct3D 12): NVIDIA, AMD, Intel | WebGPU (Vulkan): NVIDIA, AMD, Intel | WebGPU (Vulkan): Raspberry Pi 5, Mali, ... | Metal |
| NPU | Intel Core Ultra (AI Boost), extra `npu-intel` | Intel Core Ultra, extra `npu-intel` | Qualcomm (Hexagon), extra `npu-qualcomm` | – |
| TensorFlow Lite delegates | e.g. Coral Edge TPU (`"delegate"`) | ✓ | ✓ | ✓ |

`"device": ["gpu", "cpu"]` uses the GPU and the remaining CPU cores at the same time.
Not supported: the Apple Neural Engine, Hailo accelerators (Raspberry Pi AI Kit / HAT).
Keras / SavedModel agents use the GPU through TensorFlow (CUDA on Linux, `tensorflow-metal` on
macOS). `opennve benchmark` compares CPU, GPU and NPU on your computer.

## Installation

Download the wheel for your platform and Python version from the [releases](https://github.com/fr3ak2402/openNVE/releases) (or the artifacts of the Build workflow) and install it. It contains neuromorphic_drivers with the SE1-S4 driver; pip installs the other dependencies (numpy, OpenCV, LiteRT for TensorFlow Lite models) automatically. Python 3.14 or newer is required. TensorFlow for the neural networks is an optional extra: `pip install "opennve-...whl[tensorflow]"` (see [Installation](docs/installation.md)).

```sh
pip uninstall -y neuromorphic_drivers   # if a separate neuromorphic_drivers is installed
pip install opennve-1.1.0-cp314-cp314-win_amd64.whl
```

| File name contains | Platform |
|---|---|
| `win_amd64` | Windows x64 |
| `manylinux_2_28_x86_64` | Linux x64 |
| `manylinux_2_28_aarch64` | Linux ARM64 (Raspberry Pi 5 with a 64-bit OS) |
| `macosx_..._arm64` | macOS Apple Silicon |
| `macosx_..._x86_64` | macOS Intel |

`cp314` = Python 3.14, `cp315` = Python 3.15.

On Windows, install the camera driver once per computer: `opennve driver --install` (or `opennve.ensure_driver("SE1-S4")` in Python).
On Linux: `neuromorphic-drivers-install-udev-rules`.

For development (source checkout), install neuromorphic_drivers from the fork first, then `pip install -e ".[test]"`.

## Command line

```sh
opennve list                       # connected cameras
opennve list --json --config examples/cameras.json
opennve template SE1-S4 --serial 05110634 --name left > left.json
opennve check examples/cameras.json
opennve models
opennve recording recording.raw      # information and statistics of a .raw or HDF5 (.h5) recording
```

## Camera configuration (JSON)

```json
{
    "name": "left",
    "model": "SE1-S4",
    "serial": "05110634",
    "settings": {
        "biases": {"diff_on": 20, "diff_off": 20},
        "clock": "internal",
        "rate_limiter": null,
        "enable_external_trigger": true
    },
    "filter": {
        "polarity": "both",
        "background_activity": {"enabled": true, "time_us": 5000, "min_neighbors": 1}
    },
    "output": {
        "fields": ["x", "y", "p", "t"]
    },
    "window": {
        "duration_ms": 40
    }
}
```

| Field | Description |
|---|---|
| `name` | Name of the camera in your setup (unique per file) |
| `model` | `SE1-S4`, `EVK4`, `EVK3-HD`, ... (`opennve models`) |
| `serial` | Serial number, or `null` for the first camera of this model (`opennve list`) |
| `settings` | Hardware settings, passed to the camera. Omitted fields keep their default value. `opennve template <model>` shows all fields of a model |
| `filter.polarity` | `both`, `on` or `off` |
| `output.fields` | Fields of the output events, in this order (any subset): `x`, `y`, `p` = polarity (1 = ON, 0 = OFF), `t` = timestamp in µs. Default `["x", "y", "p", "t"]`. The camera name is stored once per packet (`packet.camera`), not in every event |
| `window.duration_ms` | Duration of the time windows used by `camera.windows()`, in ms (0.01 to 60000). Default 40 ms (also without a `window` section); `"window": null` disables windows. Windows are aligned to the camera clock ([0, 40) ms, [40, 80) ms, ...), so synchronized cameras produce windows with the same boundaries |
| `filter.background_activity` | Removes isolated events (noise, hot pixels): an event is kept if `min_neighbors` of its 8 neighbors fired within `time_us` µs |

SE1-S4 / EVK4 settings: `biases` (offsets from the factory values, `camera.biases_bounds()` gives the allowed range), `clock` (`internal`, `internal_with_output_enabled` = synchronization primary, `external` = synchronization secondary), `rate_limiter` (`{"reference_period_us": 200, "maximum_events_per_period": 4000}` or `null`), `enable_external_trigger`, `enable_output`, and the bit masks `x_mask`, `y_mask`, `pixel_mask`.

A file contains one camera object, a list of cameras, or `{"cameras": [...]}` (see [examples/cameras.json](examples/cameras.json)).

## Python API

```python
import opennve

# Connected cameras
for camera in opennve.list_cameras():
    print(camera.model, camera.serial, camera.connection, camera.available)
print(opennve.list_cameras_json())

# Configuration from JSON (string, dict or file)
config = opennve.CameraConfig.from_dict({"name": "left", "model": "SE1-S4", "serial": "05110634"})
configs = opennve.load_configs("examples/cameras.json")
opennve.save_configs("cameras.json", configs)

# One camera
with opennve.open_camera(path="examples/cameras.json", name="left") as camera:
    print(camera.info())  # configuration, resolution, bias bounds
    for packet in camera.events():  # filtered events
        print(packet.camera)  # camera name, once per packet: "left"
        events = packet.events  # numpy array in the output format, fields x, y, p, t
        print(events["x"], events["t"])
        print(packet.rows()[:3])  # [(412, 233, 1, 1003), ...]
        print(packet.to_csv(header=True, camera=True), end="")  # # camera: left / x,y,p,t / 412,233,1,1003
        # packet.polarity_events: original filtered events (fields t, x, y, on)

        # Change settings while running (only the given fields change)
        camera.update_configuration({"biases": {"diff_on": 30}})
        camera.update_filter({"background_activity": {"enabled": True, "time_us": 2000}})
        camera.update_output({"fields": ["t", "x", "y", "p"]})
        break

# Time windows, for instance to build frames
with opennve.open_camera(path="examples/cameras.json", name="left") as camera:
    for window in camera.windows():  # duration from "window", or camera.windows(duration_ms=20)
        print(window.camera, window.start_t, window.end_t, len(window))
        events = window.events  # all events of the window, output format (x, y, p, t)
        frame = window.to_frame("count")  # "count", "polarity", "binary" or "color"
        break

# Frames and tiles with named frame configurations (separate file, see docs/frames.md)
tiles80 = opennve.load_frame_config("examples/frames.json", "tiles80")
with opennve.open_camera(path="examples/cameras.json", name="left") as camera:
    for frame in camera.frames(tiles80):  # one frame per time window, 80 x 80 tiles
        print(frame.name, frame.data.shape, [tile.id for tile in frame.tiles])
        break

# Several cameras (synchronization secondaries are started first)
cameras = opennve.open_cameras(configs)
```

`EventCamera`, `SE1S4Driver`, `EVK4Driver` and `EVK3HDDriver` keep the previous interface (`packets()`, `read_packet()`, `resolution`, `serial`, `update_configuration()`) and accept `filter=` and `name=`.

## Examples

```sh
python examples/list_cameras.py
python examples/viewer.py examples/cameras.json   # Q: exit, F: filter on/off
python examples/frames.py examples/cameras.json left   # one frame per time window, M: frame mode
python examples/raw_player.py recording.raw             # play a .raw or .h5 recording as frames
python examples/tiles.py examples/cameras.json --camera left --frame tiles80   # frames and tiles (examples/frames.json)
python examples/agents.py examples/cameras.json --camera left                  # TensorFlow models on tiles (examples/agents.json)
python examples/networks.py recording.raw --out models                         # build a U-Net, convert it to TensorFlow Lite
opennve workflow run examples/workflows.json                                   # run the default workflow (examples/workflows.json)
```

## Wheels

GitHub Actions ([.github/workflows/build.yml](.github/workflows/build.yml)) builds neuromorphic_drivers from [fr3ak2402/neuromorphic-drivers](https://github.com/fr3ak2402/neuromorphic-drivers) (`NEUROMORPHIC_DRIVERS_REF`, default `main`) and embeds it into the openNVE wheels with [tools/bundle_wheel.py](tools/bundle_wheel.py): one wheel per platform (Windows x64, Linux x64/ARM64, macOS Apple Silicon/Intel) and Python version (3.14, 3.15). Each wheel is tested before upload. The wheels are artifacts of each run; for tags starting with `v` (`git tag v1.1.0 && git push --tags`) they are attached to a GitHub release.

neuromorphic_drivers is distributed under the MIT License, its license is included in the wheels (`opennve-*.dist-info/licenses/neuromorphic_drivers/LICENSE`).

## Tests

```sh
pip install -e ".[test]"
pytest
```
