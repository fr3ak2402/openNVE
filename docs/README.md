# openNVE documentation

openNVE (Open Neuromorphic Vision Engine) is a Python library for event cameras. It finds connected cameras, configures them with JSON, reads filtered events from cameras or `.raw` recordings and groups them into time windows and frames.

| Page | Contents |
|---|---|
| [Installation](installation.md) | Installing the wheel, camera drivers on Windows and Linux |
| [Quick start](quickstart.md) | From a connected camera to the first frame in five minutes |
| [Configuration (JSON)](configuration.md) | All fields of the camera configuration: hardware, filters, output format, time windows |
| [Python API](api.md) | Reference of all functions, classes and methods |
| [Time windows and frames](time-windows.md) | Collecting events in windows (40 ms by default) and building images |
| [Frames and tiles](frames.md) | Frame engine: time surface / binary / count tensors per window, split into tiles (for instance 80 × 80 px) |
| [Multiple cameras](multiple-cameras.md) | Reading several cameras at once and synchronizing them |
| [RAW files](raw-files.md) | Reading `.raw` recordings (EVT 2.0, 2.1, 3.0) like a live camera |
| [Command line](cli.md) | `opennve list`, `template`, `check`, `models` |
| [Troubleshooting](troubleshooting.md) | Camera not found, noise, high event rates |
| [Development](development.md) | Tests, wheel builds, releases, embedded driver |

## Overview

```text
           ┌────────────────────────────────────────────────────────────┐
JSON  ───► │ CameraConfig: name, model, serial, settings, filter,       │
           │               output, window, frame                        │
           └──────────────────────────┬─────────────────────────────────┘
                                      ▼
Camera ──► EventCamera ──► Filter ──► CameraPacket  (events in the format x, y, p, t)
 (USB)     (neuromorphic_drivers)     └──► EventWindow (40 ms) ──► Frame (tensor) ──► Tiles (80 × 80)
.raw  ───► RawReader ─────┘
```

- **Find cameras:** `opennve.list_cameras()`
- **Configure:** one JSON file per setup, with one entry per camera
- **Read:** `camera.events()` yields packets of filtered events, `camera.windows()` yields time windows
- **Recordings:** `opennve.open_raw("recording.raw")` reads `.raw` files with the same interface
- **Frames:** `camera.frames()` yields one frame tensor per window (time surface, binary or count), optionally split into tiles; `window.to_frame(...)` builds simple images

## Supported cameras

| Model (`model`) | Device |
|---|---|
| `SE1-S4` | SENSING SE1-S4-USB (Sony IMX646, 1280 × 720) |
| `EVK4` | Prophesee EVK4 (Sony IMX636, 1280 × 720) |
| `EVK3-HD` | Prophesee EVK3 HD |
| `EVK1-GEN4` | Prophesee EVK1 Gen4 |
| `DAVIS346` | iniVation DAVIS 346 |
| `DVXPLORER` | iniVation DVXplorer |
| `CENTURYARKS-VGA` | CenturyArks VGA |
| `TRITON` | Lucid Triton |

The hardware access is provided by [neuromorphic_drivers](https://github.com/fr3ak2402/neuromorphic-drivers) (fork with the SE1-S4 driver), which is already included in the openNVE wheels.
