# Python API

```python
import opennve
```

All names described here are available directly as `opennve.<name>`.

- [Finding cameras](#finding-cameras)
- [Configuration](#configuration)
- [USB driver](#usb-driver)
- [Opening cameras](#opening-cameras)
- [EventCamera](#eventcamera)
- [CameraPacket](#camerapacket)
- [EventWindow](#eventwindow)
- [Filter, output, window](#filter-output-window)
- [Frames and tiles](#frames-and-tiles)
- [RAW files](#raw-files)
- [Models](#models)
- [Helper functions](#helper-functions)

---

## Finding cameras

### `list_cameras(configs=()) -> list[CameraInfo]`

Lists the connected cameras, sorted by model and serial number. If configurations are passed, each camera gets the name of the configuration with the same model and serial number.

```python
for camera in opennve.list_cameras(opennve.load_configs("cameras.json")):
    print(camera.name, camera.model, camera.serial, camera.available, camera.error)
```

Raises `RuntimeError` if the system does not allow USB access.

### `list_cameras_json(configs=(), indent=4) -> str`

Same as `list_cameras`, as a JSON array:

```json
[
    {
        "model": "SE1-S4",
        "device_name": "SENSING SE1-S4-USB",
        "serial": "05110634",
        "connection": "USB 3.0 SuperSpeed (5.0 Gb/s)",
        "address": "1:8",
        "error": null,
        "name": "left",
        "available": true
    }
]
```

### `CameraInfo`

| Attribute / method | Description |
|---|---|
| `model` | Model, for instance `"SE1-S4"` |
| `device_name` | Device name, for instance `"SENSING SE1-S4-USB"` |
| `serial` | Serial number, or `None` if the camera could not be read |
| `connection` | USB connection, for instance `"USB 3.0 SuperSpeed (5.0 Gb/s)"` |
| `address` | Bus address |
| `error` | Error message (for instance missing permissions) or `None` |
| `name` | Name from the matching configuration, or `None` |
| `available` | `True` if the camera can be used |
| `to_dict()` | As a dictionary (JSON-compatible) |
| `default_config()` | `CameraConfig` with default values for this camera |

---

## Configuration

The JSON format is described in [Configuration](configuration.md).

### `CameraConfig`

```python
config = opennve.CameraConfig(model="SE1-S4", serial="05110634", name="left")
config = opennve.CameraConfig.from_dict({...})
config = opennve.CameraConfig.from_json('{"model": "SE1-S4"}')
```

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Camera name |
| `model` | `CameraModel` | Model (`config.model.id` = `"SE1-S4"`) |
| `serial` | `str \| None` | Serial number |
| `settings` | `neuromorphic_drivers` `Configuration` | Hardware settings (dataclass of the model) |
| `filter` | `FilterConfig` | Software filters |
| `output` | `OutputConfig` | Output format |
| `window` | `WindowConfig \| None` | Time windows (40 ms by default) |
| `frame` | `FrameConfig` | Frames and tiles |

| Method | Description |
|---|---|
| `from_dict(data)` / `from_json(text)` | From a dictionary / JSON text, all values are validated (`ValueError`) |
| `to_dict()` / `to_json(indent=4)` | As a dictionary / JSON text |
| `copy(**changes)` | Copy with changes, for instance `config.copy(name="right", serial="05110636")` |

### Files

| Function | Description |
|---|---|
| `load_configs(path) -> list[CameraConfig]` | Loads a file with one camera, a list or `{"cameras": [...]}` |
| `load_config(path, name=None) -> CameraConfig` | Loads one camera, selected by `name` if the file has several |
| `save_configs(path, configs)` | Saves one `CameraConfig` (as an object) or several (as `{"cameras": [...]}`) |

---

## USB driver

On Windows, a camera can only be used once it has the WinUSB driver. These functions install it from your program (same as Zadig, one administrator prompt, can take a few minutes). See also [Installation](installation.md#windows).

### `ensure_driver(model=None) -> bool`

Installs the driver only if a connected camera needs it. Returns `True` if all connected cameras (of `model`, for instance `"SE1-S4"`) can be used, also if none is connected. Call it once when your program starts:

```python
if not opennve.ensure_driver("SE1-S4"):
    print(opennve.driver_status("SE1-S4"))
```

Cameras that use another driver (for instance the vendor's driver) are not changed, `ensure_driver` then returns `False`.

### `install_driver(model=None, force=False) -> bool`

Installs WinUSB for the connected cameras without a driver. With `force=True`, an existing driver (for instance Metavision's) is replaced as well, which may break the vendor's software. Returns `True` if the cameras can be used afterwards.

### `driver_status(model=None) -> list[DriverStatus]`

Driver state of the connected cameras, without changing anything.

| Attribute | Description |
|---|---|
| `device` | Device type, for instance `"SENSING SE1-S4-USB"` |
| `usb_id` | USB ID, for instance `"04B4:00C4"` (Windows) |
| `driver` | Current Windows driver, for instance `"WinUSB"`, or `None` |
| `ready` | `True` if the camera can be used |
| `message` | What to do if the camera is not ready |

### Linux and macOS

On Linux, access is granted by udev rules, which require `sudo`: `install_driver` and `ensure_driver` print the command (`neuromorphic-drivers-install-udev-rules`) and return `False` if a camera is not accessible. On macOS, no driver is required and both return `True`.

### When opening cameras

`open_camera(..., install_driver=True)` and `open_cameras(..., install_driver=True)` call `ensure_driver` for the models in the configuration before opening, and raise `RuntimeError` if the driver is still not ready.

---

## Opening cameras

### `open_camera(config=None, *, path=None, name=None, iterator_timeout=0.02, install_driver=False) -> EventCamera`

Opens a camera from a `CameraConfig`, a dictionary, a JSON string or a file.

```python
camera = opennve.open_camera({"model": "SE1-S4", "serial": "05110634"})
camera = opennve.open_camera(path="cameras.json", name="left")

with opennve.open_camera(path="cameras.json", name="left") as camera:
    ...  # closed automatically at the end
```

`iterator_timeout` (seconds) is the maximum waiting time when reading. After that, `read()` returns `None` instead of blocking. With `install_driver=True`, a missing USB driver is installed first ([USB driver](#usb-driver)).

### `open_cameras(configs, *, iterator_timeout=0.02, secondary_start_delay=1.0, install_driver=False) -> dict[str, EventCamera]`

Opens several cameras and returns `{name: camera}`. Cameras with `"clock": "external"` are opened first, the others after `secondary_start_delay` seconds. If one camera fails, the cameras opened so far are closed. See [Multiple cameras](multiple-cameras.md).

---

## EventCamera

```python
camera = opennve.EventCamera(model="SE1-S4", serial="05110634", name="left",
                             configuration=None, filter=None, output=None, window=None,
                             iterator_timeout=0.02)
camera = opennve.EventCamera.from_config(config)
```

The camera is opened when the object is created. A camera must only be read by **one** thread at a time.

Shortcuts with a fixed model: `SE1S4Driver(serial=None, **kwargs)`, `EVK4Driver(...)`, `EVK3HDDriver(...)`.

### Properties

| Property | Description |
|---|---|
| `name` | Name from the configuration |
| `model` | Model, for instance `"SE1-S4"` |
| `serial` | Serial number of the open camera |
| `resolution` | `(width, height)`, for instance `(1280, 720)` |
| `config` | Current `CameraConfig` (the serial number is filled in when the camera is opened) |
| `configuration` | Current hardware settings (`config.settings`) |
| `is_open` | `True` while the camera is open |
| `device` | The underlying `neuromorphic_drivers` device (for special cases) |

### Reading

| Method | Description |
|---|---|
| `events()` | Generator: yields `CameraPacket` objects (filtered). Timeouts are skipped |
| `read()` | Reads one packet: `CameraPacket`, or `None` on timeout |
| `windows(duration_ms=None)` | Generator: yields complete `EventWindow` objects (duration from `window`, 40 ms by default) |
| `read_windows(duration_ms=None)` | Reads one packet and returns the windows it completed (often `[]`) |
| `frames(config=None, duration_ms=None)` | Generator: one `Frame` per time window, built with a frame configuration ([Frames and tiles](frames.md)) |
| `read_frames(config=None, duration_ms=None)` | Frames completed by the next packet (often `[]`) |
| `frame_sets(configs, duration_ms=None)` | Generator: `{name: Frame}` per time window, one frame per frame configuration |
| `read_frame_sets(configs, duration_ms=None)` | Frame sets completed by the next packet (often `[]`) |
| `packets()` / `read_packet()` | Unfiltered raw data `(status, packet)` from `neuromorphic_drivers` |

### Information

| Method | Description |
|---|---|
| `info()` | Configuration, resolution and allowed bias ranges as a dictionary |
| `biases_bounds()` | `{"diff_on": {"minimum": -80, "maximum": 145}, ...}` or `None` |

### Changing settings while running

| Method | Description |
|---|---|
| `update_configuration(settings)` | Changes the hardware settings. A dictionary (only the given fields change) or the model's `Configuration` |
| `update_filter(filter)` | Replaces the software filters (`FilterConfig` or dictionary) |
| `update_output(output)` | Replaces the output format, for instance `{"fields": ["t", "x", "y"]}` |
| `update_window(window)` | Changes the window duration, for instance `{"duration_ms": 20}`, or `None`. The current window restarts |
| `apply_config(config)` | Everything at once (settings, filter, output, window, name). Model and serial number must match |

```python
camera.update_configuration({"biases": {"diff_on": 30, "diff_off": 30}})
camera.update_filter({"background_activity": {"enabled": True, "time_us": 2000}})
```

### Closing

`close()` or a `with` block. Afterwards, `is_open` is `False`.

---

## CameraPacket

A packet of filtered events from one camera, returned by `camera.events()` or `camera.read()`.

| Attribute / method | Description |
|---|---|
| `camera` | Camera name (once per packet) |
| `events` | Events in the configured output format: NumPy array with the fields of `output.fields`, for instance `x, y, p, t` |
| `rows()` | Events as a list of tuples: `[(412, 233, 1, 1003), ...]` |
| `to_csv(header=False, delimiter=",", camera=False)` | Events as CSV text. `camera=True` writes `# camera: <name>` as the first line |
| `polarity_events` | Filtered events in the original format (`t, x, y, on`), or `None` |
| `trigger_events` | Trigger events (external input), or `None` |
| `raw_event_count` | Number of events **before** the filter |
| `status` | `neuromorphic_drivers` status, for instance `status.ring.backlog` (backlog) and `status.ring.current_t` (camera time) |
| `packet` | The original `neuromorphic_drivers` packet |

```python
for packet in camera.events():
    events = packet.events
    on = events[events["p"] == 1]
    print(packet.camera, len(events), events["t"][-1] if len(events) else None)
```

`rows()` and `to_csv()` create one Python object per event. For high event rates, work directly with the NumPy array `events`.

---

## EventWindow

The events of one time window, returned by `camera.windows()` or `camera.read_windows()`. See [Time windows and frames](time-windows.md).

| Attribute / method | Description |
|---|---|
| `camera` | Camera name |
| `start_t`, `end_t` | Window boundaries in µs of camera time (`start_t` included, `end_t` excluded) |
| `duration_ms` | Window duration in ms |
| `width`, `height` | Sensor size |
| `len(window)` | Number of events |
| `events` | Events in the output format (as `CameraPacket.events`) |
| `polarity_events` | Events in the original format (`t, x, y, on`) |
| `rows()`, `to_csv(...)` | As for `CameraPacket` |
| `to_frame(mode="count")` | Image (height × width) built from the events: `count`, `polarity`, `binary` or `color` |
| `frame(config=None, mode=None)` | Frame tensor and tiles with the frame engine ([Frames and tiles](frames.md)) |

---

## Filter, output, window

| Class | Fields | Description |
|---|---|---|
| `FilterConfig` | `polarity`, `background_activity` | Software filters. `from_dict`, `to_dict`, `is_active` |
| `BackgroundActivityConfig` | `enabled`, `time_us`, `min_neighbors` | Noise filter settings |
| `OutputConfig` | `fields` | Output format. `format(events)` converts `t, x, y, on` to the output format, `dtype()` returns the NumPy type |
| `WindowConfig` | `duration_ms` | Window duration (40 by default). `duration_us` in microseconds |

For custom processing pipelines without `EventCamera`:

| Class | Description |
|---|---|
| `EventFilter(config, width, height)` | Applies a `FilterConfig`: `filtered = event_filter(events)` |
| `BackgroundActivityFilter(width, height, time_us, min_neighbors=1)` | Only the noise filter |
| `WindowAccumulator(duration_us, camera, width, height, output_config=None)` | Splits events into windows: `push(events, current_t=None) -> list[EventWindow]`, `reset()` |

```python
accumulator = opennve.WindowAccumulator(40_000, "left", 1280, 720)
for packet in camera.events():
    for window in accumulator.push(packet.polarity_events, packet.status.ring.current_t):
        ...
```

---

## Frames and tiles

The frame engine builds one frame tensor per time window (height × width × 2, float32) and splits it into tiles. Frame configurations are separate from the camera configurations and have names. See [Frames and tiles](frames.md) for the file format, modes, tile layout and the migration from the previous `FrameEngine`.

| Name | Description |
|---|---|
| `load_frame_configs(path)` / `load_frame_config(path, name=None)` | Frame configurations from a file (`{"frames": [...]}`) |
| `save_frame_configs(path, configs)` | Writes frame configurations |
| `camera.frames(config=None, duration_ms=None)` | Generator: one `Frame` per time window, built with the frame configuration |
| `camera.read_frames(config=None, duration_ms=None)` | Frames completed by the next packet (often `[]`) |
| `camera.frame_sets(configs, duration_ms=None)` | Generator: `{name: Frame}` per time window |
| `window.frame(config=None, mode=None, decay=None)` | Frame of one `EventWindow` |
| `window.frames(configs)` | `{name: Frame}` of one `EventWindow` |
| `FrameEngine(width, height, tile_size=None, mode="time_surface", min_events=1)` | `build(events, start_t, end_t)`, `build_window(window)`, `generate_frame(...)` (previous interface) |
| `Frame` | `name`, `data`, `tiles`, `events`, `event_count`, `start_t`, `end_t`, `split(tile_size, min_events, data)`, `tile(id)`, `tile_dicts()`, `to_image()` |
| `split_into_tiles(data, tile_size, event_counts=None, min_events=0)` | Splits any array into tiles |
| `Tile` | `id`, `row`, `col`, `x`, `y`, `width`, `height`, `event_count`, `data` |
| `FrameConfig`, `TileConfig`, `TimeSurfaceConfig` | Frame (with `name`), tile and time surface settings (`decay`: `linear` / `exponential`, `tau_us`) |

```python
frame_configs = opennve.load_frame_configs("frames.json")
tiles80 = opennve.load_frame_config("frames.json", "tiles80")

for frame in camera.frames(tiles80):
    for tile in frame.tiles:
        print(tile.id, tile.row, tile.col, tile.event_count, tile.data.shape)

# several frame configurations from the same window
for frames in camera.frame_sets(frame_configs):
    print({name: frame.data.shape for name, frame in frames.items()})

# or: whole frame first, tiles later
for frame in camera.frames(frame_configs[0]):  # "tiles": null
    frame.data = process(frame.data)
    for tile in frame.split(80):
        ...
```

---

## Agents (TensorFlow models)

Agents run TensorFlow Lite (LiteRT) and TensorFlow / Keras models on the tiles or frames of each time window. They are configured in their own file (`agents.json`). See [Agents](agents.md) for the configuration, installation of the runtimes and results.

| Name | Description |
|---|---|
| `load_agents(path)` / `load_agent(path, name=None)` | Agents of a file (models are loaded on first use) |
| `Agent(config, model=None)` / `Agent.from_model(model, **settings)` | `load()`, `predict(frame / tiles / array) -> AgentResult`, `predict_batch(batch)`, `input`, `close()` |
| `run_agents(source, agents, frame_configs=None, duration_ms=None)` | Generator: one `AgentStep` (`window`, `frames`, `results`) per time window of a camera or RAW file |
| `process_window(window, agents, frame_configs=None)` | `AgentStep` for one window |
| `AgentConfig`, `PreprocessConfig`, `PredictionConfig` | Agent settings (`load_agent_configs`, `load_agent_config`, `save_agent_configs`) |
| `AgentResult` | `predictions`, `selected`, `outputs`, `inference_ms`, `by_tile()`, `to_dict()` |
| `Prediction` | `tile`, `tile_id`, `output`, `outputs`, `class_id`, `label`, `score`, `selected` |

```python
agents = opennve.load_agents("agents.json")
for step in opennve.run_agents(camera, agents, opennve.load_frame_configs("frames.json")):
    for prediction in step["insects"].selected:
        print(prediction.tile_id, prediction.label, prediction.score)
```

---

## RAW files

`.raw` recordings (EVT 2.0, 2.1, 3.0) are read with the same interface as a live camera. See [RAW files](raw-files.md) for all options.

| Function / class | Description |
|---|---|
| `open_raw(path, config=None, *, config_path=None, name=None, **options) -> RawReader` | Opens a RAW file. The configuration provides name, filter, output format and window |
| `RawReader` | `events()`, `read()`, `windows()`, `read_windows()`, `frames(config)`, `read_frames(config)`, `frame_sets(configs)`, `read_frame_sets(configs)`, `read_all()`, `info()`, `speed`, `close()` |
| `read_raw(path, **options) -> (events, triggers)` | Reads a whole file into memory |
| `read_header(path) -> RawHeader` | Header only: `format`, `width`, `height`, `fields`, `serial` |

```python
with opennve.open_raw("recording.raw", config_path="cameras.json", name="left") as raw:
    for window in raw.windows():
        frame = window.to_frame("count")
```

---

## Models

| Name | Description |
|---|---|
| `MODELS` | All supported models (`CameraModel`) |
| `get_model(name) -> CameraModel` | Model from a short name (`"SE1-S4"`), device name or module name. `ValueError` if unknown |
| `CameraModel.id` | Short name, for instance `"SE1-S4"` |
| `CameraModel.name` | Device name, for instance `"SENSING SE1-S4-USB"` |
| `CameraModel.module` | `neuromorphic_drivers` module, for instance `nd.sensing_se1_s4` |
| `CameraModel.configuration_type` | Dataclass of the hardware settings |

---

## Helper functions

| Function | Description |
|---|---|
| `to_rows(events)` | Structured NumPy array → list of tuples |
| `to_csv(events, header=False, delimiter=",", camera=None)` | Structured NumPy array → CSV text |
| `settings_from_dict(model, data)` | JSON settings → the model's `Configuration` |
| `settings_to_dict(settings)` | `Configuration` → JSON settings |
| `__version__` | openNVE version |
| `NEUROMORPHIC_DRIVERS_VERSION` | Version of the embedded driver, for instance `"0.19.2"` |

## Errors

| Exception | When |
|---|---|
| `ValueError` | Invalid configuration (the message names the field), unknown model, configuration does not match the camera |
| `TypeError` | `Configuration` of another model passed |
| `RuntimeError` | Camera closed, not found, or USB error (from `neuromorphic_drivers`) |
