# openNVE

Open Neuromorphic Vision Engine: a Python library to find, configure and read event cameras.

- List all connected cameras (as objects or JSON)
- Describe each camera with a JSON configuration: name, model, serial number, hardware settings and software filters
- Open one or several cameras from that configuration (with clock synchronization)
- Read filtered events

Supported models: SENSING SE1-S4-USB, Prophesee EVK4, EVK3 HD, EVK1 Gen4, iniVation DAVIS 346, DVXplorer, CenturyArks VGA, Lucid Triton (all devices of [neuromorphic_drivers](https://github.com/fr3ak2402/neuromorphic-drivers)).

## Installation

```sh
# 1. neuromorphic_drivers with the SE1-S4 driver (wheel from fr3ak2402/neuromorphic-drivers, Actions artifacts)
pip install neuromorphic_drivers-0.19.2-cp314-cp314-win_amd64.whl

# 2. openNVE
pip install -e ".[viewer]"
```

On Windows, install the camera driver once per computer: `neuromorphic-drivers-install-windows-driver`.
On Linux: `neuromorphic-drivers-install-udev-rules`.

## Command line

```sh
opennve list                       # connected cameras
opennve list --json --config examples/cameras.json
opennve template SE1-S4 --serial 05110634 --name left > left.json
opennve check examples/cameras.json
opennve models
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
        events = packet.polarity_events  # numpy array with fields t, x, y, on
        print(packet.camera, packet.raw_event_count, len(events))

        # Change settings while running (only the given fields change)
        camera.update_configuration({"biases": {"diff_on": 30}})
        camera.update_filter({"background_activity": {"enabled": True, "time_us": 2000}})
        break

# Several cameras (synchronization secondaries are started first)
cameras = opennve.open_cameras(configs)
```

`EventCamera`, `SE1S4Driver`, `EVK4Driver` and `EVK3HDDriver` keep the previous interface (`packets()`, `read_packet()`, `resolution`, `serial`, `update_configuration()`) and accept `filter=` and `name=`.

## Examples

```sh
python examples/list_cameras.py
python examples/viewer.py examples/cameras.json   # Q: exit, F: filter on/off
```

## Wheels

GitHub Actions ([.github/workflows/build.yml](.github/workflows/build.yml)) tests openNVE on Linux and Windows against neuromorphic_drivers from the fr3ak2402 fork and builds `opennve-<version>-py3-none-any.whl` (works on every platform). The wheel is an artifact of each run; for tags starting with `v` (for instance `git tag v0.1.0 && git push --tags`) it is also attached to a GitHub release.

```sh
pip install opennve-0.1.0-py3-none-any.whl
```

## Tests

```sh
pip install -e ".[test]"
pytest
```
