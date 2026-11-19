# Configuration (JSON)

Each camera is described by a JSON object:

```json
{
    "name": "left",
    "model": "SE1-S4",
    "serial": "05110634",
    "settings": {
        "biases": {"diff_on": 20, "diff_off": 20},
        "clock": "internal",
        "rate_limiter": null,
        "enable_external_trigger": true,
        "enable_output": true
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
    },
    "frame": {
        "width": 1280,
        "height": 720,
        "mode": "time_surface",
        "time_surface": {"decay": "linear", "tau_us": 100000},
        "tiles": {"size": 80, "min_events": 1}
    }
}
```

Only `model` is required. Missing fields keep their default value, also inside sections: `"biases": {"diff_on": 20}` only changes `diff_on`. Unknown fields and invalid values are rejected with an error message that names the field, for instance `settings.biases.diff_on must be in [-32768, 32767]`.

## File formats

A file contains either

- **one camera** (an object as above),
- **a list** of cameras (`[{...}, {...}]`), or
- **an object with `cameras`** (`{"cameras": [{...}, {...}]}`), which is how `save_configs` writes several cameras.

Names must be unique within a file. Check a file with `opennve check file.json`.

## General fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | `<model>-<serial>` | Name of the camera in your setup, for instance `left`. Available in every packet as `packet.camera` |
| `model` | string | – (required) | Camera model, see `opennve models`. Case-insensitive, `se1_s4` or `SENSING SE1-S4-USB` work as well |
| `serial` | string or `null` | `null` | Serial number **as a string** (`"05110634"`, not `5110634`). `null` opens the first camera of this model |

## `settings` – hardware

These settings are sent to the camera. The available fields depend on the model. `opennve template <model>` shows all of them with their default values.

### SE1-S4 and EVK4

| Field | Type | Default | Description |
|---|---|---|---|
| `biases` | object | all `0` | Sensor sensitivity (see below) |
| `clock` | string | `"internal"` | `internal`: own clock. `internal_with_output_enabled`: own clock, signal on the sync output (*primary*). `external`: clock from the sync input (*secondary*). See [Multiple cameras](multiple-cameras.md) |
| `rate_limiter` | object or `null` | `null` | Limits the event rate in the sensor: `{"reference_period_us": 200, "maximum_events_per_period": 4000}` allows at most 4000 events per 200 µs (20 million events/s). Excess events are dropped randomly |
| `enable_external_trigger` | `true`/`false` | `true` | Trigger input enabled, signals arrive as `packet.trigger_events` |
| `enable_output` | `true`/`false` | `true` | Sensor event output |
| `x_mask`, `y_mask`, `pixel_mask`, `mask_intersection_only` | lists / `true`/`false` | no mask | Disable columns, rows or single pixels. Bit masks, easiest to create in Python (see below). Only written to JSON if they differ from the default |

#### Biases (SE1-S4)

On the SE1-S4, biases are **offsets from the camera's factory values**: `0` means factory setting. The driver limits the values to the ranges recommended for the IMX646 (source: OpenEB). `camera.biases_bounds()` returns the allowed ranges.

| Bias | Range | Effect of a higher value |
|---|---|---|
| `diff_on` | −80 … 145 | less sensitive to **brighter** pixels, fewer ON events |
| `diff_off` | −30 … 200 | less sensitive to **darker** pixels, fewer OFF events |
| `hpf` | 0 … 120 | stronger suppression of slow brightness changes and noise |
| `fo` | −20 … 0 | higher bandwidth; lower values = less noise but slower response |
| `refr` | −20 … 235 | longer dead time after an event, fewer event bursts |
| `diff` | −25 … 23 | reference level, usually left unchanged |
| `pr`, `inv`, `reqpuy`, `reqpux`, `sendreqpdy`, `unknown_1`, `unknown_2` | 8-bit range around the factory value | internal currents, usually left unchanged |

To reduce noise, start with `diff_on` and `diff_off` at 20–40 and increase them step by step. Values that are too high make weak edges disappear.

On the EVK3 HD, biases are absolute values. `opennve template EVK3-HD` shows the defaults.

#### Masks (region of interest)

A set bit disables a column (`x_mask`), a row (`y_mask`) or a pixel (`pixel_mask`). With `"mask_intersection_only": false`, a pixel is disabled if its column **or** its row is masked. The easiest way to create masks is the driver's helper class. Example: only a 480 × 320 rectangle starting at (400, 200) stays active:

```python
import neuromorphic_drivers as nd
import opennve

mask = nd.sensing_se1_s4.RowColumnMask(set=True)           # everything disabled
mask.clear_rectangle(x=400, y=200, width=480, height=320)  # rectangle enabled again

config = opennve.CameraConfig.from_dict({
    "name": "left",
    "model": "SE1-S4",
    "settings": {
        "x_mask": list(mask.x_mask()),
        "y_mask": list(mask.y_mask()),
        "mask_intersection_only": False,
    },
})
opennve.save_configs("left.json", config)  # the masks are now in the JSON file
```

For the EVK4, the class is `nd.prophesee_evk4.RowColumnMask`. Masked pixels produce no events at all, which also reduces the USB load.

### Other models

| Model | Fields in `settings` |
|---|---|
| `EVK3-HD` | `biases`, `rate_limiter` |
| `DAVIS346` | `biases`, `region_of_interest`, `activity_filter`, `skip_events_every`, `polarity_filter`, `exposure_us`, `frame_interval_us` |
| others | `opennve template <model>` |

## `filter` – software filters

The filters run on the computer after the events have been read. They apply to `packet.events` and to the time windows.

| Field | Type | Default | Description |
|---|---|---|---|
| `polarity` | string | `"both"` | `both`: all events. `on`: only brighter pixels. `off`: only darker pixels |
| `background_activity.enabled` | `true`/`false` | `false` | Noise filter on/off |
| `background_activity.time_us` | number | `5000` | Time window in µs (1 … 1,000,000). Smaller = stronger filtering |
| `background_activity.min_neighbors` | number | `1` | Required active neighbor pixels (1 … 8). Larger = stronger filtering |

**Noise filter (*background activity filter*):** an event is kept only if at least `min_neighbors` of its 8 neighbor pixels also had an event within the last `time_us` microseconds. Real edges trigger neighboring pixels at almost the same time, noise and hot pixels do not. In tests with synthetic data, about 10 % of the noise remained while a moving edge was fully preserved. The filter processes about 9 million events/s.

## `output` – output format

| Field | Type | Default | Description |
|---|---|---|---|
| `fields` | list | `["x", "y", "p", "t"]` | Fields of the events, in this order. Any subset, each field at most once. A string such as `"t, x, y"` is accepted as well |

| Field | Type | Description |
|---|---|---|
| `x` | uint16 | Pixel column (0 … width−1) |
| `y` | uint16 | Pixel row (0 … height−1) |
| `p` | uint8 | Polarity: `1` = brighter (ON), `0` = darker (OFF) |
| `t` | uint64 | Camera timestamp in µs |

The **camera name** is not an event field. It is stored once per packet or window as `packet.camera` / `window.camera`. For CSV output, `to_csv(camera=True)` writes it once as the first line.

## `window` – time windows

| Field | Type | Default | Description |
|---|---|---|---|
| `duration_ms` | number | `40` | Duration of the time windows in ms (0.01 … 60,000) |

- Without a `window` section, **40 ms** is used.
- `"window": null` disables time windows. `camera.windows()` then requires `duration_ms` as an argument.

See [Time windows and frames](time-windows.md) for details.

## `frame` – frames and tiles

| Field | Type | Default | Description |
|---|---|---|---|
| `width`, `height` | number or `null` | `null` (sensor size) | Frame size in pixels; events outside are ignored |
| `mode` | string | `"time_surface"` | `time_surface` (time surfaces), `binary` or `count` |
| `time_surface` | object | `{"decay": "linear", "tau_us": 100000}` | Time surface decay: `linear` or `exponential` with the time constant `tau_us` (µs) |
| `tiles` | object or `null` | `null` | `{"size": 80, "min_events": 1}`: split each frame into 80 × 80 tiles, skip tiles with fewer than `min_events` events (`0` = all tiles) |

The frames are built from the time windows (`window`) by `camera.frames()`. Details in [Frames and tiles](frames.md).

## Example: two synchronized cameras

```json
{
    "cameras": [
        {
            "name": "left",
            "model": "SE1-S4",
            "serial": "05110634",
            "settings": {"biases": {"diff_on": 20, "diff_off": 20}, "clock": "internal_with_output_enabled"},
            "filter": {"background_activity": {"enabled": true}}
        },
        {
            "name": "right",
            "model": "SE1-S4",
            "serial": "05110636",
            "settings": {"biases": {"diff_on": 20, "diff_off": 20}, "clock": "external"},
            "filter": {"background_activity": {"enabled": true}}
        }
    ]
}
```

## Configuration in Python

```python
import opennve

config = opennve.CameraConfig.from_dict({"model": "SE1-S4", "serial": "05110634", "name": "left"})
config = opennve.CameraConfig.from_json(text)
configs = opennve.load_configs("cameras.json")
config = opennve.load_config("cameras.json", "left")

print(config.to_json())
opennve.save_configs("cameras.json", configs)  # several cameras
opennve.save_configs("left.json", config)      # one camera

right = config.copy(name="right", serial="05110636")
```
