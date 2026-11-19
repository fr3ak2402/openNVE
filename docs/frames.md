# Frames and tiles

The frame engine turns each time window (40 ms by default, see [Time windows and frames](time-windows.md)) into a frame tensor and, optionally, splits it into tiles, for instance 80 × 80 pixels for an agent or a neural network. The events of the window stay available next to the frame.

It works the same for live cameras and [RAW files](raw-files.md).

## Frame configurations

Frame configurations are separate from the [camera configurations](configuration.md). A frame configuration file holds any number of **named** frame configurations, in the same way a camera file holds several cameras. Each camera or RAW file can then be combined with any frame configuration, and one time window can be turned into several frames at once.

`examples/frames.json`:

```json
{
    "frames": [
        {
            "name": "full",
            "width": 1280,
            "height": 720,
            "mode": "time_surface",
            "time_surface": {"decay": "linear", "tau_us": 100000},
            "tiles": null
        },
        {
            "name": "tiles80",
            "width": 1280,
            "height": 720,
            "mode": "time_surface",
            "time_surface": {"decay": "linear", "tau_us": 100000},
            "tiles": {"size": 80, "min_events": 1}
        },
        {
            "name": "exponential",
            "mode": "time_surface",
            "time_surface": {"decay": "exponential", "tau_us": 20000},
            "tiles": {"size": 64, "min_events": 5}
        },
        {
            "name": "binary",
            "width": 640,
            "height": 360,
            "mode": "binary",
            "tiles": null
        }
    ]
}
```

A file contains one frame configuration (an object), a list, or `{"frames": [...]}` (written by `save_frame_configs`). Names must be unique within a file; configurations without a name are called `frame1`, `frame2`, … in the order of the file. `opennve check frames.json` validates a file and `opennve frame-template --name tiles80 --tiles 80` prints a new one (see [Command line](cli.md)).

```python
import opennve

frame_configs = opennve.load_frame_configs("frames.json")            # list of FrameConfig
tiles80 = opennve.load_frame_config("frames.json", "tiles80")       # one configuration by name
small = tiles80.copy(name="tiles80_small", width=640, height=360)  # changed copy
opennve.save_frame_configs("frames.json", [*frame_configs, small])
```

Configurations can also be created in code: `opennve.FrameConfig(name="tiles80", tiles=opennve.TileConfig(size=80))`, or passed as a dictionary or JSON string wherever a frame configuration is expected.

The cameras and the frames may also share one file, `{"cameras": [...], "frames": [...]}`: `load_configs` reads the cameras and `load_frame_configs` the frames.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string or `null` | `null` | Name of the configuration, used to select it and as `frame.name`. Must be unique within a file |
| `width`, `height` | number or `null` | `null` | Frame size in pixels. `null` = sensor size (1280 × 720 for the SE1-S4). Events outside the frame are ignored (no scaling) |
| `mode` | string | `"time_surface"` | `time_surface` to use time surfaces, or `binary` / `count` (see below) |
| `time_surface.decay` | string | `"linear"` | `linear` or `exponential` (only used with `"mode": "time_surface"`) |
| `time_surface.tau_us` | number | `100000` | Time constant of the exponential decay in µs |
| `tiles` | object or `null` | `null` | Tile grid. `null` = no tiles, only the whole frame. A number such as `80` is short for `{"size": 80}` |
| `tiles.size` | number | `80` | Tile size in pixels (square tiles) |
| `tiles.min_events` | number | `1` | Only tiles with at least this many events are returned. `1` skips empty tiles, `0` returns all tiles |

## Modes

Every frame has the shape **height × width × 2**, `float32`. Channel 0 holds the events with polarity 0 (darker), channel 1 those with polarity 1 (brighter).

| Mode | Pixel value |
|---|---|
| `time_surface`, decay `linear` | `(t_last − start_t) / (end_t − start_t)` of the latest event of the pixel and polarity in the window, between 0 and 1. Newer events have higher values. `0` where there was no event (also for an event exactly at `start_t`) |
| `time_surface`, decay `exponential` | `exp(−(end_t − t_last) / tau_us)` of the latest event, between 0 and 1. An event at the end of the window gives about 1, older events decay exponentially. `0` where there was no event |
| `binary` | `1.0` where at least one event occurred, else `0.0` |
| `count` | Number of events of the pixel and polarity |

### Linear or exponential?

- **Linear** spreads the values evenly over the window: an event in the middle of a 40 ms window gives 0.5. The values depend only on the position of the event in the window, so they are comparable between windows of the same length.
- **Exponential** emphasizes the most recent events: with `tau_us` = 20000 (20 ms), an event 20 ms before the end of the window gives `exp(−1)` ≈ 0.37, one 40 ms before ≈ 0.14. A small `tau_us` keeps only the newest motion visible (sharp edges), a large one behaves almost like `binary`. Unlike the linear decay, events at the very start of the window are not 0.

The value of `tau_us` is in microseconds and independent of the window length. With 40 ms windows, values between 10000 and 100000 are typical.

```json
{"name": "recent", "mode": "time_surface", "time_surface": {"decay": "exponential", "tau_us": 20000}}
```

`"time_surface": "exponential"` is short for `{"decay": "exponential"}`. For one frame, the decay can also be chosen in code: `window.frame(decay="exponential")`, `engine.build(events, start_t, end_t, decay="linear")`.

## Tiles

The frame is split into a grid of `size × size` tiles, row by row:

- With 1280 × 720 and 80 px tiles, the grid has 9 rows and 16 columns, 144 tiles.
- If the frame size is not a multiple of the tile size, the tiles at the right and bottom edges are smaller. With 64 px tiles, the last row is 16 pixels high (720 = 11 × 64 + 16).
- Tile ids, rows and columns start at 1: `id = (row − 1) × columns + col`. Tile 1 is the top-left tile.
- Each tile carries its event count. Tiles with fewer than `min_events` events are not returned.

## Usage

### Frames from a camera or a RAW file

```python
import opennve

tiles80 = opennve.load_frame_config("frames.json", "tiles80")

with opennve.open_camera(path="cameras.json", name="left") as camera:  # or opennve.open_raw("recording.raw", ...)
    for frame in camera.frames(tiles80):  # one frame per time window (window duration of the camera)
        frame.name          # "tiles80"
        frame.data          # height x width x 2, float32
        frame.events        # the events of the window in the output format (x, y, p, t)
        for tile in frame.tiles:
            agent.process(tile.id, tile.data)  # tile.data: tile height x tile width x 2
```

Without a frame configuration, `camera.frames()` uses the defaults: sensor size, linear time surface, no tiles. The window duration is still a camera setting (`window`), or `camera.frames(tiles80, duration_ms=20)`.

`camera.read_frames(config)` / `raw.read_frames(config)` return the frames completed by the next packet (often `[]`), for reading several cameras in one loop.

### Several frame configurations at once

`frame_sets` builds one frame per configuration from the **same** time window and yields a dictionary `{name: Frame}`:

```python
frame_configs = opennve.load_frame_configs("frames.json")

with opennve.open_camera(path="cameras.json", name="left") as camera:
    for frames in camera.frame_sets(frame_configs):
        full = frames["full"]                  # whole frame
        for tile in frames["tiles80"].tiles:   # 80 x 80 tiles of the same window
            agent.process(tile.id, tile.data)
```

`raw.frame_sets(configs)`, `camera.read_frame_sets(configs)` and `raw.read_frame_sets(configs)` work the same way, and `window.frames(configs)` does it for one window. Instead of a list, a dictionary `{name: configuration}` can be passed.

Different cameras can use different frame configurations:

```python
frame_configs = {config.name: config for config in opennve.load_frame_configs("frames.json")}
left_frames = left_camera.frames(frame_configs["tiles80"])
right_frames = right_camera.frames(frame_configs["exponential"])
```

### Frame first, tiles later

With `"tiles": null` (the default), `camera.frames(config)` returns whole frames. You can process a frame and split it into tiles afterwards:

```python
full = opennve.load_frame_config("frames.json", "full")
for frame in camera.frames(full):  # "tiles": null -> frame.tiles == []
    frame.data = denoise(frame.data)  # process the frame (in place or as a new array)
    tiles = frame.split(80)  # 80 x 80 tiles of the processed frame, empty tiles skipped
    for tile in tiles:
        agent.process(tile.id, tile.data)
```

`frame.split(tile_size=None, min_events=None, data=None)`:

- `tile_size`: tile size in pixels. Default: the `tiles` setting of the frame configuration, else 80.
- `min_events`: only tiles with at least this many events (`0` = all tiles). Default: the frame configuration, else 1 (skip empty tiles). The events are those of the frame, so this still works after changing `frame.data`.
- `data`: split another array instead of `frame.data`, for instance a one-channel image computed from the frame (`frame.split(80, data=gray)`). Its first two dimensions must be the frame size; further dimensions are kept.

The tiles are returned and also stored in `frame.tiles`. A frame can be split several times, for instance with different tile sizes. `frame.tile_event_counts(80)` returns the number of events per tile as a (rows × columns) array.

Any array can be split without a frame, with `opennve.split_into_tiles(array, 80)` (all tiles, `event_count` is `None`), or with event counts: `split_into_tiles(array, 80, event_counts, min_events=1)`.

### Events and frames side by side

The event blocks are unchanged: `camera.events()` still yields the event packets, and `camera.windows()` the time windows. A frame can be built from any window:

```python
for window in camera.windows():
    events = window.events              # x, y, p, t
    frame = window.frame(tiles80)       # with a frame configuration
    default = window.frame()            # sensor size, linear time surface, no tiles
    binary = window.frame(tiles80, mode="binary")
    custom = window.frame({"width": 640, "height": 480, "tiles": {"size": 64}})
    frames = window.frames(frame_configs)  # {name: Frame}
```

### Changing the settings while running

The frame configuration is chosen per call, so another configuration can be used for the next window at any time, for instance `window.frame(frame_configs[index])` in a loop (see `examples/tiles.py`, key F).

### Display

`frame.to_image()` returns a BGR image for OpenCV: polarity 1 green, polarity 0 red, brightness from the frame value, tile borders in white (`draw_tiles=False` hides them).

```sh
python examples/tiles.py examples/cameras.json --camera left --frame tiles80   # live
python examples/tiles.py recording.raw --frame exponential                     # RAW file
python examples/tiles.py recording.raw --frames my_frames.json --frame full
```

### Without camera

```python
engine = opennve.FrameEngine(1280, 720, tile_size=80, mode="time_surface", min_events=1)
frame = engine.build(events, start_t, end_t, camera="left")  # events: fields x, y, t and p (or on)
```

`events` can be a numpy structured array, a pandas DataFrame or a dictionary of arrays with the fields `x`, `y`, `t` and `p` (or `on`). pandas is optional, install it with the extra `opennve[pandas]` or `pip install pandas`.

## Migrating from the previous FrameEngine

The previous interface is still available, with the same results:

```python
from opennve import FrameEngine

engine = FrameEngine(frame_width=1280, frame_height=720, frame_tile_size=80)
tiles = engine.generate_frame(frame_data, threshold=5, select_insects_points=True, use_time_surface=True,
                              frame_start=start, frame_end=end)
# [{"tile_id": 1, "row": 1, "col": 1, "tile_data": ndarray}, ...]
surface = FrameEngine.create_linear_time_surface(frame_data, 1280, 720, start, end)
```

`tau` and `activity_threshold` are accepted and ignored, as before, so `generate_frame` keeps giving linear time surfaces. For exponential time surfaces, use the new interface (`FrameEngine(..., decay="exponential", tau_us=...)` or the `time_surface` configuration). `frame.tile_dicts()` converts the tiles of a new `Frame` to the same list format.

## API

| Name | Description |
|---|---|
| `FrameConfig(width=None, height=None, mode="time_surface", tiles=None, time_surface=TimeSurfaceConfig(), name=None)` | One frame configuration (`from_dict`, `to_dict`, `from_json`, `to_json`, `copy(**changes)`) |
| `load_frame_configs(path)` | All frame configurations of a file (list) |
| `load_frame_config(path, name=None)` | One frame configuration by name |
| `save_frame_configs(path, configs)` | Writes one configuration or `{"frames": [...]}` |
| `as_frame_config(value)` / `as_frame_configs(values)` | Converts a dictionary, JSON string or list (names filled in) |
| `TimeSurfaceConfig(decay="linear", tau_us=100000)` | Time surface settings |
| `TileConfig(size=80, min_events=1)` | Tile settings |
| `FrameEngine(width, height, tile_size=None, mode="time_surface", min_events=1, decay="linear", tau_us=100000)` | Builds frames: `build(events, start_t, end_t, camera="", mode=None, decay=None)`, `build_window(window)`, `from_config(config, sensor_width, sensor_height)` |
| `Frame` | `camera`, `name` (frame configuration), `start_t`, `end_t`, `mode`, `decay`, `data`, `width`, `height`, `event_count`, `tiles`, `tile_rows`, `tile_cols`, `events`, `window`, `split(tile_size, min_events, data)`, `tile_event_counts(tile_size)`, `tile(id)`, `tile_dicts()`, `to_image()` |
| `split_into_tiles(data, tile_size, event_counts=None, min_events=0)` | Splits any array (height × width × …) into tiles |
| `tile_grid(width, height, tile_size)` | `(rows, columns)` of the tile grid |
| `Tile` | `id`, `row`, `col`, `x`, `y`, `width`, `height`, `event_count` (`None` if unknown), `data`, `to_dict()` |
| `EventCamera.frames(config=None, duration_ms=None)` / `read_frames(...)` | Frames of a live camera |
| `EventCamera.frame_sets(configs, duration_ms=None)` / `read_frame_sets(...)` | `{name: Frame}` per window, one frame per configuration |
| `RawReader.frames(...)` / `read_frames(...)` / `frame_sets(...)` / `read_frame_sets(...)` | The same for RAW files |
| `EventWindow.frame(config=None, mode=None, decay=None)` | Frame of one window |
| `EventWindow.frames(configs)` | `{name: Frame}` of one window |

## Performance

Building a 1280 × 720 time surface (linear or exponential) from 2 million events takes about 70–85 ms on one CPU core, splitting it into 80 px tiles about 15 ms (about 3.5 times faster than the previous implementation). With 40 ms windows, typical event rates leave plenty of time for further processing.
