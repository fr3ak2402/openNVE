# Frames and tiles

The frame engine turns each time window (40 ms by default, see [Time windows and frames](time-windows.md)) into a frame tensor and, optionally, splits it into tiles, for instance 80 × 80 pixels for an agent or a neural network. The events of the window stay available next to the frame.

It works the same for live cameras and [RAW files](raw-files.md).

## Configuration

```json
"frame": {
    "width": 1280,
    "height": 720,
    "mode": "time_surface",
    "tiles": {"size": 80, "min_events": 1}
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `width`, `height` | number or `null` | `null` | Frame size in pixels. `null` = sensor size (1280 × 720 for the SE1-S4). Events outside the frame are ignored (no scaling) |
| `mode` | string | `"time_surface"` | `time_surface`, `binary` or `count` (see below) |
| `tiles` | object or `null` | `null` | Tile grid. `null` = no tiles, only the whole frame. A number such as `80` is short for `{"size": 80}` |
| `tiles.size` | number | `80` | Tile size in pixels (square tiles) |
| `tiles.min_events` | number | `1` | Only tiles with at least this many events are returned. `1` skips empty tiles, `0` returns all tiles |

## Modes

Every frame has the shape **height × width × 2**, `float32`. Channel 0 holds the events with polarity 0 (darker), channel 1 those with polarity 1 (brighter).

| Mode | Pixel value |
|---|---|
| `time_surface` | Linear time surface: `(t_last − start_t) / (end_t − start_t)` of the latest event of the pixel and polarity in the window, between 0 and 1. Newer events have higher values. `0` where there was no event (also for an event exactly at `start_t`) |
| `binary` | `1.0` where at least one event occurred, else `0.0` |
| `count` | Number of events of the pixel and polarity |

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

with opennve.open_camera(path="cameras.json", name="left") as camera:  # or opennve.open_raw("recording.raw", ...)
    for frame in camera.frames():  # one frame per time window
        frame.data          # height x width x 2, float32
        frame.events        # the events of the window in the output format (x, y, p, t)
        for tile in frame.tiles:
            agent.process(tile.id, tile.data)  # tile.data: tile height x tile width x 2
```

`camera.read_frames()` / `raw.read_frames()` return the frames completed by the next packet (often `[]`), for reading several cameras in one loop.

### Events and frames side by side

The event blocks are unchanged: `camera.events()` still yields the event packets, and `camera.windows()` the time windows. A frame can be built from any window:

```python
for window in camera.windows():
    events = window.events              # x, y, p, t
    frame = window.frame()              # with the "frame" configuration of the camera
    binary = window.frame(mode="binary")
    custom = window.frame({"width": 640, "height": 480, "tiles": {"size": 64}})
```

### Changing the settings while running

```python
camera.update_frame({"mode": "binary", "tiles": {"size": 64, "min_events": 10}})
```

### Display

`frame.to_image()` returns a BGR image for OpenCV: polarity 1 green, polarity 0 red, brightness from the frame value, tile borders in white (`draw_tiles=False` hides them).

```sh
python examples/tiles.py examples/cameras.json left   # live
python examples/tiles.py recording.raw                 # RAW file
```

### Without camera

```python
engine = opennve.FrameEngine(1280, 720, tile_size=80, mode="time_surface", min_events=1)
frame = engine.build(events, start_t, end_t, camera="left")  # events: fields x, y, t and p (or on)
```

`events` can be a numpy structured array, a pandas DataFrame or a dictionary of arrays with the fields `x`, `y`, `t` and `p` (or `on`).

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

`tau` and `activity_threshold` are accepted and ignored, as before. `frame.tile_dicts()` converts the tiles of a new `Frame` to the same list format.

## API

| Name | Description |
|---|---|
| `FrameConfig(width=None, height=None, mode="time_surface", tiles=None)` | Frame settings (`from_dict`, `to_dict`) |
| `TileConfig(size=80, min_events=1)` | Tile settings |
| `FrameEngine(width, height, tile_size=None, mode="time_surface", min_events=1)` | Builds frames: `build(events, start_t, end_t, camera="")`, `build_window(window)`, `from_config(config, sensor_width, sensor_height)` |
| `Frame` | `camera`, `start_t`, `end_t`, `mode`, `data`, `width`, `height`, `event_count`, `tiles`, `tile_rows`, `tile_cols`, `events`, `window`, `tile(id)`, `tile_dicts()`, `to_image()` |
| `Tile` | `id`, `row`, `col`, `x`, `y`, `width`, `height`, `event_count`, `data`, `to_dict()` |
| `EventCamera.frames()` / `read_frames()` / `update_frame()` | Frames of a live camera |
| `RawReader.frames()` / `read_frames()` | Frames of a RAW file (`open_raw(..., frame={...})` overrides the configuration) |
| `EventWindow.frame(config=None, mode=None)` | Frame of one window |

## Performance

Building a 1280 × 720 time surface with 80 px tiles from 2 million events takes about 80 ms on one CPU core (about 3.5 times faster than the previous implementation). With 40 ms windows, typical event rates leave plenty of time for further processing.
