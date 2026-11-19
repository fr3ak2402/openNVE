# Time windows and frames

An event camera does not deliver images but a continuous stream of individual events. For algorithms that expect images, for instance neural networks, openNVE collects the events of a fixed time span and builds a frame from them.

## Setting the window duration

```json
"window": {"duration_ms": 40}
```

- **Default: 40 ms** (25 frames per second), also without a `window` section
- Allowed: 0.01 ms to 60,000 ms
- `"window": null` disables windows
- In code: `camera.windows(duration_ms=20)` overrides the configuration, `camera.update_window({"duration_ms": 20})` changes it permanently

## Usage

```python
import opennve

with opennve.open_camera(path="cameras.json", name="left") as camera:
    for window in camera.windows():
        print(window.camera, window.start_t, window.end_t, len(window))
        events = window.events            # all events of the window (x, y, p, t)
        frame = window.to_frame("count")  # 720 × 1280 image
```

## How windows are built

- **Camera time:** windows follow the camera's timestamps, not the computer clock. With 40 ms, the boundaries are always multiples of 40 ms: [0, 40), [40, 80), … ms of camera time. Two [synchronized cameras](multiple-cameras.md) therefore produce windows with exactly the same boundaries, so their frames can be paired directly.
- **Complete:** every event ends up in exactly one window, also when a USB packet spans a window boundary. `start_t` belongs to the window, `end_t` does not.
- **On time without motion:** a window is complete as soon as the camera time passes its end (`status.ring.current_t`). Quiet periods produce empty windows. Only very long pauses (more than 10,000 windows in a row) are skipped.
- **Filtered:** windows contain the events after the software filter (`filter`).
- **Latency:** a window is returned as soon as the camera has sent data past its end, typically a few milliseconds later.

## Building frames

There are two ways to build frames from a window:

- `window.to_frame(mode)`: simple images, mainly for display (below).
- `window.frame()` / `camera.frames()`: the **frame engine**, with time surface, binary and count tensors (two polarity channels) and tiles, configured with separate, named frame configurations (`frames.json`). See [Frames and tiles](frames.md).

`window.to_frame(mode)` returns a NumPy array of size height × width:

| Mode | Data type | Contents |
|---|---|---|
| `"count"` | `uint16` | Number of events per pixel |
| `"polarity"` | `int16` | ON events minus OFF events per pixel: positive = brighter, negative = darker |
| `"binary"` | `uint8` | `255` where at least one event occurred, else `0` |
| `"color"` | `uint8`, 3 channels | BGR image for OpenCV: ON green, OFF red (the latest event per pixel wins) |

`count` and `polarity` are meant for further processing. To display them with OpenCV, scale them:

```python
import numpy as np

count = window.to_frame("count")
image = np.clip(count.astype(np.float32) * 64, 0, 255).astype(np.uint8)

polarity = window.to_frame("polarity")
image = np.clip(polarity.astype(np.float32) * 64 + 128, 0, 255).astype(np.uint8)  # gray = no change
```

Custom representations can be built directly from `window.polarity_events`, for instance a *time surface* from the timestamps `t`.

## Multiple cameras

`windows()` blocks until a window is complete. To read several cameras in one loop, use `read_windows()`: it reads one packet and returns the windows it completed, often an empty list.

```python
cameras = opennve.open_cameras(opennve.load_configs("cameras.json"))
latest = {}
while True:
    for name, camera in cameras.items():
        for window in camera.read_windows():
            latest[name] = window  # windows with the same start_t belong together
    # for instance: if latest["left"].start_t == latest["right"].start_t -> process the stereo pair
```

## Performance

Measured with 2 million events (1280 × 720):

| Step | Duration |
|---|---|
| Splitting into windows | about 16 ms |
| `to_frame("binary")` | about 22 ms |
| `to_frame("count")` | about 33 ms |
| `to_frame("polarity")` | about 59 ms |
| `to_frame("color")` | about 130 ms |

With 40 ms windows, 2 million events per window correspond to 50 million events/s. At realistic rates, there is plenty of time left for further processing.

## Without EventCamera

For other data sources, for instance recorded events:

```python
accumulator = opennve.WindowAccumulator(duration_us=40_000, camera="left", width=1280, height=720)
for chunk in chunks:  # NumPy arrays with fields t, x, y, on, sorted by t
    for window in accumulator.push(chunk):
        frame = window.to_frame("count")
```
