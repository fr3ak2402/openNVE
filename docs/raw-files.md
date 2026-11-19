# RAW files

Besides live cameras, openNVE reads `.raw` recordings. RAW is the file format of Prophesee / Metavision (Metavision Studio, Metavision SDK, OpenEB) and of the SENSING software. A RAW file is processed exactly like a live camera: same packets, filters, output format, time windows and frames.

## Supported formats

| Format | Word size | Typical source |
|---|---|---|
| **EVT 3.0** | 16 bit | SENSING SE1-S4 (default), Prophesee EVK4 / IMX636, Gen4.1 |
| **EVT 2.1** | 64 bit | SE1-S4 / IMX646 in EVT 2.1 mode (`endianness=little` and `legacy`) |
| **EVT 2.0** | 32 bit | Older Prophesee cameras (Gen3, Gen4.0) |

The format and sensor size are read from the file header (`% format EVT3;height=720;width=1280`, or the older `% evt 3.0` and `% geometry 1280x720`). Files without a header can be read by passing `format`, `width` and `height`.

The decoders follow the reference implementation in OpenEB and are tested against files written by an independent library ([faery](https://github.com/neuromorphicsystems/faery)). Polarity events and external trigger events are decoded; other event types (for instance monitoring events) are skipped. Events before the first time reference of the file are ignored, as in Metavision.

## Quick look at a file

```sh
opennve raw recording.raw
```

```text
recording.raw
  format:     EVT3, 1280 x 720
  serial:     05110634
  size:       182.4 MB
  events:     96214337 (48302114 ON, 47912223 OFF), 12 trigger events
  time:       0.010144 s to 30.012580 s (30.002 s, 3.21 Mev/s)
```

`opennve raw recording.raw --json` prints the same information, plus all header fields, as JSON.

## Reading a file

```python
import opennve

with opennve.open_raw("recording.raw") as raw:
    print(raw.info())  # format, size, serial number, header
    for packet in raw.events():
        print(packet.camera, len(packet.events))  # events in the output format (x, y, p, t)
```

The packets are `CameraPacket` objects, as for a live camera (see [Python API](api.md#camerapacket)): `packet.events`, `packet.rows()`, `packet.to_csv()`, `packet.polarity_events`, `packet.trigger_events`. The camera name is the file name without extension, or the name from the configuration.

### Time windows and frames

```python
with opennve.open_raw("recording.raw") as raw:
    for window in raw.windows():  # 40 ms by default
        frame = window.to_frame("count")
```

Windows work as described in [Time windows and frames](time-windows.md). At the end of the file, the last window is returned as well, even if the recording ends before its end time.

### Frames and tiles

```python
tiles80 = opennve.load_frame_config("frames.json", "tiles80")

with opennve.open_raw("recording.raw", config_path="cameras.json", name="left") as raw:
    for frame in raw.frames(tiles80):  # frame configuration, see Frames and tiles
        for tile in frame.tiles:
            ...
```

`raw.frame_sets(opennve.load_frame_configs("frames.json"))` yields one frame per frame configuration and window (`{name: Frame}`). See [Frames and tiles](frames.md).

### Same settings as the live camera

Pass the camera configuration to process the file with the same name, filter, output format and window duration as the live camera. Frames are configured separately, so the same frame configuration can be used for both. Hardware settings (`settings`) are ignored, since the events are already recorded.

```python
raw = opennve.open_raw("recording.raw", config_path="cameras.json", name="left")
raw = opennve.open_raw("recording.raw", {"name": "left", "model": "SE1-S4", "filter": {"background_activity": {"enabled": True}}})
raw = opennve.open_raw("recording.raw", filter={"polarity": "on"}, output={"fields": ["t", "x", "y"]}, window={"duration_ms": 20})
```

### Whole file at once

```python
events, triggers = opennve.read_raw("recording.raw")
# events: numpy array with fields t, x, y, on; triggers: fields t, id, rising
```

This loads all events into memory (13 bytes per event), so use `open_raw(...).events()` or `.windows()` for long recordings.

### Real-time playback

By default, files are read as fast as possible. With `speed`, packets are delivered at the pace of the recording, for instance to test a live pipeline with a recording:

```python
with opennve.open_raw("recording.raw", speed=1.0) as raw:  # 1.0 = real time, 2.0 = twice as fast
    for window in raw.windows():
        ...
        raw.speed = 4.0  # can be changed while playing
```

### Timestamps

Timestamps are the camera timestamps stored in the file, in microseconds, like the live camera. Metavision tools shift them so that the recording starts near 0; to do the same, use `zero_time=True` (the first event or trigger gets `t = 0`).

### Files without header

```python
raw = opennve.open_raw("camera_data.bin", format="EVT3", width=1280, height=720)
```

## API

### `open_raw(path, config=None, *, config_path=None, name=None, **options) -> RawReader`

| Option | Default | Description |
|---|---|---|
| `config` | `None` | `CameraConfig`, dictionary or JSON string: name, filter, output format and window |
| `config_path`, `name` | `None` | Load the configuration from a file (by camera name if it has several) |
| `filter`, `output`, `window`, `frame` | from `config` | Override single sections (`window=None` disables windows) |
| `format`, `width`, `height` | from the header | For files without header (`"EVT3"`, `"EVT2"`, `"EVT21"`) |
| `speed` | `None` | Playback speed (1.0 = real time), `None` = as fast as possible |
| `zero_time` | `False` | Shift timestamps so that the recording starts at 0 |
| `chunk_bytes` | 1 MiB | Amount of data decoded per packet |

### `RawReader`

| Attribute / method | Description |
|---|---|
| `events()` | Generator: yields `CameraPacket` objects until the end of the file |
| `read()` | Next packet, or `None` at the end of the file |
| `windows(duration_ms=None)` | Generator: yields `EventWindow` objects, including the last one |
| `read_windows(duration_ms=None)` | Windows completed by the next packet (often `[]`), `None` at the end of the file |
| `frames(config=None, duration_ms=None)` / `read_frames(...)` | Frames and tiles of the windows ([Frames and tiles](frames.md)) |
| `frame_sets(configs, duration_ms=None)` / `read_frame_sets(...)` | `{name: Frame}` per window, one frame per frame configuration |
| `read_all()` | Rest of the file at once: `(polarity events, trigger events)` |
| `info()` | Path, format, resolution, serial number, size, header, settings |
| `header` | `RawHeader`: `fields`, `format`, `width`, `height`, `endianness`, `data_offset`, `serial` |
| `name`, `width`, `height`, `resolution` | Camera name and sensor size |
| `speed` | Playback speed, can be changed while reading |
| `finished`, `is_open` | End of file reached / file open |
| `close()` | Closes the file (or use a `with` block) |

### Other functions

| Function | Description |
|---|---|
| `read_raw(path, **options) -> (events, triggers)` | Reads a whole file |
| `read_header(path, format=None, width=None, height=None) -> RawHeader` | Reads only the header |

## Performance

Measured on one CPU core with dense recordings (row segments, as produced by real scenes):

| Format | Decoding speed |
|---|---|
| EVT 3.0 | about 10 million events/s |
| EVT 2.0 | about 13–20 million events/s |

Files with very sparse events (isolated pixels) decode more slowly per event (about 4–5 million events/s for EVT 3.0).

## Example

```sh
python examples/raw_player.py recording.raw                           # plays the file as frames
python examples/raw_player.py recording.raw examples/cameras.json left  # with the settings of "left"
```

Keys: Q exit, M frame mode, + / − playback speed.
