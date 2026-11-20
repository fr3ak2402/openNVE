# HDF5 files

Besides live cameras and [RAW files](raw-files.md), openNVE reads HDF5 event files (`.h5`, `.hdf5`) with the columns **x, y, p, t** and optionally **l** (label). They are processed exactly like a live camera: same packets, filters, output format, time windows, frames, agents and workflows. The labels travel with the events, so labeled data (for instance for training or evaluating a model) keeps its labels through windows and filters.

openNVE can also write such files, for instance the labeled events of a [workflow](workflows.md).

## Quick start

```python
import opennve

with opennve.open_hdf5("recording.h5", config_path="cameras.json", name="left") as recording:
    print(recording.info())
    for window in recording.windows():                     # 40 ms, or the "window" of the configuration
        events = window.events                              # output format of the camera, for instance x, y, p, t, l
        labels = window.polarity_events["l"]                # labels of the window (if the file has them)
        frame = window.frame(opennve.load_frame_config("frames.json", "tiles80"))
```

`opennve.open_recording(path)` opens `.raw` and HDF5 files alike (by the file name, or by the HDF5 signature for other names such as `.h`). On the command line:

```sh
opennve recording recording.h5
```

```text
recording.h5
  format:     HDF5, 1280 x 720
  dataset:    /events (columns, time unit us)
  size:       41.3 MB
  events:     12000000 (6011204 ON, 5988796 OFF), 0 trigger events
  time:       0.000000 s to 30.000000 s (30.000 s, 0.40 Mev/s)
  labels:     0: 11203411, 1: 702133, 2: 94456
```

## Supported layouts

The events are found automatically. If a file contains several event datasets, choose one with `dataset="..."`.

| Layout | Example | Notes |
|---|---|---|
| One dataset per column in a group | `/events/x`, `/events/y`, `/events/p`, `/events/t`, `/events/l` | Also directly in the root group. DSEC files (`t` relative to `/t_offset`) are supported |
| One dataset with named fields | `/events` with fields `x, y, p, t, l`; Prophesee `/CD/events` | Prophesee files compressed with the ECF codec need its HDF5 plugin (Metavision); other compression filters are available if `hdf5plugin` is installed |
| 2D dataset (events × columns) | `/events` of shape (N, 4) or (N, 5) | The column order comes from the attribute `columns` (for instance `["t", "x", "y", "p"]`) or from `columns=[...]` |
| pandas DataFrame | `df.to_hdf("file.h5", key="events")` | `format="fixed"` needs only h5py; `format="table"` needs pandas and PyTables (extra `opennve[pandas]`) |

Column names are case-insensitive, with these aliases:

| Column | Names |
|---|---|
| x | `x` |
| y | `y` |
| p | `p`, `polarity`, `pol`, `on` – values > 0 are ON (`1`, `True`), all others OFF (`0`, `-1`, `False`) |
| t | `t`, `ts`, `timestamp`, `timestamps`, `time` |
| l | `l`, `label`, `labels`, `class`, `class_id` (optional) |

### Time stamps

openNVE works with integer microseconds. `time_unit="auto"` (default) uses the unit from the attributes of the dataset (`units` / `unit` / `time_unit`, for instance `"us"`, `"ms"`, `"s"`), else microseconds for integer time stamps and seconds for floating point time stamps. Set it explicitly with `time_unit="us"`, `"ms"`, `"s"` or `"ns"`. A `t_offset` dataset (DSEC) is added to all time stamps. `zero_time=True` lets the time stamps start at 0.

The events must be sorted by time. Unsorted files raise an error; `sort=True` sorts them (the whole file is loaded into memory).

### Sensor size

The sensor size comes from `width=` / `height=`, else from the attributes `width` / `height` (or `sensor_width` / `sensor_height`, `geometry` = `"1280x720"`) of the file or dataset, else from the largest x and y in the file.

## Labels

If the file has a label column, the events of packets and windows have the additional field `l` (int32): `packet.polarity_events["l"]`, `window.polarity_events["l"]`. To get the labels in the output format (`packet.events`, `window.events`, CSV), add `l` to the output fields of the camera configuration:

```json
"output": {"fields": ["x", "y", "p", "t", "l"]}
```

Sources without labels (live cameras, `.raw` files) give `l = 0`. Filters keep the labels of the remaining events. `labels=False` ignores the label column.

## Writing HDF5 files

```python
opennve.write_hdf5("labeled.h5", events, width=1280, height=720)    # new file

with opennve.Hdf5Writer("labeled.h5", width=1280, height=720) as writer:   # append in parts
    for window in recording.windows():
        writer.write(window.events)
```

`events` is a structured array with the fields `x`, `y`, `t` and `p` (or `on`), optionally `l`, for instance `window.events` or `item.events` of a workflow. The file gets one compressed, resizable dataset per column in the group `/events` (`group=` to change it) and the attributes `width`, `height` and `time_unit = "us"`. It can be read again with `open_hdf5`, h5py or other tools.

In a workflow, `{"type": "write", "format": "hdf5", "path": "output/{workflow}_{camera}.h5"}` writes the events of every item (see [Workflows](workflows.md)).

## Workflows

The source step `recording` (or `hdf5`) reads HDF5 files, with the same parameters as for RAW files plus `dataset`, `time_unit` and `sort`:

```json
{"type": "recording", "path": "recordings/*.h5", "camera": "left", "block_ms": 1000}
```

## API

| Name | Description |
|---|---|
| `open_hdf5(path, config=None, *, config_path=None, name=None, **options) -> Hdf5Reader` | Opens a file. Options: `dataset`, `columns`, `time_unit`, `width`, `height`, `labels`, `chunk_events` (default 1 000 000), `sort`, `speed` (1.0 = real time), `zero_time`, `filter`, `output`, `window` |
| `read_hdf5(path, **options)` | All events at once (fields `t, x, y, on` and `l`) |
| `open_recording(path, config=None, **options)` | `.raw` or HDF5, chosen by the file type |
| `Hdf5Reader` | Like `RawReader`: `events()`, `read()`, `windows()`, `read_windows()`, `frames()`, `frame_sets()`, `read_all()`, `info()`, `speed`, `width`, `height`, `has_labels`, `time_unit`, `event_count` |
| `write_hdf5(path, events, *, mode="w", group="events", width=None, height=None, compression="gzip", attrs=None)` | Writes events |
| `Hdf5Writer(path, ...)` | `write(events)`, `flush()`, `close()`, `count`; context manager |

HDF5 support uses [h5py](https://www.h5py.org/), which is installed with openNVE.
