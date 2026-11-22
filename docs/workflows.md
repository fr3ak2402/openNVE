# Workflows

A **workflow** is a freely configurable chain of steps, for instance: read the events of a camera → split them into time windows → build frames → split them into tiles → evaluate the tiles with a neural network → keep the blocks with detections → label the events → save them. Each step runs in its own thread(s), so reading, frame building, inference and writing run at the same time, and slow steps (the neural network) can use several threads.

```text
camera ──► windows ──► frames ×2 ──► agent ×4 ──► require ──► label_events ──► write
(source)               (threads)     (threads)                                 (sink)
   │  queue   │  queue    │  queue      │  queue     │  queue     │  queue      │
```

Workflows are defined in a JSON file (`workflows.json`) next to the [camera](configuration.md), [frame](frames.md) and [agent](agents.md) configurations, or in code.

## Example

`examples/workflows.json` (shortened):

```json
{
    "workflows": [
        {
            "name": "insects",
            "default": true,
            "cameras": "cameras.json",
            "frames": "frames.json",
            "agents": "agents.json",
            "ordered": true,
            "steps": [
                {"type": "camera", "cameras": ["left"], "block_ms": 1000},
                {"type": "windows", "count": 3, "overlap_ms": 200},
                {"type": "frames", "frame": "tiles80", "min_events_per_ms": 0.15},
                {"type": "agent", "agent": "insects"},
                {"type": "require", "agent": "insects", "min_windows": 1},
                {"type": "label_events", "agent": "insects", "label": 2},
                {"type": "log", "agent": "insects"},
                {"type": "write", "format": "csv", "path": "output/{workflow}_{camera}_{date}_{time}.csv"}
            ]
        }
    ]
}
```

Run it:

```sh
opennve workflow run examples/workflows.json                    # the default workflow, until 'stop' + Enter or Ctrl+C
opennve workflow run examples/workflows.json --name insects_raw --debug
opennve workflow list examples/workflows.json
opennve check examples/workflows.json                           # validates all workflows and steps
```

```python
import opennve

workflow = opennve.load_workflow("workflows.json", "insects")   # without a name: the default workflow
workflow.run()                                                   # blocks until the end, Ctrl+C or workflow.stop()
print(workflow.statistics_report())
```

## Workflow settings

| Field | Default | Description |
|---|---|---|
| `name` | – | Name of the workflow |
| `default` | `false` | Workflow used when no name is given (at most one per file) |
| `description` | `""` | Free text |
| `cameras`, `frames`, `agents` | `null` | Configuration files (relative to the workflow file) or the configurations themselves (lists) |
| `steps` | – | The steps, in order. The first step is a source |
| `queue_size` | `null` | Maximum number of items waiting between two steps. A full queue slows the step before it down (back pressure). `null`: automatic, 8 or twice the threads of the busiest step |
| `cpus` | `"auto"` | CPU cores the automatic threads are sized for. `"auto"`/`null`: detected (`opennve.cpu_count()`); a number, for instance to leave cores free for other programs |
| `ordered` | `true` | Sinks (`write`, `collect`) receive the items in the order they were read, also if threads finish them in another order |
| `debug` | `false` | Prints the steps at the start and the statistics at the end |

Every step has `type`, optionally `name` (for the statistics) and `threads`: a number or `"auto"` (default for `frames`, `tiles` and `agent`; the other steps default to 1). See [Multithreading](#multithreading).

## How items flow through the steps

The source creates one **item** (`WorkItem`) per block of events. The item then passes through all steps; each step adds its results:

| `WorkItem` | Set by | Content |
|---|---|---|
| `index`, `camera` | source | Sequence number (0, 1, 2, … over all cameras), camera name |
| `block` | source | The events of the block (`EventWindow`: `polarity_events`, `events`, `start_t`, `end_t`, …) |
| `windows` | `windows` | Time windows of the block (without the step: the block itself) |
| `frames` | `frames` | One `{frame configuration name: Frame}` per window |
| `results` | `agent` | One `{agent name: AgentResult}` per window |
| `events` | `label_events` | Output events with the label field |
| `data` | your steps | Free storage |
| `timings` | workflow | Seconds per step |

A step can drop an item (`require`, or a function that returns `None`); the next steps then do not see it.

## Steps

### Sources

**`camera`** – live cameras of the camera configuration.

| Parameter | Default | Description |
|---|---|---|
| `cameras` | `null` | Camera names (`null`: all cameras of the file). Several cameras are read in turn; each block keeps its camera name |
| `block_ms` | `null` | Block length in ms (`null`: the `window` duration of the camera configuration) |
| `max_blocks`, `duration_s` | `null` | Stop after this many blocks / seconds |
| `install_driver` | `false` | Install missing USB drivers first (Windows) |

**`recording`** (also `raw`, `hdf5`) – recordings: `.raw` files and [HDF5 files](hdf5-files.md) (`.h5`, `.hdf5`), chosen by the file type.

| Parameter | Default | Description |
|---|---|---|
| `path` | – | File, glob pattern (`"recordings/*.raw"`, `"data/*.h5"`) or list; files are read one after the other |
| `camera` | `null` | Camera of the camera configuration whose filter, output format and window are used |
| `block_ms` | `null` | Block length (`null`: window of the camera configuration, else 40 ms) |
| `speed` | `null` | `1.0` = real time, `null` = as fast as possible |
| `max_blocks` | `null` | Stop after this many blocks |
| `chunk_kb` | `null` | `.raw`: bytes read at once (1024 KB; 16 KB with a `speed`, for smooth playback) |
| `dataset`, `time_unit`, `sort` | `null`, `"auto"`, `false` | HDF5: dataset with the events, unit of the time stamps, sort by time (see [HDF5 files](hdf5-files.md)) |

Labels of HDF5 files are kept in `item.block.polarity_events["l"]` and, with the output field `l`, in the written events.

### Processing

**`windows`** – splits each block into time windows.

| Parameter | Description |
|---|---|
| `count`, `overlap_ms` | `count` windows of equal size that cover the block, neighbours overlapping by `overlap_ms`. Example: 1000 ms, 3 windows, 200 ms overlap → 0–467, 267–733, 533–1000 ms |
| `duration_ms`, `step_ms` | Alternatively: windows of `duration_ms` every `step_ms` (sliding windows) |
| `align` | `block` (default): the windows start at the start of the block. `first_event`: they start at the first event of the block (as `build_bounds` of the previous workflow engine) |

**`frames`** – builds the frames of each window.

| Parameter | Description |
|---|---|
| `frame` | Name of a frame configuration, an inline configuration (object), or a list (several frames per window) |
| `min_events_per_ms` | Minimum events per tile relative to the window length, for instance `0.15` = 50 events for a 333 ms window (overrides `min_events` of the configuration) |

**`tiles`** – splits frames that were built without tiles (after processing the frame, for instance).

| Parameter | Default | Description |
|---|---|---|
| `frame` | `null` | Frame configuration whose frames are split (`null`: all) |
| `size` | `80` | Tile size in pixels |
| `min_events` | `1` | Minimum events per tile (0 = all tiles) |
| `min_events_per_ms` | `null` | Minimum relative to the window length (overrides `min_events`) |

**`agent`** – runs agents on the frames / tiles of every window. Each thread loads its own copy of the model.

| Parameter | Description |
|---|---|
| `agent` | Agent name (from `agents`), an inline agent configuration, or a list |
| `frame` | Frame configuration the agent uses (default: the agent's `frame` setting, or the only frame) |

**`require`** – keeps only items in which an agent found something; drops the others.

| Parameter | Default | Description |
|---|---|---|
| `agent` | – | Agent name |
| `min_windows` | `1` | Minimum number of windows with at least one selected prediction |
| `min_pixels` | `null` | Segmentation: a prediction counts only with at least this many target pixels |

**`label_events`** – labels the events of the block with the predictions: an event is labeled if, in a window containing its time stamp, the agent predicted a target pixel at its position (segmentation) or selected its tile (classification). With overlapping windows, one match is enough.

| Parameter | Default | Description |
|---|---|---|
| `agent` | – | Agent name |
| `label` | `null` | Value for labeled events (`null`: the predicted class id) |
| `field` | `"l"` | Name of the label field |
| `only_labeled` | `true` | Keep only labeled events (`false`: all events, unlabeled with 0) |

The result (`item.events`) has the output format of the camera (for instance `x, y, p, t`) plus the label field.

**`log`** – prints one line per item, in the order of the items (one thread): block, events, windows, tiles, positive windows and pixels per agent, step timings (`"timings": false` hides them; `"agent"` limits the output to one agent).

**`callback`** – calls a Python function (only from code, see below).

### Sinks

**`write`** – writes to files (one thread).

| Parameter | Default | Description |
|---|---|---|
| `format` | `"csv"` | `csv`: one row per event (`item.events`, or all events of the block if no step set events); `hdf5`: the same events in an [HDF5 file](hdf5-files.md) (one dataset per column, labels included); `jsonl`: one JSON line per item with its windows and the selected predictions (with pixels) |
| `path` | `"output/{workflow}_{date}_{time}.csv"` | File name with placeholders `{workflow}`, `{camera}`, `{date}`, `{time}`, `{datetime}` (start of the workflow) and `{index}`. Files are appended; the CSV header is written once |
| `camera_column` | `false` | CSV: camera name as the first column |

**`collect`** – keeps the items in memory (`workflow.results`), for use from code.

Several sinks are possible, for instance a CSV and a JSON lines file.

## Multithreading

- Every step runs in its own thread; `"threads": n` starts n threads for that step. Each thread has its own step instance, so every `agent` thread has its own model.
- **Automatic threads:** `"threads": "auto"` (the default of `frames`, `tiles` and `agent`) sizes the step to the hardware. openNVE reads the CPU cores this process may use (`opennve.cpu_count()`: affinity, container limits, environment variable `OPENNVE_CPUS`, or the workflow's `cpus`):
  - `agent`: one thread per core (at most 16) with **one TensorFlow Lite interpreter thread each** — several interpreters in parallel are faster than one interpreter with several threads, and *agent threads × interpreter threads* never exceeds the cores. Above 16 cores, the remaining cores go to the interpreters. Several agent steps share the cores equally. TensorFlow (Keras) models run in one thread, TensorFlow uses all cores itself.
  - With a fixed number of agent threads, agents with `"threads": "auto"` get *cores / agent threads* interpreter threads; with fixed interpreter threads in `agents.json`, the agent step gets *cores / interpreter threads* threads.
  - `frames`, `tiles`: 1 thread up to 7 cores, 2 from 8 cores (they need ~0.1 ms per tile against ~4 ms in the agent).
  - Agents with several devices (`"device": ["gpu", "cpu"]`): 2 threads per accelerator, the other cores on the CPU; see [GPU and NPU](accelerators.md#gpu-and-cpu-together).
  - `opennve workflow threads workflows.json` (or `workflow.thread_report()`, `workflow.thread_plan()`) shows the threads for this machine; `--cpus 8` shows them for another one.
- Sources and `write` / `collect` run in one thread.
- The steps are connected by queues of `queue_size` items. When a step is slower than the source, the queue fills up and the source waits; for live cameras the camera buffer then grows, so give slow steps enough threads.
- With `"ordered": true`, the sinks write the items in the original order.
- Threads help where the work releases Python's global interpreter lock: numpy, OpenCV, LiteRT and TensorFlow do. On a 2-core machine, 2 `agent` threads with 1 interpreter thread each ran a U-Net (80 × 80 tiles) at 3.7 ms per tile, 1 agent thread with 2 interpreter threads at 4.7 ms, 1 × 1 at 6.9 ms; 2 × 2 (more threads than cores) made an int8 model 4 times slower.
- `workflow.statistics()` / `statistics_report()` show the items, dropped items and time per item of every step: the step with the highest time per item is the one to give more threads.

## Stopping and errors

- `opennve workflow run`: type `stop` and Enter, or press Ctrl+C. `--duration 60` stops after 60 s.
- In code: `workflow.run(duration_s=None)` blocks; `workflow.start()` returns immediately, `workflow.stop()` stops the source and `workflow.join()` waits.
- After a stop, the blocks already read are still processed and written.
- If a step raises an exception, the workflow stops and `run()` / `join()` raises it.

## Workflows in code

```python
import opennve

def count_on(item):                                   # a function as a step
    item.data["on_events"] = int(item.block.polarity_events["on"].sum())
    return item                                       # None drops the item

workflow = opennve.Workflow.from_steps(
    "my_workflow",
    [
        {"type": "raw", "path": "recording.raw", "block_ms": 1000},
        {"type": "frames", "frame": {"name": "t80", "tiles": {"size": 80}}},
        {"type": "agent", "agent": {"name": "insects", "model": "models/unet_int8.tflite",
                                    "prediction": {"task": "segmentation", "activation": "softmax", "target_classes": [2]}},
         "threads": 2},
        count_on,
        {"type": "collect"},
    ],
    frames="frames.json",                             # optional configuration files
)
workflow.run()
for item in workflow.results:
    print(item.index, item.data["on_events"], len(item.results[0]["insects"].selected))
```

`workflow.add_step(step, position=None, **params)` adds a step (type name, dictionary or function) before the start.

### Own step types

```python
@opennve.register_step("denoise_frames")
class DenoiseFrames(opennve.Step):
    PARAMETERS = {"kernel": 3}             # allowed parameters and defaults

    def validate(self):                    # when the workflow is created
        if self.params["kernel"] % 2 == 0:
            raise ValueError("kernel must be odd")

    def setup(self):                       # once per thread, before the first item
        ...

    def process(self, item):               # per item; return None to drop it
        for frames in item.frames:
            for frame in frames.values():
                ...
        return item

    def close(self):                       # once per thread, after the last item
        ...
```

Then `{"type": "denoise_frames", "kernel": 5, "threads": 2}` works in `workflows.json`. Sources set `kind = "source"` and implement `items(stop_event)` (a generator of `WorkItem`), sinks set `kind = "sink"`. `opennve.list_steps()` lists all registered step types. A step has access to `self.context` (workflow name, configuration files, `context.frame_config(name)`, `context.agent_config(name)`, `context.log(message)`).

## Migration from the previous workflow engine

| Previous code | openNVE |
|---|---|
| `workflow.json` with `event_cameras`, `workers`, `vision_engine`, `frame_engine`, `agent_name`, `output_file` | `workflows.json` with `cameras`, `frames`, `agents` and a list of `steps` |
| Worker `collector` (VisionEngine, `sequence_length_ms`, test data) | Step `camera` (`block_ms`) or `raw` (recordings instead of the test mode) |
| `frame_engine.sequence_windows`, `sequence_overlap_ms` (`build_bounds`) | Step `windows` with `count`, `overlap_ms` and `"align": "first_event"` (same window bounds) |
| `frame_weight_px`, `frame_height_px`, `tile_size_px` | Frame configuration (`width`, `height`, `tiles.size`) |
| Tile threshold `(upper - lower) / 10000 * 1.5` | `"min_events_per_ms": 0.15` |
| Worker `processor` (several) | Steps `frames`, `agent` (with `threads`), `require`, `label_events` |
| `true_count >= 1` | Step `require` with `min_windows` |
| Event matching per window, `l = 2` | Step `label_events` with `"label": 2` |
| Worker `writer` (CSV, `file_prefix`, date and time) | Step `write` with a `path` pattern |
| `stop` in the console | The same with `opennve workflow run` |
| Debug output (timings, detections) | Step `log`, `"debug": true`, `statistics_report()` |
