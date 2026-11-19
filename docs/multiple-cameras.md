# Multiple cameras

## Telling cameras apart

Each camera has a serial number (`opennve list`). In the configuration, `serial` gives each camera a fixed name:

```json
{
    "cameras": [
        {"name": "left",  "model": "SE1-S4", "serial": "05110634"},
        {"name": "right", "model": "SE1-S4", "serial": "05110636"}
    ]
}
```

Tip: connect the cameras one at a time, note the serial numbers and put a label on each housing.

## Opening and reading

```python
import numpy as np
import opennve

cameras = opennve.open_cameras(opennve.load_configs("cameras.json"))
try:
    names = list(cameras)
    backlogs = np.zeros(len(names), dtype=np.int64)
    while True:
        # Read the camera with the largest backlog first
        index = int(np.argmax(backlogs))
        packet = cameras[names[index]].read()
        backlogs += 1
        if packet is None:
            backlogs[index] = 0
            continue
        backlogs[index] = packet.status.ring.backlog if packet.status.ring else 0
        print(packet.camera, len(packet.events))
finally:
    for camera in cameras.values():
        camera.close()
```

The cameras are read alternately from one thread. Selecting by backlog ensures that no camera fills up while another one is being read. `examples/viewer.py` shows the complete version, with all cameras displayed side by side.

For time windows, use `read_windows()`, see [Time windows and frames](time-windows.md#multiple-cameras).

## Synchronizing clocks

Without synchronization, each camera has its own clock. The timestamps of two cameras do not match exactly and drift apart over time. With a sync cable between the cameras, one camera (*primary*) sends its clock to the others (*secondaries*):

```json
{
    "cameras": [
        {"name": "left",  "model": "SE1-S4", "serial": "05110634",
         "settings": {"clock": "internal_with_output_enabled"}},
        {"name": "right", "model": "SE1-S4", "serial": "05110636",
         "settings": {"clock": "external"}}
    ]
}
```

| `clock` | Role |
|---|---|
| `internal` | own clock, no synchronization (default) |
| `internal_with_output_enabled` | **Primary**: own clock, signal on the sync output |
| `external` | **Secondary**: clock from the sync input |

The secondaries must be ready before the primary starts its clock. `open_cameras` therefore opens all cameras with `"clock": "external"` first, waits `secondary_start_delay` seconds (1 by default) and then opens the others.

With synchronized clocks, the [time windows](time-windows.md) of all cameras have the same boundaries. Windows with the same `start_t` belong together.

> **Not verified for the SE1-S4:** the driver uses the sync input and output pins of the Prophesee EVK4. Check the synchronization once: with a correct connection, the time offset between the cameras stays constant over minutes instead of slowly growing.

## USB and power

- Each SE1-S4 can deliver up to 1.6 Gb/s. Two cameras on a shared USB 3 hub (5 Gb/s) are not a problem.
- **Raspberry Pi 5:** connect one camera to each of the two blue USB 3 ports. With the 27 W power supply, the Pi provides at most 1.6 A over USB in total. For two cameras, a powered USB 3 hub is safer.
