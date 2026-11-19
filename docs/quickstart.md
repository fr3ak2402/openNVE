# Quick start

## 1. Find the camera

```sh
opennve list
```

```text
SE1-S4     05110634   USB 3.0 SuperSpeed (5.0 Gb/s)    ok
```

If the list is empty, see [Troubleshooting](troubleshooting.md#camera-not-found).

## 2. Create a configuration

```sh
opennve template SE1-S4 --serial 05110634 --name left > cameras.json
```

The file contains all settings with their default values. A short version is enough to start, since missing fields keep their default value:

```json
{
    "name": "left",
    "model": "SE1-S4",
    "serial": "05110634",
    "settings": {
        "biases": {"diff_on": 20, "diff_off": 20}
    },
    "filter": {
        "background_activity": {"enabled": true}
    }
}
```

```sh
opennve check cameras.json
```

## 3. Read events

```python
import opennve

with opennve.open_camera(path="cameras.json") as camera:
    print(camera.info())
    for packet in camera.events():
        print(packet.camera, len(packet.events))
        print(packet.events[:5])  # fields x, y, p, t
```

## 4. Build frames (40 ms)

```python
import cv2
import opennve

with opennve.open_camera(path="cameras.json") as camera:
    for window in camera.windows():  # 40 ms windows by default
        frame = window.to_frame("color")  # ON green, OFF red
        cv2.imshow("SE1-S4", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
cv2.destroyAllWindows()
```

## 5. Examples in the repository

```sh
python examples/list_cameras.py                         # cameras and ready-to-use configurations
python examples/viewer.py examples/cameras.json          # live view of all cameras, F = filter on/off
python examples/frames.py examples/cameras.json left     # one frame per time window, M = frame mode
python examples/raw_player.py recording.raw              # play a .raw recording as frames
python examples/tiles.py examples/cameras.json left      # frames and 80 x 80 tiles from the frame engine
```

Continue with the [configuration](configuration.md) or the [Python API](api.md).
