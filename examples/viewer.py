"""
Live viewer for the cameras of a configuration file (filters applied).

    python examples/viewer.py examples/cameras.json

Q / ESC: exit, F: software filter on/off
"""

import sys
import time

import cv2
import numpy as np

import opennve

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "examples/cameras.json"
DISPLAY_FPS = 25
DISPLAY_DOWNSCALE = 2  # 1 = full size; events are max-pooled so they stay visible
ON_COLOR = (0, 255, 0)
OFF_COLOR = (0, 0, 255)


def downscale(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return frame
    height, width, channels = frame.shape
    return frame.reshape(height // factor, factor, width // factor, factor, channels).max(axis=(1, 3))


def main():
    configs = opennve.load_configs(CONFIG_PATH)
    print(opennve.list_cameras_json(configs))

    cameras = opennve.open_cameras(configs, iterator_timeout=0.01)
    try:
        frames = {}
        rates = {name: [0, 0, 0.0, 0.0] for name in cameras}  # raw, kept, raw/s, kept/s
        for name, camera in cameras.items():
            width, height = camera.resolution
            frames[name] = np.zeros((height, width, 3), dtype=np.uint8)
            print(f"{name}: {camera.model} {camera.config.serial} {width} x {height}")

        names = list(cameras)
        backlogs = np.zeros(len(names), dtype=np.int64)
        filters = {name: camera.config.filter for name, camera in cameras.items()}
        filter_enabled = True
        last_stats = last_display = time.monotonic()

        while True:
            # Read the camera with the largest backlog
            index = int(np.argmax(backlogs))
            name = names[index]
            packet = cameras[name].read()
            backlogs += 1
            if packet is None:
                backlogs[index] = 0
            else:
                backlogs[index] = packet.status.ring.backlog if packet.status.ring else 0
                rates[name][0] += packet.raw_event_count
                events = packet.polarity_events
                if events is not None and len(events) > 0:
                    rates[name][1] += len(events)
                    on = events["on"]
                    frame = frames[name]
                    frame[events["y"][on], events["x"][on]] = ON_COLOR
                    frame[events["y"][~on], events["x"][~on]] = OFF_COLOR

            now = time.monotonic()
            if now - last_stats >= 1.0:
                for rate in rates.values():
                    rate[2], rate[3] = rate[0] / (now - last_stats), rate[1] / (now - last_stats)
                    rate[0] = rate[1] = 0
                last_stats = now

            if now - last_display >= 1.0 / DISPLAY_FPS:
                tiles = []
                for name in names:
                    tile = downscale(frames[name], DISPLAY_DOWNSCALE).copy()
                    raw, kept = rates[name][2], rates[name][3]
                    cv2.putText(
                        tile,
                        f"{name} | {raw / 1e6:.2f} MEv/s | kept {kept / 1e6:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    tiles.append(tile)
                    frames[name].fill(0)
                cv2.imshow("openNVE", np.hstack(tiles))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("f"):
                    filter_enabled = not filter_enabled
                    for name, camera in cameras.items():
                        camera.update_filter(filters[name] if filter_enabled else opennve.FilterConfig())
                last_display = now
    finally:
        for camera in cameras.values():
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
