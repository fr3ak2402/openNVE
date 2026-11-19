"""
Collects the events of a camera in time windows and shows one frame per window.

    python examples/frames.py examples/cameras.json left

The window duration is "window": {"duration_ms": ...} in the configuration.
Keys: Q / ESC exit, M switch frame mode (color, count, polarity, binary).
"""

import sys

import cv2
import numpy as np

import opennve

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "examples/cameras.json"
CAMERA_NAME = sys.argv[2] if len(sys.argv) > 2 else None
MODES = ("color", "count", "polarity", "binary")


def to_image(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "count":
        return np.clip(frame.astype(np.float32) * 64, 0, 255).astype(np.uint8)
    if mode == "polarity":
        return np.clip(frame.astype(np.float32) * 64 + 128, 0, 255).astype(np.uint8)
    return frame  # color and binary are already images


def main():
    mode = 0
    with opennve.open_camera(path=CONFIG_PATH, name=CAMERA_NAME) as camera:
        window_config = camera.config.window or opennve.WindowConfig()  # 40 ms
        print(f"{camera.name}: {window_config.duration_ms} ms windows")
        for window in camera.windows(window_config.duration_ms):
            # window.events: events of this window in the output format (x, y, p, t)
            frame = window.to_frame(MODES[mode])
            image = to_image(frame, MODES[mode])
            cv2.putText(
                image,
                f"{window.camera} | {window.start_t / 1e6:.3f} s | {len(window)} events | {MODES[mode]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("openNVE frames", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = (mode + 1) % len(MODES)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
