"""
Plays a recording (RAW file with EVT 2.0, 2.1 or 3.0, or an HDF5 file) as frames, with the settings
of a camera configuration.

    python examples/raw_player.py recording.raw
    python examples/raw_player.py recording.h5            (HDF5 with x, y, p, t, optionally l)
    python examples/raw_player.py recording.raw examples/cameras.json left

Keys: Q / ESC exit, M switch frame mode, + / - playback speed.
"""

import sys

import cv2

import opennve

RAW_PATH = sys.argv[1]
CONFIG_PATH = sys.argv[2] if len(sys.argv) > 2 else None
CAMERA_NAME = sys.argv[3] if len(sys.argv) > 3 else None
MODES = ("color", "binary")


def main():
    mode = 0
    speed = 1.0  # 1.0 = real time
    config = opennve.load_config(CONFIG_PATH, CAMERA_NAME) if CONFIG_PATH else None
    with opennve.open_recording(RAW_PATH, config, speed=speed) as raw:  # .raw or HDF5 (.h5)
        print(raw.info())
        for window in raw.windows():  # 40 ms by default, or "window" of the configuration
            image = window.to_frame(MODES[mode])
            cv2.putText(
                image,
                f"{window.camera} | {window.start_t / 1e6:.3f} s | {len(window)} events | x{raw.speed:g}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("openNVE RAW player", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = (mode + 1) % len(MODES)
            if key in (ord("+"), ord("=")):
                raw.speed = min(raw.speed * 2, 64)
            if key == ord("-"):
                raw.speed = max(raw.speed / 2, 1 / 64)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
