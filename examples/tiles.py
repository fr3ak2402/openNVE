"""
Frames and tiles from a camera or a RAW file, built with the "frame" configuration.

    python examples/tiles.py examples/cameras.json left        # live camera "left"
    python examples/tiles.py recording.raw                      # RAW file (default frame settings + 80 px tiles)
    python examples/tiles.py recording.raw examples/cameras.json left

Keys: Q / ESC exit, T show/hide tile borders.
"""

import sys

import cv2

import opennve


def open_source(arguments):
    if arguments[0].lower().endswith(".raw"):
        config_path = arguments[1] if len(arguments) > 1 else None
        name = arguments[2] if len(arguments) > 2 else None
        frame = None if config_path else {"tiles": {"size": 80}}
        return opennve.open_raw(arguments[0], config_path=config_path, name=name, frame=frame, speed=1.0)
    return opennve.open_camera(path=arguments[0], name=arguments[1] if len(arguments) > 1 else None)


def main():
    draw_tiles = True
    with open_source(sys.argv[1:]) as source:
        for frame in source.frames():  # one frame per time window (40 ms by default)
            # frame.data: height x width x 2 (float32), frame.tiles: selected tiles, frame.events: x, y, p, t
            for tile in frame.tiles:
                pass  # for instance: agent.process(tile.id, tile.data)
            image = frame.to_image(draw_tiles=draw_tiles)
            cv2.putText(
                image,
                f"{frame.camera} | {frame.start_t / 1e6:.3f} s | {frame.event_count} events | "
                f"{len(frame.tiles)} tiles | {frame.mode}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("openNVE tiles", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                draw_tiles = not draw_tiles
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
