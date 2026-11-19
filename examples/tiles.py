"""
Frames and tiles from a camera or a RAW file, built with a named frame configuration.

    python examples/tiles.py examples/cameras.json --camera left --frame tiles80
    python examples/tiles.py recording.raw --frame exponential
    python examples/tiles.py recording.raw --frames my_frames.json --frame full

The frame configurations are read from examples/frames.json (--frames), separately from the
camera configuration. Keys: Q / ESC exit, T show/hide tile borders, F next frame configuration.
"""

import argparse

import cv2

import opennve


def open_source(arguments):
    if arguments.source.lower().endswith(".raw"):
        return opennve.open_raw(arguments.source, speed=1.0)
    return opennve.open_camera(path=arguments.source, name=arguments.camera)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="camera configuration file or .raw file")
    parser.add_argument("--camera", help="camera name in the configuration file")
    parser.add_argument("--frames", default="examples/frames.json", help="frame configuration file")
    parser.add_argument("--frame", help="name of the frame configuration (default: the first)")
    arguments = parser.parse_args()

    frame_configs = opennve.load_frame_configs(arguments.frames)
    names = [config.name for config in frame_configs]
    index = names.index(arguments.frame) if arguments.frame else 0
    draw_tiles = True
    with open_source(arguments) as source:
        for window in source.windows():  # one time window per 40 ms (camera "window" setting)
            frame = window.frame(frame_configs[index])
            # frame.data: height x width x 2 (float32), frame.tiles: selected tiles, frame.events: x, y, p, t
            for tile in frame.tiles:
                pass  # for instance: agent.process(tile.id, tile.data)
            image = frame.to_image(draw_tiles=draw_tiles)
            cv2.putText(
                image,
                f"{frame.camera} | {frame.name} | {frame.start_t / 1e6:.3f} s | {frame.event_count} events | "
                f"{len(frame.tiles)} tiles",
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
            if key == ord("f"):
                index = (index + 1) % len(frame_configs)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
