"""
Runs agents (TensorFlow / TensorFlow Lite models) on the tiles or frames of a camera or a RAW file
and shows the tiles the agents selected.

    python examples/agents.py examples/cameras.json --camera left
    python examples/agents.py recording.raw --agents my_agents.json

Agents are configured in examples/agents.json (--agents), frames in examples/frames.json (--frames).
The model files named in agents.json (models/unet_int8.tflite, ...) are your own models; paths are
relative to agents.json. --agent selects some agents, for instance --agent insects.
Keys: Q / ESC exit.
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
    parser.add_argument("--agents", default="examples/agents.json", help="agent configuration file")
    parser.add_argument("--agent", action="append", help="only these agents (default: all)")
    parser.add_argument("--statistics", action="store_true", help="print segmentation statistics per frame")
    arguments = parser.parse_args()

    agents = opennve.load_agents(arguments.agents)
    if arguments.agent:
        agents = [agent for agent in agents if agent.name in arguments.agent]
    frame_configs = opennve.load_frame_configs(arguments.frames)
    for agent in agents:
        agent.load()  # load the models before the camera starts
        print(f"{agent.name}: {agent.config.model}, input {list(agent.input.shape)}")

    with open_source(arguments) as source:
        for step in opennve.run_agents(source, agents, frame_configs):
            for name, result in step.results.items():
                selected = result.selected
                print(
                    f"{step.window.start_t / 1e6:.3f} s {name}: {len(selected)} / {len(result)} selected, "
                    f"{result.inference_ms:.1f} ms"
                )
                for prediction in selected[:5]:
                    where = f"tile {prediction.tile_id}" if prediction.tile is not None else "frame"
                    score = "-" if prediction.score is None else f"{prediction.score:.2f}"
                    pixels = f", {prediction.pixel_count} pixels" if prediction.pixels is not None else ""
                    print(f"    {where}: {prediction.label or prediction.class_id} {score}{pixels}")
                if arguments.statistics:
                    result.print_statistics()
            # show the frame of the first agent with its selected tiles
            first = next(iter(step.results.values()))
            image = first.frame.to_image(draw_tiles=False)
            for prediction in first.selected:
                if prediction.pixels is not None and len(prediction.pixels):  # segmentation: target pixels
                    image[prediction.pixels["y"], prediction.pixels["x"]] = (0, 0, 255)
                if prediction.tile is not None:
                    tile = prediction.tile
                    cv2.rectangle(image, (tile.x, tile.y), (tile.x + tile.width - 1, tile.y + tile.height - 1), (0, 255, 255), 2)
            cv2.imshow("openNVE agents", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
