"""
Builds a U-Net, optionally transfers the weights of another trained U-Net, and converts it to
TensorFlow Lite (int8) with a representative dataset from a RAW file, for the agents.

    python examples/networks.py recording.raw --out models
    python examples/networks.py recording.raw --weights models/unet.weights.h5 --out models
    python examples/networks.py recording.raw --source models/hsnr_unet.keras --out models

Needs TensorFlow (Python 3.14: pip install tf-nightly).
"""

import argparse
import pathlib

import opennve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help=".raw file for the representative dataset")
    parser.add_argument("--frames", default="examples/frames.json", help="frame configuration file")
    parser.add_argument("--frame", default="tiles80", help="frame configuration with the tiles")
    parser.add_argument("--classes", type=int, default=3, help="number of classes")
    parser.add_argument("--weights", help="trained weights (.weights.h5) for the U-Net")
    parser.add_argument("--source", help="trained HSNR U-Net (.keras) to transfer the weights from")
    parser.add_argument("--out", default="models", help="output folder")
    arguments = parser.parse_args()

    frame_config = opennve.load_frame_config(arguments.frames, arguments.frame)
    tile_size = frame_config.tiles.size if frame_config.tiles else 80
    parameters = opennve.NetworkParameters(arguments.classes, tile_size, tile_size, 2)

    engine = opennve.NeuralEngine(model_directory=arguments.out)
    unet = engine.add_network("unet", "UNet", parameters=parameters)
    if arguments.weights:
        unet.load_weights(arguments.weights)
    if arguments.source:
        engine.add_network("source", path=arguments.source, safe_mode=False)
        engine.transfer_weights("unet", "source").print()
    unet.summary(expand_nested=True)

    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    engine.save("unet", out / "unet.keras")
    with opennve.open_raw(arguments.raw) as raw:
        dataset = opennve.representative_dataset((raw, frame_config), input_shape=parameters.input_shape)
    for quantization in ("none", "float16", "int8"):
        path = engine.quantize("unet", quantization, representative_dataset=dataset)
        print(f"{quantization}: {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
