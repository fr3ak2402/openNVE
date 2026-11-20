"""Frames: converts the events of a time window into frames (tensors) and tiles.

Frame configurations are separate from the camera configurations. A frame configuration file
contains one or more named frame configurations (like the cameras in a camera configuration file):

    {
        "frames": [
            {
                "name": "full",
                "width": 1280,
                "height": 720,
                "mode": "time_surface",
                "time_surface": {"decay": "linear", "tau_us": 100000},
                "tiles": null
            },
            {
                "name": "tiles80",
                "mode": "time_surface",
                "time_surface": {"decay": "exponential", "tau_us": 20000},
                "tiles": {"size": 80, "min_events": 1}
            }
        ]
    }

Load it with load_frame_configs("frames.json") (list) or load_frame_config("frames.json", "tiles80"),
and pass a configuration to camera.frames(config), raw.frames(config) or window.frame(config).
Several configurations at once: camera.frame_sets(configs) or window.frames(configs).
width / height null = sensor size of the camera or file.

Modes (all frames have the shape height x width x 2, float32, channel = polarity 0 / 1):

    time_surface: time surface of the latest event per pixel and polarity, 0 where there was no event
                  decay "linear":      (t_last - start_t) / (end_t - start_t)
                  decay "exponential": exp(-(end_t - t_last) / tau_us)
    binary:       1.0 where at least one event occurred, else 0.0
    count:        number of events per pixel and polarity

Tiles split the frame into a grid of size x size pixels (row by row, the tiles at the right and
bottom edges are smaller if the frame size is not a multiple of the tile size). Tile ids, rows
and columns start at 1. Only tiles with at least min_events events are returned
(min_events = 0 returns all tiles, including empty ones).

Modules: config (FrameConfig, TileConfig, TimeSurfaceConfig, files), frame (Frame, Tile, tiles),
engine (FrameEngine).
"""

from .config import (
    FRAME_MODES,
    TIME_SURFACE_DECAYS,
    FrameConfig,
    TileConfig,
    TimeSurfaceConfig,
    as_frame_config,
    as_frame_configs,
    frame_configs_from_data,
    load_frame_config,
    load_frame_configs,
    save_frame_configs,
)
from .engine import FrameEngine
from .frame import Frame, Tile, split_into_tiles, tile_grid

__all__ = [
    "FRAME_MODES",
    "TIME_SURFACE_DECAYS",
    "Frame",
    "FrameConfig",
    "FrameEngine",
    "Tile",
    "TileConfig",
    "TimeSurfaceConfig",
    "as_frame_config",
    "as_frame_configs",
    "frame_configs_from_data",
    "load_frame_config",
    "load_frame_configs",
    "save_frame_configs",
    "split_into_tiles",
    "tile_grid",
]
