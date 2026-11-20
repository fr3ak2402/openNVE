"""openNVE - Open Neuromorphic Vision Engine.

    import opennve

    opennve.list_cameras()                       # connected cameras
    config = opennve.CameraConfig.from_json(...) # or opennve.load_configs("cameras.json")
    with opennve.open_camera(config) as camera:
        for packet in camera.events():
            print(packet.events)  # structured array in the configured output format (x, y, p, t)

    frame_configs = opennve.load_frame_configs("frames.json")  # named frame configurations
    with opennve.open_camera(config) as camera:
        for frame in camera.frames(frame_configs[0]):  # one frame per time window (40 ms)
            print(frame.name, frame.data.shape, len(frame.tiles))
"""

from .agents import (
    Agent,
    AgentConfig,
    AgentResult,
    AgentStep,
    Prediction,
    PredictionConfig,
    PreprocessConfig,
    load_agent,
    load_agent_config,
    load_agent_configs,
    load_agents,
    process_window,
    run_agents,
    save_agent_configs,
    TaskInput,
    list_backends,
    list_tasks,
    register_backend,
    register_task,
)
from .networks import (
    NetworkParameters,
    Network,
    NeuralEngine,
    compare_weights,
    custom_objects,
    list_networks,
    register_custom_objects,
    register_network,
    register_weight_transfer,
    representative_dataset,
)
from .camera import CameraPacket, EventCamera, open_camera, open_cameras
from .config import (
    CameraConfig,
    load_config,
    load_configs,
    save_configs,
    settings_from_dict,
    settings_to_dict,
)
from .devices import EVK3HDDriver, EVK4Driver, SE1S4Driver
from .discovery import CameraInfo, list_cameras, list_cameras_json
from .driver import DriverStatus, driver_status, ensure_driver, install_driver
from .frames import (
    FRAME_MODES,
    TIME_SURFACE_DECAYS,
    Frame,
    FrameConfig,
    FrameEngine,
    Tile,
    TileConfig,
    TimeSurfaceConfig,
    as_frame_config,
    as_frame_configs,
    load_frame_config,
    load_frame_configs,
    save_frame_configs,
    split_into_tiles,
    tile_grid,
)
from .filters import BackgroundActivityConfig, BackgroundActivityFilter, EventFilter, FilterConfig
from .models import MODELS, CameraModel, get_model
from .output import OutputConfig, to_csv, to_rows
from .raw import RawHeader, RawReader, open_raw, read_header, read_raw
from .window import EventWindow, WindowAccumulator, WindowConfig

__version__ = "0.6.2"

try:  # wheels with embedded neuromorphic_drivers (tools/bundle_wheel.py)
    from ._bundle import NEUROMORPHIC_DRIVERS_VERSION
except ImportError:
    import importlib.metadata as _metadata

    try:
        NEUROMORPHIC_DRIVERS_VERSION = _metadata.version("neuromorphic_drivers")
    except _metadata.PackageNotFoundError:
        NEUROMORPHIC_DRIVERS_VERSION = None

__all__ = [
    "TaskInput",
    "list_backends",
    "list_tasks",
    "register_backend",
    "register_task",
    "Network",
    "NetworkParameters",
    "NeuralEngine",
    "compare_weights",
    "custom_objects",
    "list_networks",
    "register_custom_objects",
    "register_network",
    "register_weight_transfer",
    "representative_dataset",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "Prediction",
    "PredictionConfig",
    "PreprocessConfig",
    "load_agent",
    "load_agent_config",
    "load_agent_configs",
    "load_agents",
    "process_window",
    "run_agents",
    "save_agent_configs",
    "NEUROMORPHIC_DRIVERS_VERSION",
    "MODELS",
    "FRAME_MODES",
    "Frame",
    "FrameConfig",
    "FrameEngine",
    "Tile",
    "TileConfig",
    "TimeSurfaceConfig",
    "TIME_SURFACE_DECAYS",
    "as_frame_config",
    "as_frame_configs",
    "load_frame_config",
    "load_frame_configs",
    "save_frame_configs",
    "split_into_tiles",
    "tile_grid",
    "RawHeader",
    "RawReader",
    "open_raw",
    "read_header",
    "read_raw",
    "DriverStatus",
    "driver_status",
    "ensure_driver",
    "install_driver",
    "OutputConfig",
    "EventWindow",
    "WindowAccumulator",
    "WindowConfig",
    "to_csv",
    "to_rows",
    "BackgroundActivityConfig",
    "BackgroundActivityFilter",
    "CameraConfig",
    "CameraInfo",
    "CameraModel",
    "CameraPacket",
    "EVK3HDDriver",
    "EVK4Driver",
    "EventCamera",
    "EventFilter",
    "FilterConfig",
    "SE1S4Driver",
    "get_model",
    "list_cameras",
    "list_cameras_json",
    "load_config",
    "load_configs",
    "open_camera",
    "open_cameras",
    "save_configs",
    "settings_from_dict",
    "settings_to_dict",
]
