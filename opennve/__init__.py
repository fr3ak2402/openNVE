"""openNVE - Open Neuromorphic Vision Engine.

    import opennve

    opennve.list_cameras()                       # connected cameras
    config = opennve.CameraConfig.from_json(...) # or opennve.load_configs("cameras.json")
    with opennve.open_camera(config) as camera:
        for packet in camera.events():
            print(packet.events)  # structured array in the configured output format (x, y, p, t)
"""

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
from .filters import BackgroundActivityConfig, BackgroundActivityFilter, EventFilter, FilterConfig
from .models import MODELS, CameraModel, get_model
from .output import OutputConfig, to_csv, to_rows
from .window import EventWindow, WindowAccumulator, WindowConfig

__version__ = "0.12.1"

try:  # wheels with embedded neuromorphic_drivers (tools/bundle_wheel.py)
    from ._bundle import NEUROMORPHIC_DRIVERS_VERSION
except ImportError:
    import importlib.metadata as _metadata

    try:
        NEUROMORPHIC_DRIVERS_VERSION = _metadata.version("neuromorphic_drivers")
    except _metadata.PackageNotFoundError:
        NEUROMORPHIC_DRIVERS_VERSION = None

__all__ = [
    "NEUROMORPHIC_DRIVERS_VERSION",
    "MODELS",
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
