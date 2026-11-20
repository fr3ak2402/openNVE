"""Event cameras: configuration (JSON), discovery, USB drivers, opening and reading cameras.

Modules: camera (EventCamera, CameraPacket, open_camera, open_cameras), config (CameraConfig,
camera configuration files), discovery (list_cameras), driver (WinUSB driver installation),
models (supported camera models), devices (model-specific driver classes).
"""

from .camera import CameraPacket, EventCamera, open_camera, open_cameras
from .config import CameraConfig, load_config, load_configs, save_configs, settings_from_dict, settings_to_dict
from .devices import EVK3HDDriver, EVK4Driver, SE1S4Driver
from .discovery import CameraInfo, list_cameras, list_cameras_json
from .driver import DriverStatus, driver_status, ensure_driver, install_driver
from .models import MODELS, CameraModel, get_model

__all__ = [
    "MODELS",
    "CameraConfig",
    "CameraInfo",
    "CameraModel",
    "CameraPacket",
    "DriverStatus",
    "EVK3HDDriver",
    "EVK4Driver",
    "EventCamera",
    "SE1S4Driver",
    "driver_status",
    "ensure_driver",
    "get_model",
    "install_driver",
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
