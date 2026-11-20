"""Lists connected cameras."""

from __future__ import annotations

import dataclasses
import json
import typing

import neuromorphic_drivers as nd

from .config import CameraConfig
from .models import model_from_device_name


@dataclasses.dataclass(frozen=True)
class CameraInfo:
    """A connected camera.

    Attributes:
        model: Short model name, for instance "SE1-S4".
        device_name: Device name, for instance "SENSING SE1-S4-USB".
        serial: Serial number (None if the camera could not be read, see error).
        connection: USB/Ethernet connection, for instance "USB 3.0 SuperSpeed (5.0 Gb/s)".
        address: Bus address.
        error: Error message if the camera is detected but cannot be used (for instance
            missing permissions or driver), otherwise None.
        name: Name of the matching configuration, if configs were passed to list_cameras.
    """

    model: str
    device_name: str
    serial: typing.Optional[str]
    connection: str
    address: str
    error: typing.Optional[str] = None
    name: typing.Optional[str] = None

    @property
    def available(self) -> bool:
        return self.error is None and self.serial is not None

    def to_dict(self) -> dict[str, typing.Any]:
        return {**dataclasses.asdict(self), "available": self.available}

    def default_config(self) -> CameraConfig:
        """Configuration with default settings for this camera."""
        return CameraConfig(model=self.model, serial=self.serial, name=self.name)


def _match(info: CameraInfo, configs: typing.Sequence[CameraConfig]) -> typing.Optional[str]:
    for config in configs:
        if config.model.id == info.model and config.serial == info.serial:
            return config.name
    return None


def list_cameras(configs: typing.Sequence[CameraConfig] = ()) -> list[CameraInfo]:
    """Lists the connected cameras, sorted by model and serial.

    If configs are given, each camera gets the name of the configuration
    with the same model and serial number.
    """
    result = []
    for device in nd.list_devices():
        try:
            model = model_from_device_name(device.name).id
        except ValueError:
            continue
        info = CameraInfo(
            model=model,
            device_name=device.name.value,
            serial=device.serial,
            connection=device.connection.value,
            address=str(device.address),
            error=device.error,
        )
        name = _match(info, configs)
        if name is not None:
            info = dataclasses.replace(info, name=name)
        result.append(info)
    return sorted(result, key=lambda info: (info.model, info.serial or ""))


def list_cameras_json(configs: typing.Sequence[CameraConfig] = (), indent: typing.Optional[int] = 4) -> str:
    """Connected cameras as a JSON array."""
    return json.dumps([info.to_dict() for info in list_cameras(configs)], indent=indent)
