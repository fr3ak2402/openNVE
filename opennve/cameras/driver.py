"""USB driver of the cameras.

Windows only lets programs access a USB device that has a driver. The cameras use
Microsoft's WinUSB driver, which install_driver() installs (same as Zadig, one
administrator prompt). On Linux, the udev rules give access to the cameras
(neuromorphic-drivers-install-udev-rules, requires sudo). macOS needs no driver.

    import opennve

    opennve.ensure_driver("SE1-S4")   # installs WinUSB only if needed
    camera = opennve.open_camera(path="cameras.json", install_driver=True)
"""

from __future__ import annotations

import dataclasses
import sys
import typing

import neuromorphic_drivers as nd

from .models import get_model

# USB IDs per model (VENDOR_AND_PRODUCT_IDS of the neuromorphic_drivers devices)
_EVK_IDS = {(0x04B4, 0x00F4), (0x04B4, 0x00F5)}
MODEL_USB_IDS: dict[str, set[tuple[int, int]]] = {
    "SE1-S4": {(0x04B4, 0x00C4)},
    "EVK4": _EVK_IDS | {(0x31F7, 0x0003), (0x31F7, 0x0004), (0x1409, 0x8E00)},
    "EVK3-HD": set(_EVK_IDS),
    "EVK1-GEN4": set(_EVK_IDS),
    "CENTURYARKS-VGA": {(0x31F7, 0x0002), (0x03FD, 0x5832)},
    "DVXPLORER": {(0x152A, 0x8419)},
    "DAVIS346": {(0x152A, 0x841A)},
}

LINUX_HINT = "run neuromorphic-drivers-install-udev-rules, run the sudo command it prints and reconnect the camera"


@dataclasses.dataclass(frozen=True)
class DriverStatus:
    """Driver state of a connected camera.

    Attributes:
        device: Device type, for instance "SENSING SE1-S4-USB".
        usb_id: USB vendor and product ID, for instance "04B4:00C4" (None on Linux/macOS).
        driver: Current Windows driver, for instance "WinUSB", or None if there is none.
        ready: True if the camera can be used.
        message: Explanation if the camera is not ready, otherwise None.
    """

    device: str
    usb_id: typing.Optional[str]
    driver: typing.Optional[str]
    ready: bool
    message: typing.Optional[str] = None

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


def _usb_ids(model: typing.Optional[str]) -> typing.Optional[set[tuple[int, int]]]:
    if model is None:
        return None
    model_id = get_model(model).id
    if model_id not in MODEL_USB_IDS:
        raise ValueError(f"{model_id} is not a USB camera, it needs no driver")
    return MODEL_USB_IDS[model_id]


def _windows_driver():
    try:
        from neuromorphic_drivers import windows_driver
    except ImportError as error:
        raise RuntimeError(
            "this neuromorphic_drivers version cannot install drivers, install neuromorphic_drivers 0.19.2 or newer"
        ) from error
    return windows_driver


def driver_status(model: typing.Optional[str] = None) -> list[DriverStatus]:
    """Driver state of the connected cameras (all models, or only `model`, for instance "SE1-S4")."""
    usb_ids = _usb_ids(model)
    if sys.platform == "win32":
        result = []
        for device in _windows_driver().list_windows_devices():
            if usb_ids is not None and (device.vendor_id, device.product_id) not in usb_ids:
                continue
            ready = device.has_winusb
            if ready:
                message = None
            elif device.has_driver:
                message = f"uses the driver '{device.service}', install_driver(force=True) replaces it with WinUSB"
            else:
                message = "no driver, install_driver() installs WinUSB"
            result.append(
                DriverStatus(
                    device=device.supported_device.name,
                    usb_id=device.usb_id,
                    driver=device.service,
                    ready=ready,
                    message=message,
                )
            )
        return result

    model_name = None if model is None else get_model(model).name
    result = []
    for device in nd.list_devices():
        if model_name is not None and device.name.value != model_name:
            continue
        ready = device.error is None
        message = None
        if not ready:
            message = f"{device.error} ({LINUX_HINT})" if sys.platform.startswith("linux") else device.error
        result.append(DriverStatus(device=device.name.value, usb_id=None, driver=None, ready=ready, message=message))
    return result


def install_driver(model: typing.Optional[str] = None, force: bool = False) -> bool:
    """Installs the WinUSB driver for connected cameras (Windows).

    By default, only cameras without any driver are changed. Cameras that use another
    driver (for instance the vendor's driver) are only changed with force=True, since
    this may break the vendor's software. Shows one administrator prompt and can take a
    few minutes. Returns True if the selected cameras can be used afterwards.

    On Linux, this cannot be done without sudo: it prints the command to run and returns
    False if a camera is not accessible. On macOS, no driver is required.
    """
    usb_ids = _usb_ids(model)
    if sys.platform == "win32":
        return _windows_driver().install_windows_driver(
            usb_ids=None if usb_ids is None else sorted(usb_ids), force=force
        )
    statuses = driver_status(model)
    if all(status.ready for status in statuses):
        return True
    if sys.platform.startswith("linux"):
        print(f"Camera access denied: {LINUX_HINT}")
    return False


def ensure_driver(model: typing.Optional[str] = None) -> bool:
    """Installs the driver only if a connected camera needs it. Returns True if all
    (selected) connected cameras can be used (also if no camera is connected).

    Call it once when your program starts, for instance opennve.ensure_driver("SE1-S4").
    """
    statuses = driver_status(model)
    if all(status.ready for status in statuses):
        return True
    if sys.platform == "win32" and all(
        status.ready or status.driver is not None for status in statuses
    ):
        # Only cameras with a vendor driver are left: replacing it needs force=True
        return False
    return install_driver(model)
