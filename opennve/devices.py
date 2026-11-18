"""Model-specific camera classes (shortcuts for EventCamera(model=...))."""

from __future__ import annotations

from .camera import EventCamera


class SE1S4Driver(EventCamera):
    """SENSING SE1-S4-USB (Sony IMX646, 1280 x 720)."""

    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="SE1-S4", serial=serial, **kwargs)


class EVK4Driver(EventCamera):
    """Prophesee EVK4 (Sony IMX636, 1280 x 720)."""

    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="EVK4", serial=serial, **kwargs)


class EVK3HDDriver(EventCamera):
    """Prophesee EVK3 HD (Sony IMX636, 1280 x 720)."""

    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="EVK3-HD", serial=serial, **kwargs)
