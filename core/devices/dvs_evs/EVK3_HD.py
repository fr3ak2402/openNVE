from driver.EventCamera import EventCamera


class EVK3HDDriver(EventCamera):
    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="EVK3-HD", serial=serial, **kwargs)