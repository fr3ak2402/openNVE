from driver.EventCamera import EventCamera


class EVK4Driver(EventCamera):
    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="EVK4", serial=serial, **kwargs)