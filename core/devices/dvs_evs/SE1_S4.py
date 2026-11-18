from driver.EventCamera import EventCamera


class SE1S4Driver(EventCamera):
    def __init__(self, serial: str | None = None, **kwargs):
        super().__init__(model="SE1-S4", serial=serial, **kwargs)