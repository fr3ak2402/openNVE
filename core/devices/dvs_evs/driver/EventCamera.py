from __future__ import annotations
from typing import Any, Iterator

import neuromorphic_drivers as nd


def _config_type(model: str):
    """Return the configuration class for the specified camera model."""
    types = {
        "EVK4": nd.prophesee_evk4.Configuration,
        "EVK3-HD": nd.prophesee_evk3_hd.Configuration,
        "SE1-S4": nd.sensing_se1_s4.Configuration,
    }

    try:
        return types[model]
    except KeyError as exc:
        raise ValueError(
            f"Unknown camera model: {model!r}. "
            f"Supported models: {', '.join(types)}"
        ) from exc


class EventCamera:
    """
    Opens an event camera directly during initialization.

    The camera can be closed explicitly using close() or automatically
    by using the class as a context manager.

    Note:
        The camera must only be read by one thread at a time.
    """

    def __init__(
        self,
        model: str,
        serial: str | None = None,
        *,
        configuration: Any = None,
        iterator_timeout: float = 0.02,
    ) -> None:

        # Normalize the camera model name
        model = model.strip().upper().replace("_", "-")

        # Get the expected configuration type
        expected = _config_type(model)

        # Create or validate the camera configuration
        if configuration is None:
            configuration = expected()
        elif not isinstance(configuration, expected):
            raise TypeError(
                f"Configuration must be of type "
                f"{expected.__name__} for {model}."
            )

        # Store camera information
        self.model = model
        self.requested_serial = serial
        self.configuration = configuration

        # Initialize internal device references
        self._context = None
        self._device = None
        self._iterator = None

        # Open the selected camera
        context = nd.open(
            configuration=configuration,
            serial=serial,
            iterator_timeout=iterator_timeout,
        )

        try:
            device = context.__enter__()
            iterator = iter(device)
        except BaseException:
            context.__exit__(None, None, None)
            raise

        self._context = context
        self._device = device
        self._iterator = iterator

    @property
    def is_open(self) -> bool:
        """Return True if the camera connection is active."""
        return self._device is not None

    @property
    def device(self) -> Any:
        """Return the underlying neuromorphic driver device."""
        if not self.is_open:
            raise RuntimeError("Camera is closed.")

        return self._device

    @property
    def serial(self) -> str:
        """Return the serial number of the connected camera."""
        return self.device.serial()

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the camera resolution as (width, height)."""
        props = self.device.properties()
        return (props.width, props.height)

    def read_packet(self):
        """
        Read the next event packet from the camera.

        Returns:
            A tuple containing (status, packet).
            The packet may be None if a timeout occurs.
        """
        if self._iterator is None:
            raise RuntimeError("Camera is closed.")

        return next(self._iterator)

    def packets(self) -> Iterator:
        """
        Continuously yield event packets while the camera is open.

        Stops when the camera is closed or the device iterator ends.
        """
        while self.is_open:
            try:
                yield self.read_packet()
            except StopIteration:
                return

    def update_configuration(self, configuration: Any) -> None:
        """
        Update the camera configuration.

        Args:
            configuration: A configuration object compatible
                with the selected camera model.

        Raises:
            TypeError: If the configuration type does not match
                the camera model.
        """
        if not isinstance(configuration, _config_type(self.model)):
            raise TypeError(
                f"Configuration is not compatible with {self.model}."
            )

        self.device.update_configuration(configuration)
        self.configuration = configuration

    def close(self) -> None:
        """Close the camera and release all associated resources."""
        context = self._context

        self._context = None
        self._device = None
        self._iterator = None

        if context is not None:
            context.__exit__(None, None, None)

    def __enter__(self) -> "EventCamera":
        """Enter the camera context manager."""
        if not self.is_open:
            raise RuntimeError("Camera is closed.")

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Close the camera when leaving the context manager."""
        self.close()
        return False