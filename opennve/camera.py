from __future__ import annotations

import dataclasses
import time
import typing
from typing import Any, Iterator

import neuromorphic_drivers as nd
import numpy as np

from .config import CameraConfig, PathLike, load_config, settings_to_dict
from .driver import MODEL_USB_IDS, ensure_driver
from .filters import EventFilter, FilterConfig
from .frames import Frame, FrameConfig, FrameEngine, as_frame_config, as_frame_configs
from .output import OutputConfig, to_csv, to_rows
from .window import EventWindow, WindowAccumulator, WindowConfig
from .models import get_model


@dataclasses.dataclass
class CameraPacket:
    """Events read from a camera, after the software filter.

    Attributes:
        camera: Name of the camera.
        status: neuromorphic_drivers status (ring buffer backlog, etc.).
        polarity_events: Filtered polarity events (fields t, x, y, on), or None.
        trigger_events: External trigger events, or None.
        raw_event_count: Number of polarity events before the filter.
        packet: The original neuromorphic_drivers packet.
        output_config: Output format of the camera configuration ("output" section).
    """

    camera: str
    status: Any
    polarity_events: typing.Optional[np.ndarray]
    trigger_events: typing.Optional[np.ndarray]
    raw_event_count: int
    packet: Any
    output_config: OutputConfig = dataclasses.field(default_factory=OutputConfig, repr=False)

    @property
    def events(self) -> np.ndarray:
        """Filtered polarity events in the configured output format, for instance with
        "fields": ["x", "y", "p", "t"] a structured array with fields x, y, p, t.
        The camera name is not part of the events, it is self.camera (once per packet)."""
        if self.polarity_events is None:
            return np.empty(0, dtype=self.output_config.dtype())
        return self.output_config.format(self.polarity_events)

    def rows(self) -> list[tuple]:
        """Events in the output format as tuples, for instance [(12, 40, 1, 1003), ...]."""
        return to_rows(self.events)

    def to_csv(self, header: bool = False, delimiter: str = ",", camera: bool = False) -> str:
        """Events in the output format as CSV text, for instance "12,40,1,1003".
        With camera=True, the first line is "# camera: <name>"."""
        return to_csv(
            self.events, header=header, delimiter=delimiter, camera=self.camera if camera else None
        )


class EventCamera:
    """
    Opens an event camera directly during initialization.

    The camera can be closed explicitly using close() or automatically
    by using the class as a context manager.

    Use EventCamera.from_config (or opennve.open_camera) to open a camera
    from a JSON configuration (name, model, serial, settings, filter).

    Note:
        The camera must only be read by one thread at a time.
    """

    def __init__(
        self,
        model: str,
        serial: str | None = None,
        *,
        configuration: Any = None,
        filter: FilterConfig | dict | None = None,
        output: OutputConfig | dict | None = None,
        window: WindowConfig | dict | None = None,
        name: str | None = None,
        iterator_timeout: float = 0.02,
    ) -> None:

        # Validate the model, settings and filter
        config = CameraConfig(
            model=get_model(model),
            name=name,
            serial=serial,
            settings=configuration,
            filter=filter if filter is not None else FilterConfig(),
            output=output if output is not None else OutputConfig(),
            window=window,
        )

        # Initialize internal device references
        self._context = None
        self._device = None
        self._iterator = None
        self._filter: EventFilter | None = None
        self._accumulator: WindowAccumulator | None = None

        # Open the selected camera
        context = nd.open(
            configuration=config.settings,
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

        # Store the actual serial number (serial may be None = first camera of this model)
        self.config = dataclasses.replace(config, serial=device.serial())
        if name is None:
            self.config.name = f"{self.config.model.id}-{self.config.serial}"
        self._build_filter()

    @classmethod
    def from_config(cls, config: CameraConfig | dict | str, *, iterator_timeout: float = 0.02) -> "EventCamera":
        """Opens a camera from a CameraConfig, a dict or a JSON string."""
        if isinstance(config, str):
            config = CameraConfig.from_json(config)
        elif isinstance(config, dict):
            config = CameraConfig.from_dict(config)
        return cls(
            model=config.model.id,
            serial=config.serial,
            configuration=config.settings,
            filter=config.filter,
            output=config.output,
            window=config.window,
            name=config.name,
            iterator_timeout=iterator_timeout,
        )

    # Information

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model.id

    @property
    def configuration(self) -> Any:
        """Hardware settings (neuromorphic_drivers Configuration)."""
        return self.config.settings

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

    def biases_bounds(self) -> dict[str, Any] | None:
        """Allowed bias values {bias: {"minimum": .., "maximum": ..}}, None if the model has no biases."""
        if not hasattr(self.device, "biases_bounds"):
            return None
        bounds = self.device.biases_bounds()
        return {
            field.name: dataclasses.asdict(getattr(bounds, field.name))
            for field in dataclasses.fields(bounds)
        }

    def info(self) -> dict[str, Any]:
        """Camera information and current configuration, as a JSON-compatible dict."""
        width, height = self.resolution
        return {
            **self.config.to_dict(),
            "resolution": {"width": width, "height": height},
            "biases_bounds": self.biases_bounds(),
        }

    # Reading

    def read_packet(self):
        """
        Read the next event packet from the camera (unfiltered).

        Returns:
            A tuple containing (status, packet).
            The packet may be None if a timeout occurs.
        """
        if self._iterator is None:
            raise RuntimeError("Camera is closed.")

        return next(self._iterator)

    def packets(self) -> Iterator:
        """
        Continuously yield unfiltered (status, packet) tuples while the camera is open.

        Stops when the camera is closed or the device iterator ends.
        """
        while self.is_open:
            try:
                yield self.read_packet()
            except StopIteration:
                return

    def read(self) -> CameraPacket | None:
        """Reads the next packet and applies the software filter. Returns None on timeout."""
        status, packet = self.read_packet()
        if packet is None:
            return None
        polarity_events = getattr(packet, "polarity_events", None)
        raw_event_count = 0 if polarity_events is None else len(polarity_events)
        if polarity_events is not None and self._filter is not None:
            polarity_events = self._filter(polarity_events)
        return CameraPacket(
            camera=self.name,
            status=status,
            polarity_events=polarity_events,
            trigger_events=getattr(packet, "trigger_events", None),
            raw_event_count=raw_event_count,
            packet=packet,
            output_config=self.config.output,
        )

    def events(self) -> Iterator[CameraPacket]:
        """Continuously yield filtered packets while the camera is open (timeouts are skipped)."""
        while self.is_open:
            try:
                camera_packet = self.read()
            except StopIteration:
                return
            if camera_packet is not None:
                yield camera_packet

    # Time windows

    def _window_accumulator(self, duration_ms: float | None) -> WindowAccumulator:
        if duration_ms is None:
            if self.config.window is None:
                raise ValueError(
                    "no window duration: set \"window\": {\"duration_ms\": ...} in the configuration "
                    "or pass duration_ms"
                )
            duration_us = self.config.window.duration_us
        else:
            window = WindowConfig(duration_ms=duration_ms)
            window.validate()
            duration_us = window.duration_us
        if self._accumulator is None or self._accumulator.duration_us != duration_us:
            width, height = self.resolution
            self._accumulator = WindowAccumulator(
                duration_us, self.name, width, height, self.config.output
            )
        self._accumulator.camera = self.name
        self._accumulator.output_config = self.config.output
        return self._accumulator

    def read_windows(self, duration_ms: float | None = None) -> list[EventWindow]:
        """Reads one packet and returns the time windows completed by it (often none).

        Use this to read several cameras in one loop. duration_ms overrides the
        "window" section of the configuration.
        """
        accumulator = self._window_accumulator(duration_ms)
        packet = self.read()
        if packet is None:
            return []
        current_t = packet.status.ring.current_t if packet.status.ring is not None else None
        return accumulator.push(packet.polarity_events, current_t)

    def windows(self, duration_ms: float | None = None) -> Iterator[EventWindow]:
        """Continuously yield complete time windows (events of duration_ms each, filtered).

        Example:
            for window in camera.windows():          # duration from the configuration
                frame = window.to_frame("color")     # or "count", "polarity", "binary"
        """
        while self.is_open:
            try:
                windows = self.read_windows(duration_ms)
            except StopIteration:
                return
            yield from windows

    # Frames

    def read_frames(
        self, config: FrameConfig | dict | str | None = None, duration_ms: float | None = None
    ) -> list[Frame]:
        """Reads one packet and returns the frames of the time windows completed by it (often none).

        config: frame configuration (see frames()).
        """
        engine = self._frame_engine(config)
        return [engine.build_window(window) for window in self.read_windows(duration_ms)]

    def frames(
        self, config: FrameConfig | dict | str | None = None, duration_ms: float | None = None
    ) -> Iterator[Frame]:
        """Continuously yield one frame per time window (40 ms by default).

        config: frame configuration, for instance opennve.load_frame_config("frames.json", "tiles80"),
            a FrameConfig, a dict or a JSON string. Default: sensor size, linear time surface, no tiles.
        Each frame has frame.data, frame.tiles and frame.events (the events of the window).

        Example:
            tiles80 = opennve.load_frame_config("frames.json", "tiles80")
            for frame in camera.frames(tiles80):
                for tile in frame.tiles:
                    agent.process(tile.id, tile.data)
        """
        engine = self._frame_engine(config)
        for window in self.windows(duration_ms):
            yield engine.build_window(window)

    def read_frame_sets(self, configs, duration_ms: float | None = None) -> list[dict[str, Frame]]:
        """Like read_frames with several frame configurations: one {name: Frame} per completed window."""
        engines = self._frame_engines(configs)
        return [
            {name: engine.build_window(window) for name, engine in engines.items()}
            for window in self.read_windows(duration_ms)
        ]

    def frame_sets(self, configs, duration_ms: float | None = None) -> Iterator[dict[str, Frame]]:
        """Continuously yield one frame per frame configuration and time window: {name: Frame}.

        configs: list of frame configurations (for instance opennve.load_frame_configs("frames.json"))
            or a dict {name: configuration}. All frames of one set come from the same time window.

        Example:
            for frames in camera.frame_sets(opennve.load_frame_configs("frames.json")):
                full, tiles = frames["full"], frames["tiles80"]
        """
        engines = self._frame_engines(configs)
        for window in self.windows(duration_ms):
            yield {name: engine.build_window(window) for name, engine in engines.items()}

    def _frame_engine(self, config) -> FrameEngine:
        width, height = self.resolution
        return FrameEngine.from_config(as_frame_config(config), width, height)

    def _frame_engines(self, configs) -> dict[str, FrameEngine]:
        width, height = self.resolution
        return {config.name: FrameEngine.from_config(config, width, height) for config in as_frame_configs(configs)}

    def update_window(self, window: WindowConfig | dict | None) -> None:
        """Replace the window duration, for instance {"duration_ms": 20} (restarts the current window)."""
        if isinstance(window, dict):
            window = WindowConfig.from_dict(window)
        if window is not None:
            window.validate()
        self.config.window = window
        self._accumulator = None

    # Configuration

    def _build_filter(self) -> None:
        if self.config.filter.is_active and self.config.model.id != "TRITON":
            width, height = self.resolution
            self._filter = EventFilter(self.config.filter, width, height)
        else:
            self._filter = None

    def update_configuration(self, configuration: Any) -> None:
        """
        Update the hardware settings.

        Args:
            configuration: A neuromorphic_drivers Configuration of the camera
                model, or a JSON-compatible dict (only the given fields change).

        Raises:
            TypeError: If the configuration type does not match the camera model.
        """
        if isinstance(configuration, dict):
            configuration = CameraConfig(
                model=self.config.model,
                settings={**settings_to_dict(self.config.settings), **configuration},
            ).settings
        elif not isinstance(configuration, self.config.model.configuration_type):
            raise TypeError(f"Configuration is not compatible with {self.model}.")

        self.device.update_configuration(configuration)
        self.config.settings = configuration

    def update_filter(self, filter: FilterConfig | dict) -> None:
        """Replace the software filter settings."""
        if isinstance(filter, dict):
            filter = FilterConfig.from_dict(filter)
        filter.validate()
        self.config.filter = filter
        self._build_filter()

    def update_output(self, output: OutputConfig | dict) -> None:
        """Replace the output format, for instance {"fields": ["x", "y", "p", "t"]}."""
        if isinstance(output, dict):
            output = OutputConfig.from_dict(output)
        output.validate()
        self.config.output = output

    def apply_config(self, config: CameraConfig | dict | str) -> None:
        """Apply a complete camera configuration (settings, filter, output and name) to the open camera."""
        if isinstance(config, str):
            config = CameraConfig.from_json(config)
        elif isinstance(config, dict):
            config = CameraConfig.from_dict(config)
        if config.model.id != self.model:
            raise ValueError(f"the configuration is for {config.model.id}, the camera is a {self.model}")
        if config.serial is not None and config.serial != self.config.serial:
            raise ValueError(f"the configuration is for serial {config.serial}, the camera is {self.config.serial}")
        self.update_configuration(config.settings)
        self.update_filter(config.filter)
        self.update_output(config.output)
        self.update_window(config.window)
        self.config.name = config.name

    # Lifecycle

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


def open_camera(
    config: CameraConfig | dict | str | None = None,
    *,
    path: PathLike | None = None,
    name: str | None = None,
    iterator_timeout: float = 0.02,
    install_driver: bool = False,
) -> EventCamera:
    """Opens a camera from a configuration (object, dict or JSON string) or from a file.

    With install_driver=True, the USB driver is installed first if the camera needs it
    (Windows: WinUSB, one administrator prompt, see opennve.ensure_driver).

    Examples:
        open_camera({"model": "SE1-S4", "serial": "05110634"})
        open_camera(path="cameras.json", name="left", install_driver=True)
    """
    if path is not None:
        config = load_config(path, name)
    if config is None:
        raise ValueError("config or path is required")
    if isinstance(config, str):
        config = CameraConfig.from_json(config)
    elif isinstance(config, dict):
        config = CameraConfig.from_dict(config)
    if install_driver:
        _ensure_drivers([config])
    return EventCamera.from_config(config, iterator_timeout=iterator_timeout)


def _ensure_drivers(configs: typing.Sequence[CameraConfig]) -> None:
    for model in sorted({config.model.id for config in configs}):
        if model in MODEL_USB_IDS and not ensure_driver(model):
            raise RuntimeError(
                f"the driver for {model} is not ready, see opennve.driver_status(\"{model}\")"
            )


def _is_clock_secondary(config: CameraConfig) -> bool:
    clock = getattr(config.settings, "clock", None)
    return clock is not None and clock.name == "EXTERNAL"


def open_cameras(
    configs: typing.Sequence[CameraConfig],
    *,
    iterator_timeout: float = 0.02,
    secondary_start_delay: float = 1.0,
    install_driver: bool = False,
) -> dict[str, EventCamera]:
    """Opens several cameras, returns {name: camera}.

    Cameras with "clock": "external" (synchronization secondaries) are opened first,
    so that they listen before the primary ("internal_with_output_enabled") starts its clock.
    If opening one camera fails, the cameras opened so far are closed.
    With install_driver=True, missing USB drivers are installed first (see open_camera).
    """
    if install_driver:
        _ensure_drivers(configs)
    order = sorted(configs, key=lambda config: not _is_clock_secondary(config))
    cameras: dict[str, EventCamera] = {}
    try:
        for config in order:
            if config.name in cameras:
                raise ValueError(f"duplicate camera name {config.name!r}")
            if (
                cameras
                and not _is_clock_secondary(config)
                and all(_is_clock_secondary(camera.config) for camera in cameras.values())
            ):
                time.sleep(secondary_start_delay)
            cameras[config.name] = EventCamera.from_config(config, iterator_timeout=iterator_timeout)
    except BaseException:
        for camera in cameras.values():
            camera.close()
        raise
    return {config.name: cameras[config.name] for config in configs}
