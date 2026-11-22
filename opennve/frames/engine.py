"""FrameEngine: builds frame tensors (time surface, binary, count) and tiles from events."""

from __future__ import annotations

import typing

import numpy as np

from .config import FRAME_MODES, TIME_SURFACE_DECAYS, FrameConfig, TileConfig, TimeSurfaceConfig, as_frame_config
from .frame import Frame, Tile, split_into_tiles



def _columns(events) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """x, y, p, t as numpy arrays from a structured array (fields x, y, t and p or on),
    a pandas DataFrame or a dict of arrays."""
    names = getattr(getattr(events, "dtype", None), "names", None)
    if names is None and hasattr(events, "columns"):
        names = tuple(events.columns)  # pandas DataFrame
    elif names is None and isinstance(events, dict):
        names = tuple(events)
    if names is None:
        raise TypeError("events must be a numpy structured array, a pandas DataFrame or a dict of arrays")
    missing = {"x", "y", "t"} - set(names)
    if missing or not ({"p", "on"} & set(names)):
        raise ValueError(f"events need the fields x, y, t and p (or on), got {', '.join(names)}")

    def column(name, dtype):
        value = events[name]
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        return np.asarray(value).astype(dtype, copy=False)

    polarity = column("p" if "p" in names else "on", np.int64)
    return column("x", np.int64), column("y", np.int64), polarity, column("t", np.float64)


class FrameEngine:
    """Builds frames and tiles from events.

    Args:
        width, height: Frame size in pixels. Events outside the frame are ignored.
        tile_size: Tile size in pixels, or None for no tiles.
        mode: "time_surface", "binary" or "count".
        min_events: Minimum number of events of a returned tile (0 = all tiles).
        decay: Time surface decay, "linear" or "exponential".
        tau_us: Time constant of the exponential decay in µs.

    The original arguments frame_width, frame_height and frame_tile_size are accepted as well,
    so FrameEngine(frame_width=1280, frame_height=720, frame_tile_size=80) works as before.
    """

    def __init__(
        self,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        tile_size: typing.Optional[int] = None,
        mode: str = "time_surface",
        min_events: int = 1,
        decay: str = "linear",
        tau_us: float = 100_000,
        *,
        frame_width: typing.Optional[int] = None,
        frame_height: typing.Optional[int] = None,
        frame_tile_size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
    ):
        width = frame_width if width is None else width
        height = frame_height if height is None else height
        tile_size = frame_tile_size if tile_size is None else tile_size
        if width is None or height is None:
            raise ValueError("width and height are required")
        config = FrameConfig(
            width=int(width),
            height=int(height),
            mode=mode,
            tiles=None if tile_size is None else TileConfig(size=int(tile_size), min_events=int(min_events)),
            time_surface=TimeSurfaceConfig(decay=decay, tau_us=tau_us),
            name=name,
        )
        config.validate()
        self.config = config
        self.name = name
        self.width = config.width
        self.height = config.height
        self.mode = config.mode
        self.decay = config.time_surface.decay
        self.tau_us = float(config.time_surface.tau_us)
        self.tile_size = None if config.tiles is None else config.tiles.size
        self.min_events = 1 if config.tiles is None else config.tiles.min_events
        size = self.tile_size or 1
        self.tile_cols = (self.width + size - 1) // size if self.tile_size else 0
        self.tile_rows = (self.height + size - 1) // size if self.tile_size else 0

    @classmethod
    def from_config(
        cls, config: typing.Union[FrameConfig, dict, str, None], sensor_width: int, sensor_height: int
    ) -> "FrameEngine":
        """Engine for a FrameConfig (or dict / JSON); width/height None means the sensor size."""
        config = as_frame_config(config)
        return cls(
            config.width or sensor_width,
            config.height or sensor_height,
            tile_size=None if config.tiles is None else config.tiles.size,
            mode=config.mode,
            min_events=1 if config.tiles is None else config.tiles.min_events,
            decay=config.time_surface.decay,
            tau_us=config.time_surface.tau_us,
            name=config.name,
        )

    # Frames

    def build(
        self,
        events,
        start_t: int,
        end_t: int,
        camera: str = "",
        window: typing.Any = None,
        mode: typing.Optional[str] = None,
        decay: typing.Optional[str] = None,
    ) -> Frame:
        """Builds the frame and the tiles of the events in [start_t, end_t).
        mode and decay override the engine settings for this frame."""
        mode = mode or self.mode
        decay = decay or self.decay
        if mode not in FRAME_MODES:
            raise ValueError(f"mode must be one of {', '.join(FRAME_MODES)}")
        if decay not in TIME_SURFACE_DECAYS:
            raise ValueError(f"decay must be one of {', '.join(TIME_SURFACE_DECAYS)}")
        if end_t <= start_t:
            raise ValueError("end_t must be greater than start_t")
        x, y, p, t = _columns(events)
        valid = (
            (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            & (p >= 0) & (p <= 1) & (t >= start_t) & (t < end_t)
        )
        if not np.all(valid):
            x, y, p, t = x[valid], y[valid], p[valid], t[valid]
        data = self._frame(x, y, p, t, float(start_t), float(end_t - start_t), mode, decay)
        tiles = self._tiles(data, x, y)
        return Frame(
            camera=camera,
            start_t=int(start_t),
            end_t=int(end_t),
            mode=mode,
            data=data,
            event_count=len(x),
            tiles=tiles,
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            window=window,
            decay=decay if mode == "time_surface" else None,
            event_x=x.astype(np.uint16),
            event_y=y.astype(np.uint16),
            _tile_size=self.tile_size,
            _min_events=self.min_events,
            name=self.name,
        )

    def build_window(self, window, mode: typing.Optional[str] = None, decay: typing.Optional[str] = None) -> Frame:
        """Builds the frame of an EventWindow."""
        return self.build(
            window.polarity_events,
            window.start_t,
            window.end_t,
            camera=window.camera,
            window=window,
            mode=mode,
            decay=decay,
        )

    def _frame(self, x, y, p, t, start_t: float, duration: float, mode: str, decay: str = "linear") -> np.ndarray:
        size = self.height * self.width * 2
        index = (y * self.width + x) * 2 + p
        if mode == "time_surface":
            # value of every event, then the largest per pixel = the latest event (also for
            # unsorted events: both decays grow with t). Only the events are computed, not the
            # whole frame; pixels without events stay 0.
            relative = t - start_t
            if decay == "linear":
                values = (relative / duration).astype(np.float32)
            else:
                # exp(-(end_t - t_last) / tau): 1 for an event at the end of the window
                values = np.exp((relative - duration) / self.tau_us).astype(np.float32)
            data = np.zeros(size, dtype=np.float32)
            np.maximum.at(data, index, values)
        elif mode == "binary":
            data = np.zeros(size, dtype=np.float32)
            data[index] = 1.0
        else:
            data = np.bincount(index, minlength=size).astype(np.float32)
        return data.reshape(self.height, self.width, 2)

    def _tiles(self, data: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[Tile]:
        if not self.tile_size:
            return []
        counts = np.bincount(
            (y // self.tile_size) * self.tile_cols + (x // self.tile_size), minlength=self.tile_rows * self.tile_cols
        )
        return split_into_tiles(
            data, self.tile_size, counts.reshape(self.tile_rows, self.tile_cols), self.min_events
        )

    # Interface of the original FrameEngine (frame_data: pandas DataFrame or array with x, y, p, t)

    @staticmethod
    def create_linear_time_surface(frame_data, frame_width: int, frame_height: int, frame_start: int, frame_end: int):
        """Linear time surface, shape (height, width, 2), float32 (same as the original FrameEngine)."""
        engine = FrameEngine(frame_width, frame_height)
        if len(_columns(frame_data)[0]) == 0:
            if frame_end <= frame_start:
                raise ValueError("frame_end must be greater than frame_start")
            return np.zeros((engine.height, engine.width, 2), dtype=np.float32)
        return engine.build(frame_data, frame_start, frame_end).data

    def generate_frame(
        self,
        frame_data,
        threshold=0,
        select_insects_points=True,
        use_time_surface=True,
        tau=100_000,
        frame_end=None,
        activity_threshold=0.04,
        frame_start=None,
    ) -> list[dict[str, typing.Any]]:
        """Tiles as [{"tile_id", "row", "col", "tile_data"}] (same as the original FrameEngine).

        tau and activity_threshold are accepted for compatibility and ignored. Empty tiles are
        skipped; with select_insects_points, tiles with fewer than threshold events are skipped too.
        Without tiles configured, the whole frame is returned as one tile.
        """
        del tau, activity_threshold
        _, _, _, t = _columns(frame_data)
        if len(t) == 0:
            return []
        if frame_end is None:
            frame_end = int(t.max()) + 1
        if frame_start is None:
            frame_start = int(t.min())
        if frame_end <= frame_start:
            raise ValueError("invalid time window (frame_end <= frame_start)")
        minimum = max(1, int(threshold)) if select_insects_points else 1
        engine = FrameEngine(
            self.width,
            self.height,
            tile_size=self.tile_size or max(self.width, self.height),
            mode="time_surface" if use_time_surface else "binary",
            min_events=minimum,
        )
        frame = engine.build(frame_data, frame_start, frame_end)
        return frame.tile_dicts()
