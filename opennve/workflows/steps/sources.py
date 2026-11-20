"""Source steps: camera (live cameras) and raw (recordings)."""

from __future__ import annotations

import glob
import os
import pathlib
import time

from ..item import WorkItem
from .base import Step, _positive_number, register_step



class _Source(Step):
    kind = "source"
    MAX_THREADS = 1

    def _limits(self):
        return self.params.get("max_blocks"), self.params.get("duration_s")


@register_step("camera")
class CameraSource(_Source):
    """Live cameras: one item per block of block_ms (default: the camera "window" duration).

    cameras: camera names of the camera configuration (null: all cameras of the file).
    max_blocks / duration_s: stop after this many blocks / seconds (null: until stopped).
    install_driver: install missing USB drivers first (Windows, SE1-S4).
    """

    PARAMETERS = {"cameras": None, "block_ms": None, "max_blocks": None, "duration_s": None, "install_driver": False}

    def validate(self):
        _positive_number(self, "block_ms")
        _positive_number(self, "max_blocks", integer=True)
        _positive_number(self, "duration_s")
        cameras = self.params["cameras"]
        if isinstance(cameras, str):
            self.params["cameras"] = [cameras]
        elif cameras is not None and not (isinstance(cameras, list) and all(isinstance(c, str) for c in cameras)):
            raise ValueError("step camera: cameras must be a camera name, a list of names or null")

    def items(self, stop):
        from ...cameras.camera import open_cameras

        configs = self.context.camera_configs()
        names = self.params["cameras"]
        if names is not None:
            by_name = {config.name: config for config in configs}
            missing = [name for name in names if name not in by_name]
            if missing:
                raise ValueError(f"step camera: unknown camera(s) {', '.join(missing)} (available: {', '.join(by_name)})")
            configs = [by_name[name] for name in names]
        max_blocks, duration_s = self._limits()
        started = time.monotonic()
        index = 0
        cameras = open_cameras(configs, install_driver=bool(self.params["install_driver"]))
        try:
            while not stop.is_set():
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    return
                for camera in cameras.values():
                    for window in camera.read_windows(self.params["block_ms"]):
                        if stop.is_set():
                            return
                        yield WorkItem(index=index, camera=window.camera, block=window)
                        index += 1
                        if max_blocks is not None and index >= max_blocks:
                            return
        finally:
            for camera in cameras.values():
                camera.close()


@register_step("recording")
class RecordingSource(_Source):
    """Recordings (.raw or HDF5 .h5 / .hdf5, chosen by the file type): one item per block of block_ms.

    path: file, glob pattern ("recordings/*.h5") or list (relative to the workflow file).
    camera: camera name in the camera configuration whose filter, output format and window are
        used (null: defaults; the camera name is then the file name).
    speed: 1.0 = real time, null = as fast as possible.
    chunk_kb: .raw: bytes read at once (null: 1024 KB, or 16 KB with a speed, for smooth playback).
    dataset, time_unit, sort: HDF5 files (see opennve.open_hdf5).
    """

    PARAMETERS = {
        "path": None,
        "camera": None,
        "block_ms": None,
        "speed": None,
        "max_blocks": None,
        "chunk_kb": None,
        "dataset": None,
        "time_unit": "auto",
        "sort": False,
    }

    def validate(self):
        if not self.params["path"]:
            raise ValueError(f"step {self.TYPE}: path is required")
        _positive_number(self, "block_ms")
        _positive_number(self, "speed")
        _positive_number(self, "max_blocks", integer=True)
        _positive_number(self, "chunk_kb", integer=True)

    def paths(self) -> list[pathlib.Path]:
        values = self.params["path"]
        values = [values] if isinstance(values, (str, os.PathLike)) else values
        paths: list[pathlib.Path] = []
        for value in values:
            pattern = str(self.context.path(value))
            matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
            paths.extend(pathlib.Path(match) for match in matches)
        if not paths:
            raise ValueError(f"step {self.TYPE}: no file matches {values}")
        return paths

    def _open(self, path: pathlib.Path, config):
        from ...recordings.hdf5 import HDF5_SUFFIXES, is_hdf5, open_hdf5
        from ...recordings.raw import open_raw

        speed = self.params["speed"]
        if path.suffix.lower() in HDF5_SUFFIXES or (path.suffix.lower() != ".raw" and is_hdf5(path)):
            return open_hdf5(
                path,
                config,
                speed=speed,
                dataset=self.params["dataset"],
                time_unit=self.params["time_unit"],
                sort=bool(self.params["sort"]),
                chunk_events=20_000 if speed else 1_000_000,
            )
        chunk_kb = self.params["chunk_kb"] or (16 if speed else 1024)
        return open_raw(path, config, speed=speed, chunk_bytes=chunk_kb * 1024)

    def items(self, stop):
        config = None
        if self.params["camera"] is not None:
            configs = {c.name: c for c in self.context.camera_configs()}
            if self.params["camera"] not in configs:
                raise ValueError(f"step {self.TYPE}: unknown camera {self.params['camera']!r}")
            config = configs[self.params["camera"]]
        block_ms = self.params["block_ms"]
        index = 0
        for path in self.paths():
            with self._open(path, config) as recording:
                if block_ms is None and recording.window_config is None:
                    block_ms = 40
                while not stop.is_set():
                    windows = recording.read_windows(block_ms)
                    if windows is None:
                        break
                    for window in windows:
                        if stop.is_set():
                            return
                        yield WorkItem(index=index, camera=window.camera, block=window)
                        index += 1
                        if self.params["max_blocks"] is not None and index >= self.params["max_blocks"]:
                            return
            if stop.is_set():
                return


@register_step("raw")
class RawSource(RecordingSource):
    """.raw recordings (the same as "recording")."""


@register_step("hdf5")
class Hdf5Source(RecordingSource):
    """HDF5 recordings (the same as "recording")."""
