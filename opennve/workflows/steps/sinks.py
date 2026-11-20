"""Sink steps: write (CSV, JSON lines) and collect (memory)."""

from __future__ import annotations

import json
import pathlib
import threading
import typing

import numpy as np

from ..item import WorkItem
from .base import Step, StepContext, _positive_number, register_step



def _format_path(context: StepContext, pattern: str, item: WorkItem) -> pathlib.Path:
    started = context.started
    values = {
        "workflow": context.workflow,
        "camera": item.camera,
        "date": started.strftime("%Y%m%d"),
        "time": started.strftime("%H%M"),
        "datetime": started.strftime("%Y%m%d_%H%M%S"),
        "index": item.index,
    }
    try:
        return context.path(pattern.format(**values))
    except KeyError as error:
        raise ValueError(f"unknown placeholder {error} in path {pattern!r} (supported: {', '.join(values)})") from None


@register_step("write")
class WriteStep(Step):
    """Writes the results to files.

    format: "csv" (item.events, or the block events if no step set events; one row per event),
        "hdf5" (the same events as an HDF5 file with one dataset per column, see opennve.write_hdf5)
        or "jsonl" (one JSON line per item: block, windows and selected predictions with pixels).
    path: file name pattern with {workflow}, {camera}, {date}, {time}, {datetime}, {index},
        for instance "output/{workflow}_{camera}_{date}_{time}.csv". Files are appended.
    camera_column: add the camera name as first CSV column.
    """

    kind = "sink"
    MAX_THREADS = 1
    PARAMETERS = {"format": "csv", "path": "output/{workflow}_{date}_{time}.csv", "camera_column": False}

    def validate(self):
        if self.params["format"] not in ("csv", "jsonl", "hdf5"):
            raise ValueError("step write: format must be csv, hdf5 or jsonl")
        if not isinstance(self.params["path"], str):
            raise ValueError("step write: path must be a file name pattern")
        self._files: dict[pathlib.Path, typing.Any] = {}
        self._lock = threading.Lock()
        self.written: dict[str, int] = {}

    def _file(self, path: pathlib.Path):
        if path not in self._files:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists() or path.stat().st_size == 0
            self._files[path] = (open(path, "a", encoding="utf-8", newline=""), new)
        return self._files[path]

    def _hdf5(self, path: pathlib.Path, item: WorkItem):
        from ...recordings.hdf5 import Hdf5Writer

        if path not in self._files:
            self._files[path] = (
                Hdf5Writer(path, width=item.block.width, height=item.block.height, attrs={"camera": item.camera}),
                False,
            )
        return self._files[path][0]

    def process(self, item):
        path = _format_path(self.context, self.params["path"], item)
        with self._lock:
            if self.params["format"] == "hdf5":
                events = item.events if item.events is not None else item.block.events
                writer = self._hdf5(path, item)
                if len(events):
                    writer.write(events)
                writer.flush()
                self.written[str(path)] = self.written.get(str(path), 0) + len(events)
                return item
            handle, new = self._file(path)
            if self.params["format"] == "jsonl":
                handle.write(json.dumps(item.summary()) + "\n")
            else:
                events = item.events if item.events is not None else item.block.events
                names = list(events.dtype.names)
                if new:
                    header = (["camera"] if self.params["camera_column"] else []) + names
                    handle.write(",".join(header) + "\n")
                    self._files[path] = (handle, False)
                if len(events):
                    columns = [events[name].astype(np.int64) if events[name].dtype != np.bool_ else events[name].astype(np.int8) for name in names]
                    rows = np.column_stack(columns)
                    prefix = f"{item.camera}," if self.params["camera_column"] else ""
                    handle.write("".join(prefix + ",".join(map(str, row)) + "\n" for row in rows.tolist()))
            handle.flush()
            self.written[str(path)] = self.written.get(str(path), 0) + (1 if self.params["format"] == "jsonl" else len(item.events if item.events is not None else item.block))
        return item

    def close(self):
        with self._lock:
            for handle, _ in self._files.values():
                handle.close()
            self._files.clear()


@register_step("collect")
class CollectStep(Step):
    """Keeps the items in memory: workflow.results (for use from code and tests)."""

    kind = "sink"
    MAX_THREADS = 1
    PARAMETERS = {"max_items": None}

    def validate(self):
        _positive_number(self, "max_items", integer=True)
        self.items_collected: list[WorkItem] = []
        self._lock = threading.Lock()

    def process(self, item):
        with self._lock:
            self.items_collected.append(item)
            if self.params["max_items"] is not None and len(self.items_collected) > self.params["max_items"]:
                self.items_collected.pop(0)
        return item
