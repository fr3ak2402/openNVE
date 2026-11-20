"""Tests for HDF5 event files (layouts, labels, time units, windows, writing, workflows)."""

import json

import numpy as np
import pytest

import opennve

h5py = pytest.importorskip("h5py")


def _events(count=5000, seed=0, width=64, height=32, span=200_000, start=1_000_000):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.integers(0, width, count).astype(np.uint16),
        "y": rng.integers(0, height, count).astype(np.uint16),
        "p": (rng.random(count) < 0.5).astype(np.uint8),
        "t": np.sort(rng.integers(start, start + span, count)).astype(np.int64),
        "l": rng.integers(0, 3, count).astype(np.int32),
    }


def _check(events, columns, labels=True):
    assert np.array_equal(events["x"], columns["x"]) and np.array_equal(events["y"], columns["y"])
    assert np.array_equal(events["t"], columns["t"]) and np.array_equal(events["on"], columns["p"] > 0)
    if labels:
        assert np.array_equal(events["l"], columns["l"])
    else:
        assert "l" not in events.dtype.names


def test_column_layout_with_labels_and_attributes(tmp_path):
    data = _events()
    path = tmp_path / "columns.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("events")
        for name, values in data.items():
            group.create_dataset(name, data=values)
        group.attrs["width"], group.attrs["height"] = 1280, 720
    with opennve.open_hdf5(path, chunk_events=700) as recording:
        info = recording.info()
        assert info["layout"] == "columns" and info["dataset"] == "/events" and info["labels"]
        assert recording.resolution == (1280, 720) and recording.event_count == 5000
        events, triggers = recording.read_all()
    _check(events, data)
    assert len(triggers) == 0
    assert opennve.read_hdf5(path, labels=False).dtype.names == ("t", "x", "y", "on")


def test_compound_matrix_and_aliases(tmp_path):
    data = _events(seed=1)
    path = tmp_path / "compound.hdf5"
    compound = np.zeros(5000, dtype=[("Timestamp", "<i8"), ("X", "<u2"), ("Y", "<u2"), ("Polarity", "i1"), ("Label", "<i4")])
    compound["Timestamp"], compound["X"], compound["Y"] = data["t"], data["x"], data["y"]
    compound["Polarity"] = np.where(data["p"] > 0, 1, -1)  # -1 / 1 polarity
    compound["Label"] = data["l"]
    with h5py.File(path, "w") as file:
        file.create_dataset("CD/events", data=compound)
    events = opennve.read_hdf5(path)
    _check(events, data)
    with opennve.open_hdf5(path) as recording:  # size from the largest x / y
        assert recording.resolution == (int(data["x"].max()) + 1, int(data["y"].max()) + 1)

    # 2D matrix: column order from the attribute or from columns=[...]
    path = tmp_path / "matrix.h5"
    matrix = np.column_stack([data["t"], data["x"], data["y"], data["p"]]).astype(np.int64)
    with h5py.File(path, "w") as file:
        file.create_dataset("with_attr", data=matrix).attrs["columns"] = ["t", "x", "y", "p"]
        file.create_dataset("plain", data=matrix)
    _check(opennve.read_hdf5(path, dataset="with_attr"), data, labels=False)
    _check(opennve.read_hdf5(path, dataset="plain", columns=["t", "x", "y", "p"]), data, labels=False)
    with pytest.raises(ValueError, match="columns=\\[...\\]"):
        opennve.open_hdf5(path, dataset="plain")
    with pytest.raises(ValueError, match="no dataset or group 'missing'"):
        opennve.open_hdf5(path, dataset="missing")


def test_pandas_files(tmp_path):
    pandas = pytest.importorskip("pandas")
    data = _events(seed=2)
    frame = pandas.DataFrame(data)
    fixed = tmp_path / "fixed.h5"
    frame.to_hdf(fixed, key="events", format="fixed")
    _check(opennve.read_hdf5(fixed), data)
    pytest.importorskip("tables")
    table = tmp_path / "table.h5"
    frame.to_hdf(table, key="data", format="table")
    with opennve.open_hdf5(table, chunk_events=1234) as recording:
        assert recording.info()["layout"] == "pandas_table"
        events = recording.read_all()[0]
    _check(events, data)


def test_time_units_offset_sorting(tmp_path):
    data = _events(seed=3)
    path = tmp_path / "seconds.h5"
    with h5py.File(path, "w") as file:  # float seconds, relative times with t_offset (DSEC)
        group = file.create_group("events")
        group.create_dataset("x", data=data["x"])
        group.create_dataset("y", data=data["y"])
        group.create_dataset("p", data=data["p"])
        group.create_dataset("t", data=(data["t"] - 1_000_000) / 1e6)
        file.create_dataset("t_offset", data=1_000_000)
    with opennve.open_hdf5(path) as recording:
        assert recording.time_unit == "s"
        events = recording.read_all()[0]
    assert np.abs(events["t"].astype(np.int64) - data["t"]).max() <= 1

    path = tmp_path / "unsorted.h5"
    data["t"] = 1_000_000 + np.arange(5000, dtype=np.int64) * 40  # unique time stamps: one sorted order
    order = np.random.default_rng(0).permutation(5000)
    with h5py.File(path, "w") as file:
        for name, values in data.items():
            file.create_dataset(f"events/{name}", data=values[order])
    with pytest.raises(ValueError, match="sort=True"):
        opennve.read_hdf5(path)
    _check(opennve.read_hdf5(path, sort=True), data)


def test_windows_output_labels_and_filter(tmp_path):
    data = _events(seed=4, count=20_000, span=400_000)
    path = tmp_path / "labeled.h5"
    with h5py.File(path, "w") as file:
        for name, values in data.items():
            file.create_dataset(f"events/{name}", data=values)
    config = {"name": "left", "model": "SE1-S4", "output": {"fields": ["x", "y", "p", "t", "l"]}, "window": {"duration_ms": 100}}
    with opennve.open_hdf5(path, config) as recording:
        windows = list(recording.windows())
    assert [w.start_t for w in windows] == [1_000_000, 1_100_000, 1_200_000, 1_300_000]
    assert windows[0].camera == "left" and sum(len(w) for w in windows) == 20_000
    out = windows[1].events
    assert out.dtype.names == ("x", "y", "p", "t", "l")
    mask = (data["t"] >= 1_100_000) & (data["t"] < 1_200_000)
    assert np.array_equal(out["l"], data["l"][mask])
    frame = windows[0].frame({"tiles": {"size": 16}})
    assert frame.data.shape == (32, 64, 2)
    # a live camera / .raw file without labels gives l = 0
    plain = np.zeros(3, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
    assert opennve.OutputConfig(fields=["x", "l"]).format(plain)["l"].tolist() == [0, 0, 0]
    # filter (polarity) keeps the labels of the remaining events
    with opennve.open_hdf5(path, filter={"polarity": "on"}) as recording:
        events = recording.read_all()[0]
    on = data["p"] > 0
    assert np.array_equal(events["l"], data["l"][on])


def test_write_hdf5_and_open_recording(tmp_path):
    data = _events(seed=5)
    events = np.zeros(5000, dtype=[("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<u8"), ("l", "<i4")])
    for name in events.dtype.names:
        events[name] = data[name]
    path = tmp_path / "out.h5"
    opennve.write_hdf5(path, events[:2000], width=64, height=32)
    with opennve.Hdf5Writer(path) as writer:  # append
        writer.write(events[2000:])
        assert writer.count == 5000
    with opennve.open_recording(path) as recording:
        assert isinstance(recording, opennve.Hdf5Reader) and recording.resolution == (64, 32)
        _check(recording.read_all()[0], data)
    # .h suffix and files without suffix are detected by the HDF5 signature
    other = tmp_path / "events.h"
    other.write_bytes(path.read_bytes())
    assert isinstance(opennve.open_recording(other), opennve.Hdf5Reader)
    assert opennve.recordings.is_hdf5(other) and not opennve.recordings.is_hdf5(tmp_path / "missing")


def test_workflow_with_hdf5_source_and_sink(tmp_path):
    data = _events(seed=6, count=10_000, span=1_000_000)
    source = tmp_path / "in.h5"
    with h5py.File(source, "w") as file:
        for name, values in data.items():
            file.create_dataset(f"events/{name}", data=values)
        file["events"].attrs["width"], file["events"].attrs["height"] = 64, 32

    def keep_all_events(item):  # all events of the block, with the labels of the file
        item.events = item.block.events
        return item

    workflow = opennve.Workflow.from_steps(
        "h5",
        [
            {"type": "recording", "path": str(source), "camera": "in", "block_ms": 250},
            {"type": "frames", "frame": {"name": "f", "tiles": {"size": 16}}, "threads": 2},
            keep_all_events,
            {"type": "write", "format": "hdf5", "path": str(tmp_path / "{workflow}_{camera}.h5")},
            {"type": "collect"},
        ],
        cameras=[{"name": "in", "model": "SE1-S4", "output": {"fields": ["x", "y", "p", "t", "l"]}}],
    )
    workflow.run()
    assert len(workflow.results) == 4
    written = opennve.read_hdf5(tmp_path / "h5_in.h5")
    _check(written, data)
    with h5py.File(tmp_path / "h5_in.h5") as file:
        assert file["events"].attrs["width"] == 64
    assert json.dumps(workflow.results[0].summary())


def test_cli_recording(tmp_path, capsys):
    from opennve.cli import main

    events = np.zeros(100, dtype=[("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<u8"), ("l", "<i4")])
    events["t"] = np.arange(100) * 1000
    events["l"][:30] = 2
    opennve.write_hdf5(tmp_path / "a.h5", events, width=10, height=10)
    assert main(["recording", str(tmp_path / "a.h5"), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["events"] == 100 and info["label_counts"] == {"0": 70, "2": 30} and info["layout"] == "columns"
