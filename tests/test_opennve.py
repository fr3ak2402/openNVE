"""Tests without hardware (neuromorphic_drivers is mocked where a camera is needed)."""

import json
import types

import neuromorphic_drivers as nd
import numpy as np
import pytest

import opennve
from opennve import camera as camera_module
from opennve import discovery as discovery_module

EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


# ------------------------------------------------------------------ models


def test_model_aliases():
    for name in ("SE1-S4", "se1_s4", "SENSING SE1-S4-USB", "sensing_se1_s4"):
        assert opennve.get_model(name).id == "SE1-S4"
    assert opennve.get_model("evk3_hd").id == "EVK3-HD"
    with pytest.raises(ValueError, match="unknown camera model"):
        opennve.get_model("EVK9")


# ------------------------------------------------------------------ config


def test_partial_settings_keep_defaults():
    config = opennve.CameraConfig.from_dict(
        {"model": "SE1-S4", "settings": {"biases": {"diff_on": 20}, "clock": "external"}}
    )
    assert config.settings.biases.diff_on == 20
    assert config.settings.biases.diff_off == 0
    assert config.settings.clock == nd.sensing_se1_s4.Clock.EXTERNAL
    assert config.settings.enable_external_trigger is True
    assert config.name == "SE1-S4"


def test_json_round_trip():
    config = opennve.CameraConfig.from_dict(
        {
            "name": "left",
            "model": "SE1-S4",
            "serial": "05110634",
            "settings": {
                "biases": {"diff_on": 20, "diff_off": 25},
                "rate_limiter": {"reference_period_us": 200, "maximum_events_per_period": 4000},
            },
            "filter": {"polarity": "on", "background_activity": {"enabled": True, "time_us": 2000}},
        }
    )
    data = json.loads(config.to_json())
    assert "x_mask" not in data["settings"]  # default masks are omitted
    assert opennve.CameraConfig.from_dict(data) == config
    assert config.settings.rate_limiter.maximum_events_per_period == 4000


def test_settings_object_is_accepted():
    settings = nd.sensing_se1_s4.Configuration(biases=nd.sensing_se1_s4.Biases(diff_on=5))
    config = opennve.CameraConfig(model="SE1-S4", settings=settings)
    assert config.settings.biases.diff_on == 5
    with pytest.raises(TypeError):
        opennve.CameraConfig(model="EVK4", settings=settings)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"model": "SE1-S4", "settings": {"biases": {"diff_on": 40000}}}, "diff_on must be in"),
        ({"model": "SE1-S4", "settings": {"biases": {"diff_on": "high"}}}, "must be an integer"),
        ({"model": "SE1-S4", "settings": {"clock": "fast"}}, "clock must be one of"),
        ({"model": "SE1-S4", "settings": {"bias": {}}}, "unknown setting"),
        ({"model": "SE1-S4", "settings": {"enable_output": 1}}, "true or false"),
        ({"model": "SE1-S4", "filter": {"polarity": "up"}}, "polarity"),
        ({"model": "SE1-S4", "filter": {"background_activity": {"time_us": 0}}}, "time_us"),
        ({"model": "SE1-S4", "filter": {"median": {}}}, "unknown filter"),
        ({"model": "SE1-S4", "serial": 5110634}, "serial must be a string"),
        ({"model": "SE1-S4", "colour": "red"}, "unknown camera setting"),
        ({"name": "x"}, "model is required"),
    ],
)
def test_invalid_configs(data, message):
    with pytest.raises(ValueError, match=message):
        opennve.CameraConfig.from_dict(data)


def test_load_and_save_configs(tmp_path):
    left = opennve.CameraConfig(model="SE1-S4", serial="05110634", name="left")
    right = opennve.CameraConfig(model="SE1-S4", serial="05110636", name="right")

    path = tmp_path / "cameras.json"
    opennve.save_configs(path, [left, right])
    assert opennve.load_configs(path) == [left, right]
    assert opennve.load_config(path, "right") == right
    with pytest.raises(ValueError, match="specify a name"):
        opennve.load_config(path)

    single = tmp_path / "single.json"
    opennve.save_configs(single, left)
    assert opennve.load_config(single) == left

    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([left.to_dict(), left.to_dict()]))
    with pytest.raises(ValueError, match="duplicate camera name"):
        opennve.load_configs(as_list)


def test_example_config_is_valid():
    import pathlib

    configs = opennve.load_configs(pathlib.Path(__file__).parent.parent / "examples" / "cameras.json")
    assert [config.name for config in configs] == ["left", "right"]


# ------------------------------------------------------------------ output


def test_output_config():
    assert opennve.CameraConfig(model="SE1-S4").output.fields == ["c", "x", "y", "p", "t"]
    config = opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"fields": "t, x, y"}})
    assert config.output.fields == ["t", "x", "y"]
    assert opennve.CameraConfig.from_dict(config.to_dict()) == config
    for fields, message in ((["x", "z"], "unknown output field"), (["x", "x"], "duplicate"), ([], "non-empty")):
        with pytest.raises(ValueError, match=message):
            opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"fields": fields}})
    with pytest.raises(ValueError, match="unknown output setting"):
        opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"format": "csv"}})


def test_output_format():
    events = _events([(1003, 412, 233, True), (1010, 5, 6, False)])
    output = opennve.OutputConfig(fields=["c", "x", "y", "p", "t"])
    formatted = output.format(events, "left")
    assert formatted.dtype.names == ("c", "x", "y", "p", "t")
    assert opennve.to_rows(formatted) == [("left", 412, 233, 1, 1003), ("left", 5, 6, 0, 1010)]
    assert opennve.to_csv(formatted, header=True) == "c,x,y,p,t\nleft,412,233,1,1003\nleft,5,6,0,1010\n"
    reordered = opennve.OutputConfig(fields=["t", "p"]).format(events, "left")
    assert opennve.to_rows(reordered) == [(1003, 1), (1010, 0)]


# ------------------------------------------------------------------ filters


def _events(points):
    events = np.zeros(len(points), dtype=EVENT_DTYPE)
    for index, (t, x, y, on) in enumerate(points):
        events[index] = (t, x, y, on)
    return events


def test_background_activity_filter():
    noise_filter = opennve.BackgroundActivityFilter(100, 100, time_us=1000, min_neighbors=1)
    events = _events(
        [
            (0, 10, 10, True),  # isolated -> removed
            (100, 50, 50, True),
            (150, 51, 50, True),  # neighbors -> kept
            (5000, 50, 52, True),  # neighbor is too old -> removed
            (6000, 0, 0, False),  # border pixel, isolated -> removed
        ]
    )
    kept = noise_filter(events)
    assert [(int(e["x"]), int(e["y"])) for e in kept] == [(50, 50), (51, 50)]


def test_polarity_filter():
    event_filter = opennve.EventFilter(opennve.FilterConfig(polarity="off"), 10, 10)
    kept = event_filter(_events([(0, 1, 1, True), (1, 2, 2, False)]))
    assert len(kept) == 1 and not kept[0]["on"]


# ------------------------------------------------------------------ discovery


def test_list_cameras(monkeypatch):
    listed = [
        types.SimpleNamespace(
            name=nd.Name.SENSING_SE1_S4,
            connection=types.SimpleNamespace(value="USB 3.0 SuperSpeed (5.0 Gb/s)"),
            address="1:8",
            serial=serial,
            error=None,
        )
        for serial in ("05110636", "05110634")
    ]
    monkeypatch.setattr(discovery_module.nd, "list_devices", lambda: listed)
    configs = [opennve.CameraConfig(model="SE1-S4", serial="05110634", name="left")]
    cameras = opennve.list_cameras(configs)
    assert [(c.serial, c.name, c.available) for c in cameras] == [
        ("05110634", "left", True),
        ("05110636", None, True),
    ]
    data = json.loads(opennve.list_cameras_json())
    assert data[0]["model"] == "SE1-S4" and data[0]["available"] is True


# ------------------------------------------------------------------ camera (mocked device)


class FakeDevice:
    def __init__(self, serial, configuration):
        self._serial = serial
        self.configuration = configuration
        self.updates = []
        self._packets = [
            (types.SimpleNamespace(ring=None), None),
            (
                types.SimpleNamespace(ring=types.SimpleNamespace(backlog=0)),
                types.SimpleNamespace(
                    polarity_events=_events([(0, 5, 5, True), (10, 6, 5, True), (20, 90, 90, False)]),
                    trigger_events=None,
                ),
            ),
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self._packets)

    def serial(self):
        return self._serial

    def properties(self):
        return types.SimpleNamespace(width=100, height=100)

    def biases_bounds(self):
        return nd.sensing_se1_s4.BiasesBounds()

    def update_configuration(self, configuration):
        self.updates.append(configuration)


@pytest.fixture
def fake_open(monkeypatch):
    opened = []

    def open(configuration, serial=None, iterator_timeout=None):
        device = FakeDevice(serial or "05110634", configuration)
        opened.append(device)
        return device

    monkeypatch.setattr(camera_module.nd, "open", open)
    return opened


def test_camera_from_json_config(fake_open):
    config = {
        "name": "left",
        "model": "SE1-S4",
        "serial": None,
        "filter": {"background_activity": {"enabled": True, "time_us": 1000}},
    }
    with opennve.open_camera(config) as camera:
        assert camera.name == "left"
        assert camera.config.serial == "05110634"  # actual serial is filled in
        packets = list(camera.events())
        assert len(packets) == 1
        assert packets[0].raw_event_count == 3
        assert len(packets[0].polarity_events) == 2  # isolated event removed
        assert packets[0].rows() == [("left", 5, 5, 1, 0), ("left", 6, 5, 1, 10)]
        camera.update_output({"fields": ["t", "x"]})
        assert camera.config.output.fields == ["t", "x"]
        info = camera.info()
        assert info["resolution"] == {"width": 100, "height": 100}
        assert "diff_on" in info["biases_bounds"]
    assert not camera.is_open


def test_camera_apply_config(fake_open):
    camera = opennve.SE1S4Driver(serial="05110634")
    camera.update_configuration({"biases": {"diff_on": 30}})
    assert fake_open[0].updates[-1].biases.diff_on == 30
    camera.apply_config(
        {
            "name": "left",
            "model": "SE1-S4",
            "settings": {"biases": {"diff_off": 10}},
            "filter": {"polarity": "on"},
            "output": {"fields": ["x", "y", "t"]},
        }
    )
    assert camera.config.output.fields == ["x", "y", "t"]
    assert fake_open[0].updates[-1].biases.diff_off == 10
    assert camera.config.filter.polarity == "on"
    assert camera.name == "left"
    with pytest.raises(ValueError, match="EVK4"):
        camera.apply_config({"model": "EVK4"})
    camera.close()


def test_open_cameras_starts_secondaries_first(fake_open, monkeypatch):
    monkeypatch.setattr(camera_module.time, "sleep", lambda seconds: None)
    primary = opennve.CameraConfig(
        model="SE1-S4", serial="A", name="primary", settings={"clock": "internal_with_output_enabled"}
    )
    secondary = opennve.CameraConfig(model="SE1-S4", serial="B", name="secondary", settings={"clock": "external"})
    cameras = opennve.open_cameras([primary, secondary])
    assert [device._serial for device in fake_open] == ["B", "A"]
    assert list(cameras) == ["primary", "secondary"]
    for camera in cameras.values():
        camera.close()


def test_cli(capsys, tmp_path):
    from opennve.__main__ import main

    assert main(["template", "SE1-S4", "--serial", "05110634", "--name", "left"]) == 0
    output = capsys.readouterr().out
    config = opennve.CameraConfig.from_json(output)
    assert config.name == "left"

    path = tmp_path / "cameras.json"
    path.write_text(output)
    assert main(["check", str(path)]) == 0
    path.write_text('{"model": "SE1-S4", "settings": {"clock": "fast"}}')
    assert main(["check", str(path)]) == 1
