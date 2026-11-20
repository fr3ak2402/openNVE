"""Tests without hardware (neuromorphic_drivers is mocked where a camera is needed)."""

import json
import types

import neuromorphic_drivers as nd
import numpy as np
import pytest

import opennve
from opennve.cameras import camera as camera_module
from opennve.cameras import discovery as discovery_module

EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


# ------------------------------------------------------------------ models


def test_version_matches_pyproject():
    import pathlib
    import tomllib

    pyproject = tomllib.loads((pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert opennve.__version__ == pyproject["project"]["version"]


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
    assert opennve.CameraConfig(model="SE1-S4").output.fields == ["x", "y", "p", "t"]
    config = opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"fields": "t, x, y"}})
    assert config.output.fields == ["t", "x", "y"]
    assert opennve.CameraConfig.from_dict(config.to_dict()) == config
    for fields, message in (
        (["c", "x"], "once per packet"),
        (["x", "z"], "unknown output field"),
        (["x", "x"], "duplicate"),
        ([], "non-empty"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"fields": fields}})
    with pytest.raises(ValueError, match="unknown output setting"):
        opennve.CameraConfig.from_dict({"model": "SE1-S4", "output": {"format": "csv"}})


def test_output_format():
    events = _events([(1003, 412, 233, True), (1010, 5, 6, False)])
    formatted = opennve.OutputConfig().format(events)
    assert formatted.dtype.names == ("x", "y", "p", "t")
    assert opennve.to_rows(formatted) == [(412, 233, 1, 1003), (5, 6, 0, 1010)]
    assert opennve.to_csv(formatted, header=True, camera="left") == (
        "# camera: left\nx,y,p,t\n412,233,1,1003\n5,6,0,1010\n"
    )
    reordered = opennve.OutputConfig(fields=["t", "p"]).format(events)
    assert opennve.to_rows(reordered) == [(1003, 1), (1010, 0)]


# ------------------------------------------------------------------ windows


def test_window_config():
    config = opennve.CameraConfig.from_dict({"model": "SE1-S4", "window": {"duration_ms": 2.5}})
    assert config.window.duration_us == 2500
    assert opennve.CameraConfig.from_dict(config.to_dict()) == config
    assert opennve.CameraConfig(model="SE1-S4").window.duration_ms == 40
    assert opennve.CameraConfig.from_dict({"model": "SE1-S4"}).window.duration_us == 40_000
    disabled = opennve.CameraConfig.from_dict({"model": "SE1-S4", "window": None})
    assert disabled.window is None and disabled.to_dict()["window"] is None
    assert opennve.CameraConfig.from_dict(disabled.to_dict()).window is None
    for window, message in (
        ({"duration_ms": 0}, "must be in"),
        ({"duration_ms": "10"}, "must be a number"),
        ({"duration": 10}, "unknown window setting"),
        (10, "must be an object"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.CameraConfig.from_dict({"model": "SE1-S4", "window": window})


def test_window_accumulator_boundaries():
    accumulator = opennve.WindowAccumulator(1000, "left", 100, 100)
    windows = accumulator.push(_events([(100, 1, 1, True), (900, 2, 2, False), (1100, 3, 3, True), (2500, 4, 4, True)]))
    assert [(w.start_t, w.end_t, len(w)) for w in windows] == [(0, 1000, 2), (1000, 2000, 1)]
    assert windows[0].camera == "left" and windows[0].duration_ms == 1.0
    assert windows[0].rows() == [(1, 1, 1, 100), (2, 2, 0, 900)]
    # Camera time passes the end of the open window without new events
    windows = accumulator.push(_events([]), current_t=4200)
    assert [(w.start_t, len(w)) for w in windows] == [(2000, 1), (3000, 0)]
    windows = accumulator.push(_events([(4300, 5, 5, True)]), current_t=5000)
    assert [(w.start_t, len(w)) for w in windows] == [(4000, 1)]


def test_window_accumulator_split_packets_and_gaps():
    accumulator = opennve.WindowAccumulator(10, "left", 10, 10)
    windows = []
    for t in range(0, 100, 3):  # one event per packet
        windows += accumulator.push(_events([(t, 0, 0, True)]), current_t=t)
    assert [w.start_t for w in windows] == list(range(0, 90, 10))
    assert sum(len(w) for w in windows) == len(range(0, 90, 3))
    # Very long gap: no millions of empty windows
    windows = accumulator.push(_events([(10_000_000, 0, 0, True)]))
    assert len(windows) == 1 and windows[0].start_t == 90
    windows = accumulator.push(_events([]), current_t=10_000_010)
    assert [w.start_t for w in windows] == [10_000_000]


def test_window_frames():
    window = opennve.EventWindow(
        camera="left",
        start_t=0,
        end_t=10,
        width=4,
        height=3,
        polarity_events=_events([(0, 1, 2, True), (1, 1, 2, True), (2, 1, 2, False), (3, 3, 0, False)]),
    )
    count = window.to_frame("count")
    assert count.shape == (3, 4) and count[2, 1] == 3 and count[0, 3] == 1 and count.sum() == 4
    polarity = window.to_frame("polarity")
    assert polarity[2, 1] == 1 and polarity[0, 3] == -1
    assert window.to_frame("binary")[2, 1] == 255
    color = window.to_frame("color")
    assert color.shape == (3, 4, 3) and tuple(color[0, 3]) == (0, 0, 255)
    with pytest.raises(ValueError, match="unknown frame mode"):
        window.to_frame("gray")


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
                types.SimpleNamespace(ring=types.SimpleNamespace(backlog=0, current_t=25)),
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
        assert packets[0].camera == "left"
        assert packets[0].rows() == [(5, 5, 1, 0), (6, 5, 1, 10)]
        camera.update_output({"fields": ["t", "x"]})
        assert camera.config.output.fields == ["t", "x"]
        info = camera.info()
        assert info["resolution"] == {"width": 100, "height": 100}
        assert "diff_on" in info["biases_bounds"]
    assert not camera.is_open


def test_camera_windows(fake_open):
    config = {
        "name": "left",
        "model": "SE1-S4",
        "filter": {"background_activity": {"enabled": True, "time_us": 1000}},
        "window": {"duration_ms": 0.01},
    }
    with opennve.open_camera(config) as camera:
        windows = list(camera.windows())
        # events at 0 and 10 µs (the isolated event at 20 µs is filtered), camera time 25 µs
        assert [(w.start_t, w.end_t, len(w)) for w in windows] == [(0, 10, 1), (10, 20, 1)]
        assert windows[0].to_frame("count").shape == (100, 100)
        camera.update_window(None)
        with pytest.raises(ValueError, match="no window duration"):
            camera.read_windows()


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
    from opennve.cli import main

    assert main(["template", "SE1-S4", "--serial", "05110634", "--name", "left"]) == 0
    output = capsys.readouterr().out
    config = opennve.CameraConfig.from_json(output)
    assert config.name == "left"

    path = tmp_path / "cameras.json"
    path.write_text(output)
    assert main(["check", str(path)]) == 0
    path.write_text('{"model": "SE1-S4", "settings": {"clock": "fast"}}')
    assert main(["check", str(path)]) == 1


# ------------------------------------------------------------------ driver


class FakeWindowsDriver:
    """Simulates neuromorphic_drivers.windows_driver on Windows."""

    def __init__(self, devices):
        self.devices = devices  # list of (vendor_id, product_id, service)
        self.installs = []

    def list_windows_devices(self):
        from neuromorphic_drivers import windows_driver

        result = []
        for vendor_id, product_id, service in self.devices:
            supported = next(d for d in windows_driver.SUPPORTED_DEVICES if d.matches(vendor_id, product_id))
            result.append(
                windows_driver.WindowsUsbDevice(
                    instance_id=f"USB\\VID_{vendor_id:04X}&PID_{product_id:04X}\\1",
                    name="camera",
                    status="OK",
                    service=service,
                    vendor_id=vendor_id,
                    product_id=product_id,
                    supported_device=supported,
                )
            )
        return result

    def install_windows_driver(self, usb_ids=None, force=False):
        self.installs.append((usb_ids, force))
        self.devices = [
            (v, p, "WinUSB" if (usb_ids is None or (v, p) in usb_ids) and (s is None or force) else s)
            for v, p, s in self.devices
        ]
        return True


@pytest.fixture
def windows(monkeypatch):
    from opennve.cameras import driver as driver_module

    fake = FakeWindowsDriver([(0x04B4, 0x00C4, None), (0x04B4, 0x00F5, "CYUSB3")])
    monkeypatch.setattr(driver_module.sys, "platform", "win32")
    monkeypatch.setattr(driver_module, "_windows_driver", lambda: fake)
    return fake


def test_driver_status_windows(windows):
    statuses = opennve.driver_status()
    assert [(s.usb_id, s.driver, s.ready) for s in statuses] == [("04B4:00C4", None, False), ("04B4:00F5", "CYUSB3", False)]
    assert "install_driver()" in statuses[0].message and "force=True" in statuses[1].message
    assert [s.usb_id for s in opennve.driver_status("SE1-S4")] == ["04B4:00C4"]
    with pytest.raises(ValueError, match="needs no driver"):
        opennve.driver_status("TRITON")


def test_ensure_driver_installs_only_when_needed(windows):
    assert opennve.ensure_driver("SE1-S4") is True
    assert windows.installs == [([(0x04B4, 0x00C4)], False)]
    assert opennve.driver_status("SE1-S4")[0].driver == "WinUSB"
    assert opennve.ensure_driver("SE1-S4") is True
    assert len(windows.installs) == 1  # already installed: nothing to do
    # EVK4 uses a vendor driver: not replaced without force
    assert opennve.ensure_driver("EVK4") is False
    assert len(windows.installs) == 1
    assert opennve.install_driver("EVK4", force=True) is True
    assert opennve.driver_status("EVK4")[0].ready


def test_open_camera_installs_driver(windows, fake_open):
    with opennve.open_camera({"model": "SE1-S4", "serial": "05110634"}, install_driver=True) as camera:
        assert camera.config.serial == "05110634"
    assert windows.installs == [([(0x04B4, 0x00C4)], False)]


def test_open_camera_driver_not_ready(windows, fake_open, monkeypatch):
    monkeypatch.setattr(windows, "install_windows_driver", lambda usb_ids=None, force=False: False)
    with pytest.raises(RuntimeError, match="driver for SE1-S4 is not ready"):
        opennve.open_camera({"model": "SE1-S4"}, install_driver=True)
    assert fake_open == []  # the camera was not opened


def test_driver_status_linux(monkeypatch):
    from opennve.cameras import driver as driver_module

    listed = [
        types.SimpleNamespace(name=nd.Name.SENSING_SE1_S4, error="permission denied"),
        types.SimpleNamespace(name=nd.Name.PROPHESEE_EVK4, error=None),
    ]
    monkeypatch.setattr(driver_module.sys, "platform", "linux")
    monkeypatch.setattr(driver_module.nd, "list_devices", lambda: listed)
    statuses = opennve.driver_status()
    assert [s.ready for s in statuses] == [False, True]
    assert "udev" in statuses[0].message
    assert opennve.ensure_driver("EVK4") is True
    assert opennve.ensure_driver("SE1-S4") is False


def test_cli_driver(windows, capsys):
    from opennve.cli import main

    assert main(["driver"]) == 1
    assert "not ready" in capsys.readouterr().out
    assert main(["driver", "--install", "--model", "SE1-S4"]) == 0
    assert main(["driver", "--model", "SE1-S4"]) == 0
    assert "ready" in capsys.readouterr().out
