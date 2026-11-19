"""Tests for the frame engine (frames and tiles)."""

import pathlib

import numpy as np
import pytest

import opennve

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
FRAMES_JSON = str(EXAMPLES / "frames.json")
CAMERAS_JSON = str(EXAMPLES / "cameras.json")

EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


def _events(points):
    events = np.zeros(len(points), dtype=EVENT_DTYPE)
    for index, point in enumerate(points):
        events[index] = point
    return events


def test_frame_config():
    config = opennve.FrameConfig.from_dict(
        {"name": "tiles80", "width": 1280, "height": 720, "mode": "binary", "tiles": {"size": 80}}
    )
    assert config.name == "tiles80" and config.tiles.size == 80 and config.tiles.min_events == 1
    assert opennve.FrameConfig.from_dict(config.to_dict()) == config
    assert opennve.FrameConfig.from_json(config.to_json()) == config
    assert config.copy(name="small", width=640).width == 640 and config.width == 1280
    assert opennve.FrameConfig.from_dict({"tiles": 64}).tiles.size == 64
    assert opennve.as_frame_config(None) == opennve.FrameConfig()  # sensor size, time surface, no tiles
    for frame, message in (
        ({"mode": "gray"}, "frame.mode"),
        ({"width": 0}, "frame.width"),
        ({"tiles": {"size": -1}}, "frame.tiles.size"),
        ({"tiles": {"size": 80, "min_events": -1}}, "min_events"),
        ({"tiles": {"width": 80}}, "unknown frame.tiles"),
        ({"resolution": "1280x720"}, "unknown frame setting"),
        ({"name": ""}, "frame.name"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.FrameConfig.from_dict(frame)


def test_frame_config_is_not_part_of_the_camera_config():
    assert "frame" not in opennve.CameraConfig(model="SE1-S4").to_dict()
    with pytest.raises(ValueError, match="frame configuration file"):
        opennve.CameraConfig.from_dict({"model": "SE1-S4", "frame": {"tiles": 80}})


def test_frame_config_files(tmp_path):
    configs = opennve.load_frame_configs(FRAMES_JSON)
    assert [c.name for c in configs] == ["full", "tiles80", "exponential", "binary"]
    assert opennve.load_frame_config(FRAMES_JSON, "exponential").time_surface.decay == "exponential"
    with pytest.raises(ValueError, match="no frame configuration named 'x'"):
        opennve.load_frame_config(FRAMES_JSON, "x")
    with pytest.raises(ValueError, match="specify a name"):
        opennve.load_frame_config(FRAMES_JSON)
    path = tmp_path / "frames.json"
    opennve.save_frame_configs(path, configs)
    assert opennve.load_frame_configs(path) == configs
    opennve.save_frame_configs(path, configs[1])  # one object
    assert opennve.load_frame_config(path) == configs[1]
    # names are optional (frame1, frame2, ...) but must be unique
    path.write_text('[{"tiles": 80}, {"mode": "count"}]')
    assert [c.name for c in opennve.load_frame_configs(path)] == ["frame1", "frame2"]
    path.write_text('{"frames": [{"name": "a"}, {"name": "a"}]}')
    with pytest.raises(ValueError, match="duplicate frame configuration name"):
        opennve.load_frame_configs(path)
    path.write_text('{"frames": [{"name": "a", "mode": "gray"}]}')
    with pytest.raises(ValueError, match="frame configuration 0: frame.mode"):
        opennve.load_frame_configs(path)
    # one file with cameras and frames
    path.write_text('{"cameras": [{"name": "left", "model": "SE1-S4"}], "frames": [{"name": "full"}]}')
    assert opennve.load_configs(path)[0].name == "left" and opennve.load_frame_configs(path)[0].name == "full"
    # dict {name: configuration}
    assert [c.name for c in opennve.as_frame_configs({"a": {"tiles": 80}, "b": opennve.FrameConfig()})] == ["a", "b"]


def test_cli_frame_config(tmp_path, capsys):
    from opennve.__main__ import main

    assert main(["frame-template", "--name", "t", "--tiles", "80", "--width", "1280", "--height", "720"]) == 0
    path = tmp_path / "frames.json"
    path.write_text(capsys.readouterr().out)
    assert opennve.load_frame_config(path, "t").tiles.size == 80
    assert main(["check", CAMERAS_JSON, str(path)]) == 0
    output = capsys.readouterr().out
    assert "camera left" in output and "frame t: 1280 x 720, time_surface (linear), tiles 80 px" in output


def test_time_surface():
    engine = opennve.FrameEngine(4, 3)
    events = _events([(1000, 1, 2, True), (1500, 1, 2, True), (1250, 1, 2, False), (1000, 3, 0, False), (2000, 0, 0, True)])
    frame = engine.build(events, 1000, 2000, camera="left")
    assert frame.data.shape == (3, 4, 2) and frame.data.dtype == np.float32
    assert frame.data[2, 1, 1] == pytest.approx(0.5)  # latest ON event at 1500
    assert frame.data[2, 1, 0] == pytest.approx(0.25)  # OFF event at 1250
    assert frame.data[0, 3, 0] == 0.0  # event at start_t: value 0
    assert frame.event_count == 4  # the event at end_t is outside the window
    assert frame.tiles == [] and (frame.width, frame.height) == (4, 3)


def test_binary_and_count():
    events = _events([(0, 1, 1, True), (1, 1, 1, True), (2, 0, 0, False)])
    binary = opennve.FrameEngine(2, 2, mode="binary").build(events, 0, 10).data
    count = opennve.FrameEngine(2, 2, mode="count").build(events, 0, 10).data
    assert binary[1, 1, 1] == 1.0 and binary[0, 0, 0] == 1.0 and binary.sum() == 2
    assert count[1, 1, 1] == 2.0 and count.sum() == 3


def test_tiles():
    engine = opennve.FrameEngine(1280, 720, tile_size=64, min_events=2)
    assert (engine.tile_rows, engine.tile_cols) == (12, 20)  # 720 / 64 = 11.25 -> 12 rows
    events = _events([(0, 0, 0, True), (1, 63, 63, True), (2, 1279, 719, False), (3, 1270, 710, True), (4, 700, 300, True)])
    frame = engine.build(events, 0, 10)
    assert [(t.id, t.row, t.col, t.event_count) for t in frame.tiles] == [(1, 1, 1, 2), (240, 12, 20, 2)]
    edge = frame.tile(240)
    assert (edge.x, edge.y, edge.width, edge.height) == (1216, 704, 64, 16)
    assert edge.data.shape == (16, 64, 2)
    assert frame.tile(5) is None
    assert frame.tile_dicts()[0].keys() == {"tile_id", "row", "col", "tile_data"}
    all_tiles = opennve.FrameEngine(1280, 720, tile_size=80, min_events=0).build(events, 0, 10).tiles
    assert len(all_tiles) == 16 * 9 and all_tiles[-1].id == 144


def test_events_outside_the_frame_are_ignored():
    engine = opennve.FrameEngine(10, 10, tile_size=5)
    frame = engine.build(_events([(0, 9, 9, True), (0, 10, 2, True), (0, 2, 12, False)]), 0, 1)
    assert frame.event_count == 1 and [t.id for t in frame.tiles] == [4]


def test_original_interface():
    # Same calls as the original FrameEngine (frame_data with columns x, y, p, t)
    engine = opennve.FrameEngine(frame_width=160, frame_height=80, frame_tile_size=80)
    frame_data = {
        "x": np.array([5, 6, 100]),
        "y": np.array([5, 5, 40]),
        "p": np.array([1, 0, 1]),
        "t": np.array([100, 150, 199]),
    }
    tiles = engine.generate_frame(frame_data, threshold=2)
    assert [(t["tile_id"], t["row"], t["col"]) for t in tiles] == [(1, 1, 1)]  # tile 2 has only 1 event
    assert tiles[0]["tile_data"][5, 5, 1] == pytest.approx(0.0)  # frame_start = min(t) = 100
    tiles = engine.generate_frame(frame_data, threshold=2, select_insects_points=False, use_time_surface=False)
    assert [t["tile_id"] for t in tiles] == [1, 2] and tiles[1]["tile_data"][40, 20, 1] == 1.0
    surface = opennve.FrameEngine.create_linear_time_surface(frame_data, 160, 80, 100, 200)
    assert surface.shape == (80, 160, 2) and surface[40, 100, 1] == pytest.approx(0.99)
    assert engine.generate_frame({"x": [], "y": [], "p": [], "t": []}) == []


def test_window_frame_and_image():
    window = opennve.EventWindow(
        camera="left",
        start_t=0,
        end_t=40_000,
        width=160,
        height=80,
        polarity_events=_events([(20_000, 10, 10, True), (30_000, 90, 50, False)]),
    )
    frame = window.frame(opennve.FrameConfig(name="tiles", tiles=opennve.TileConfig(size=80)))
    assert frame.camera == "left" and frame.name == "tiles" and [t.id for t in frame.tiles] == [1, 2]
    assert frame.events.dtype.names == ("x", "y", "p", "t") and len(frame.events) == 2
    assert window.frame({"mode": "binary"}).data[10, 10, 1] == 1.0
    frames = window.frames([{"name": "binary", "mode": "binary"}, {"name": "tiles", "tiles": 80}])
    assert list(frames) == ["binary", "tiles"] and frames["binary"].name == "binary"
    assert frames["binary"].data[10, 10, 1] == 1.0 and len(frames["tiles"].tiles) == 2
    with pytest.raises(TypeError, match="duration_ms"):
        window.frame(40)
    image = frame.to_image()
    assert image.shape == (80, 160, 3) and image[10, 10, 1] == 127 and image[50, 90, 2] == 191
    assert tuple(image[0, 0]) == (255, 255, 255)  # tile border


def test_camera_frames(monkeypatch):
    from test_opennve import FakeDevice  # simulated camera
    from opennve import camera as camera_module

    monkeypatch.setattr(camera_module.nd, "open", lambda configuration, serial=None, iterator_timeout=None: FakeDevice("05110634", configuration))
    config = {
        "name": "left",
        "model": "SE1-S4",
        "window": {"duration_ms": 0.01},
    }
    frame_configs = [
        opennve.FrameConfig(name="binary", mode="binary", tiles=opennve.TileConfig(size=50)),
        opennve.FrameConfig(name="small", width=50, height=50),
    ]
    with opennve.open_camera(config) as camera:
        frames = list(camera.frames(frame_configs[0]))
        assert [(f.start_t, f.end_t) for f in frames] == [(0, 10), (10, 20)]
        assert frames[0].data.shape == (100, 100, 2) and [t.id for t in frames[0].tiles] == [1]
        assert frames[0].name == "binary" and frames[0].mode == "binary"
    with opennve.open_camera(config) as camera:
        sets = list(camera.frame_sets(frame_configs))
        assert [list(s) for s in sets] == [["binary", "small"]] * 2
        assert sets[0]["small"].data.shape == (50, 50, 2)
        assert sets[0]["binary"].window is sets[0]["small"].window  # same time window


def test_raw_frames(tmp_path):
    from test_raw import _header, _random_events, encode_evt3

    events = _random_events(10, count=2000, span=120_000)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events))
    with opennve.open_raw(path) as raw:
        frames = list(raw.frames({"name": "tiles80", "tiles": {"size": 80}}))
    assert [f.start_t for f in frames] == [1_000_000, 1_040_000, 1_080_000]
    with opennve.open_raw(path) as raw:
        sets = list(raw.frame_sets(opennve.load_frame_configs(FRAMES_JSON)))
    assert len(sets) == 3 and sets[0]["binary"].data.shape == (360, 640, 2)
    assert sum(s["full"].event_count for s in sets) == 2000
    assert sum(f.event_count for f in frames) == 2000
    assert all(len(f.tiles) > 0 and f.data.shape == (720, 1280, 2) for f in frames)


def test_split_later():
    # Frame without tiles first, process it, then split it
    engine = opennve.FrameEngine(160, 100)  # no tiles
    events = _events([(10, 5, 5, True), (20, 6, 5, True), (30, 150, 90, False)])
    frame = engine.build(events, 0, 40)
    assert frame.tiles == []
    frame.data *= 2.0  # process the frame in place
    tiles = frame.split(80)  # default min_events: 1 -> empty tiles skipped
    assert [(t.id, t.row, t.col, t.event_count) for t in tiles] == [(1, 1, 1, 2), (4, 2, 2, 1)]
    assert frame.tiles is tiles and (frame.tile_rows, frame.tile_cols) == (2, 2)
    assert tiles[0].data[5, 6, 1] == pytest.approx(2 * 20 / 40)  # the processed values
    assert (tiles[1].width, tiles[1].height) == (80, 20)  # bottom-right edge tile
    assert len(frame.split(80, min_events=0)) == 4
    assert [t.id for t in frame.split(50, min_events=2)] == [1]
    # split a different array with the event counts of the frame, for instance a one-channel image
    gray = frame.data.max(axis=2)
    tiles = frame.split(80, data=gray)
    assert tiles[0].data.shape == (80, 80) and tiles[0].event_count == 2
    with pytest.raises(ValueError, match="frame size"):
        frame.split(80, data=np.zeros((10, 10)))


def test_split_uses_configured_tiles():
    engine = opennve.FrameEngine(160, 100, tile_size=40, min_events=2)
    frame = engine.build(_events([(1, 1, 1, True), (2, 2, 2, True), (3, 100, 50, True)]), 0, 10)
    assert [t.id for t in frame.tiles] == [1]
    assert [t.id for t in frame.split()] == [1]  # same configuration
    assert [t.id for t in frame.split(min_events=1)] == [1, 7]  # 4 tiles per row: (100, 50) is in row 2, column 3


def test_split_into_tiles_without_events():
    image = np.arange(100 * 160).reshape(100, 160)
    tiles = opennve.split_into_tiles(image, 80)
    assert [t.id for t in tiles] == [1, 2, 3, 4] and tiles[0].event_count is None
    assert tiles[3].data[0, 0] == image[80, 80]
    assert opennve.tile_grid(1280, 720, 80) == (9, 16)
    with pytest.raises(ValueError, match="needs event_counts"):
        opennve.split_into_tiles(image, 80, min_events=1)


def test_camera_frames_then_split(monkeypatch):
    window = opennve.EventWindow(
        camera="left", start_t=0, end_t=40, width=160, height=80,
        polarity_events=_events([(5, 1, 1, True), (6, 100, 10, False)]),
    )
    frame = window.frame()  # default configuration: no tiles
    assert frame.tiles == []
    assert [t.id for t in frame.split(80)] == [1, 2]


def test_exponential_time_surface():
    import math

    events = _events([(1000, 1, 2, True), (1500, 1, 2, True), (1250, 1, 2, False), (1000, 3, 0, False)])
    engine = opennve.FrameEngine(4, 3, decay="exponential", tau_us=500)
    frame = engine.build(events, 1000, 2000)
    assert frame.decay == "exponential" and frame.data.dtype == np.float32
    assert frame.data[2, 1, 1] == pytest.approx(math.exp(-(2000 - 1500) / 500))  # latest ON event
    assert frame.data[2, 1, 0] == pytest.approx(math.exp(-(2000 - 1250) / 500))
    assert frame.data[0, 3, 0] == pytest.approx(math.exp(-2))  # an event at start_t is not 0
    assert frame.data[0, 0, 0] == 0.0  # no event
    linear = engine.build(events, 1000, 2000, decay="linear")
    assert linear.decay == "linear" and linear.data[2, 1, 1] == pytest.approx(0.5)
    assert engine.build(events, 1000, 2000, mode="binary").decay is None


def test_time_surface_config():
    config = opennve.FrameConfig.from_dict(
        {"mode": "time_surface", "time_surface": {"decay": "Exponential", "tau_us": 20000}}
    )
    assert config.time_surface == opennve.TimeSurfaceConfig(decay="exponential", tau_us=20000)
    assert opennve.FrameConfig.from_dict(config.to_dict()) == config
    assert opennve.FrameConfig().time_surface.decay == "linear"
    assert opennve.FrameConfig.from_dict({"time_surface": "exponential"}).time_surface.decay == "exponential"
    for time_surface, message in (
        ({"decay": "log"}, "decay must be one of"),
        ({"tau_us": 0}, "tau_us must be a positive"),
        ({"tau": 1000}, "unknown frame.time_surface"),
    ):
        with pytest.raises(ValueError, match=message):
            opennve.FrameConfig.from_dict({"time_surface": time_surface})
    window = opennve.EventWindow(
        camera="left", start_t=0, end_t=40_000, width=4, height=4,
        polarity_events=_events([(30_000, 1, 1, True)]),
    )
    assert window.frame(config).data[1, 1, 1] == pytest.approx(np.exp(-0.5))
    assert window.frame(config, decay="linear").data[1, 1, 1] == pytest.approx(0.75)


def test_pandas_dataframe_input():
    pd = pytest.importorskip("pandas")
    frame_data = pd.DataFrame({"x": [5, 6, 100], "y": [5, 5, 40], "p": [1, 0, 1], "t": [100, 150, 199]})
    frame = opennve.FrameEngine(160, 80, tile_size=80).build(frame_data, 100, 200)
    assert frame.event_count == 3 and [t.id for t in frame.tiles] == [1, 2]
    tiles = opennve.FrameEngine(frame_width=160, frame_height=80, frame_tile_size=80).generate_frame(frame_data)
    assert [t["tile_id"] for t in tiles] == [1, 2]
