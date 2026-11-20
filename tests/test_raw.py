"""Tests for reading RAW files (EVT 2.0, 2.1, 3.0)."""

import pathlib

import numpy as np
import pytest

import opennve

DATA = pathlib.Path(__file__).parent / "data"
EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])


def _sorted(events, polarity_field="on"):
    order = np.lexsort((events[polarity_field], events["x"], events["y"], events["t"]))
    return events[order]


def _same_events(a, b, b_polarity="on"):
    a, b = _sorted(a), _sorted(b, b_polarity)
    return len(a) == len(b) and all(
        np.array_equal(a[x], b[y]) for x, y in (("t", "t"), ("x", "x"), ("y", "y"), ("on", b_polarity))
    )


def _random_events(seed=0, count=2000, start=1_000_000, span=200_000, width=1280, height=720):
    rng = np.random.default_rng(seed)
    events = np.zeros(count, dtype=EVENT_DTYPE)
    events["t"] = np.sort(rng.integers(start, start + span, count))
    events["x"] = rng.integers(0, width, count)
    events["y"] = rng.integers(0, height, count)
    events["on"] = rng.random(count) < 0.5
    return events


# ------------------------------------------------------------------ test encoders


def _header(format_string: str, extra: str = "") -> bytes:
    return (f"% camera_integrator_name SENSING\n% format {format_string}\n{extra}% end\n").encode()


def encode_evt3(events, triggers=(), vectorize=True) -> bytes:
    """Simple EVT 3.0 encoder: time high/low, y, single events or 12/8 vectors per row segment."""
    words = []
    high = low = y = None
    items = [(int(e["t"]), 0, int(e["y"]), int(e["x"]), bool(e["on"])) for e in events]
    items += [(int(t), 1, trigger_id, 0, rising) for t, trigger_id, rising in triggers]
    items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    index = 0
    while index < len(items):
        t, kind, value, x, on = items[index]
        if t >> 12 != high:
            high = t >> 12
            words.append(0x8000 | (high & 0xFFF))
            low = None
        if t & 0xFFF != low:
            low = t & 0xFFF
            words.append(0x6000 | low)
        if kind == 1:
            words.append(0xA000 | (value << 8) | int(on))
            index += 1
            continue
        if value != y:
            y = value
            words.append(0x0000 | y)
        # group consecutive events with the same t, y and polarity
        group = [x]
        while (
            vectorize
            and index + len(group) < len(items)
            and items[index + len(group)][:3] == (t, 0, value)
            and items[index + len(group)][4] == on
            and items[index + len(group)][3] - x < 32
        ):
            group.append(items[index + len(group)][3])
        if len(group) == 1:
            words.append(0x2000 | (int(on) << 11) | x)
        else:
            words.append(0x3000 | (int(on) << 11) | x)
            mask = 0
            for gx in group:
                mask |= 1 << (gx - x)
            words += [0x4000 | (mask & 0xFFF), 0x4000 | ((mask >> 12) & 0xFFF), 0x5000 | ((mask >> 24) & 0xFF)]
        index += len(group)
    return np.array(words, dtype="<u2").tobytes()


def encode_evt2(events, triggers=()) -> bytes:
    words = []
    high = None
    items = [(int(e["t"]), 0, e) for e in events] + [(int(t), 1, (i, r)) for t, i, r in triggers]
    items.sort(key=lambda item: (item[0], item[1]))
    for t, kind, item in items:
        if t >> 6 != high:
            high = t >> 6
            words.append(0x80000000 | (high & 0xFFFFFFF))
        if kind == 0:
            words.append((int(item["on"]) << 28) | ((t & 0x3F) << 22) | (int(item["x"]) << 11) | int(item["y"]))
        else:
            trigger_id, rising = item
            words.append(0xA0000000 | ((t & 0x3F) << 22) | (trigger_id << 8) | int(rising))
    return np.array(words, dtype="<u4").tobytes()


def encode_evt21(events, legacy=False) -> bytes:
    words = []
    high = None
    events = np.sort(events, order=["t", "y", "on", "x"])
    index = 0
    while index < len(events):
        e = events[index]
        t = int(e["t"])
        if t >> 6 != high:
            high = t >> 6
            content = 0x80000000 | (high & 0xFFFFFFF)
            words.append(content if legacy else content << 32)
        base = int(e["x"]) & ~31  # 32-pixel aligned groups
        mask = 0
        while (
            index < len(events)
            and int(events[index]["t"]) == t
            and int(events[index]["y"]) == int(e["y"])
            and bool(events[index]["on"]) == bool(e["on"])
            and int(events[index]["x"]) - base < 32
        ):
            mask |= 1 << (int(events[index]["x"]) - base)
            index += 1
        head = (int(e["on"]) << 28) | ((t & 0x3F) << 22) | (base << 11) | int(e["y"])
        words.append((mask << 32) | head if legacy else (head << 32) | mask)
    return np.array(words, dtype="<u8").tobytes()


# ------------------------------------------------------------------ reference files (written by faery)


@pytest.mark.parametrize("name", ["faery_evt3.raw", "faery_evt2.raw"])
def test_reference_files(name):
    reference = np.load(DATA / "faery_reference.npy")
    events, triggers = opennve.read_raw(DATA / name)
    assert _same_events(events, reference, "p")  # crosses the EVT 3.0 24-bit time loop
    assert len(triggers) == 0


@pytest.mark.parametrize("chunk_bytes", [8, 64, 4096])
def test_small_chunks_give_the_same_events(chunk_bytes):
    reference, _ = opennve.read_raw(DATA / "faery_evt3.raw")
    events, _ = opennve.read_raw(DATA / "faery_evt3.raw", chunk_bytes=chunk_bytes)
    assert np.array_equal(events, reference)


def test_header():
    header = opennve.read_header(DATA / "faery_evt3.raw")
    assert (header.format, header.width, header.height) == ("EVT3", 1280, 720)
    assert header.data_offset > 0


# ------------------------------------------------------------------ formats


@pytest.mark.parametrize("vectorize", [False, True])
def test_evt3_round_trip(tmp_path, vectorize):
    events = _random_events(1)
    # dense row segments to exercise the vectors
    segment = np.zeros(40, dtype=EVENT_DTYPE)
    segment["t"], segment["x"], segment["y"], segment["on"] = 1_100_000, np.arange(100, 140), 5, True
    events = np.sort(np.concatenate([events, segment]), order="t")
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events, vectorize=vectorize))
    decoded, triggers = opennve.read_raw(path)
    assert _same_events(decoded, events)


def test_evt3_triggers_and_em_rows(tmp_path):
    events = _random_events(2, count=50)
    triggers = [(1_050_000, 3, True), (1_060_000, 3, False)]
    data = encode_evt3(events, triggers)
    # An EM row (type 1) followed by an event that must be ignored, then a CD row
    words = np.frombuffer(data, dtype="<u2").tolist()
    words += [0x1000 | 10, 0x2000 | 20, 0x0000 | 11, 0x2000 | (1 << 11) | 21]
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + np.array(words, dtype="<u2").tobytes())
    decoded, decoded_triggers = opennve.read_raw(path)
    assert len(decoded) == len(events) + 1
    assert (int(decoded[-1]["x"]), int(decoded[-1]["y"]), bool(decoded[-1]["on"])) == (21, 11, True)
    assert decoded_triggers.tolist() == [(1_050_000, 3, True), (1_060_000, 3, False)]


def test_evt2_round_trip_with_triggers(tmp_path):
    events = _random_events(3)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT2;height=720;width=1280") + encode_evt2(events, [(1_100_000, 1, True)]))
    decoded, triggers = opennve.read_raw(path)
    assert _same_events(decoded, events)
    assert triggers.tolist() == [(1_100_000, 1, True)]


@pytest.mark.parametrize("legacy", [False, True])
def test_evt21_round_trip(tmp_path, legacy):
    events = _random_events(4, height=720)
    segment = np.zeros(30, dtype=EVENT_DTYPE)
    segment["t"], segment["x"], segment["y"], segment["on"] = 1_100_000, np.arange(64, 94), 3, False
    events = np.concatenate([events, segment])
    path = tmp_path / "a.raw"
    endianness = ";endianness=legacy" if legacy else ""
    path.write_bytes(_header(f"EVT21;height=720;width=1280{endianness}") + encode_evt21(events, legacy=legacy))
    decoded, _ = opennve.read_raw(path)
    assert _same_events(decoded, events)


def test_legacy_header_and_overrides(tmp_path):
    events = _random_events(5, count=100)
    path = tmp_path / "old.raw"
    path.write_bytes(b"% evt 3.0\n% geometry 640x480\n" + encode_evt3(_random_events(5, 100, width=640, height=480)))
    header = opennve.read_header(path)
    assert (header.format, header.width, header.height) == ("EVT3", 640, 480)
    # file without header: format and size passed explicitly
    bare = tmp_path / "bare.raw"
    bare.write_bytes(encode_evt3(events))
    with pytest.raises(ValueError, match="event format is unknown"):
        opennve.read_raw(bare)
    decoded, _ = opennve.read_raw(bare, format="EVT3", width=1280, height=720)
    assert _same_events(decoded, events)


def test_unsupported_format(tmp_path):
    path = tmp_path / "a.raw"
    path.write_bytes(b"% format HISTO3D;height=480;width=640\n% end\n")
    with pytest.raises(ValueError, match="unsupported RAW event format"):
        opennve.open_raw(path)


def test_evt2_time_loop():
    decoder = opennve.recordings.raw._Evt2Decoder(1280, 720, vectorized=False)
    near_end = (1 << 34) - 64
    words = np.array(
        [
            0x80000000 | (near_end >> 6),
            (1 << 28) | (10 << 22) | (5 << 11) | 6,  # ON event at near_end + 10
            0x80000000 | 0,  # time high wraps to 0
            (0 << 28) | (3 << 22) | (7 << 11) | 8,  # OFF event at 2^34 + 3
        ],
        dtype="<u4",
    )
    events, _, _ = decoder.decode(words)
    assert events["t"].tolist() == [near_end + 10, (1 << 34) + 3]


# ------------------------------------------------------------------ reader


def test_reader_with_camera_config(tmp_path):
    events = _random_events(6, count=3000, span=200_000)
    path = tmp_path / "left.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280", "% serial_number 05110634\n") + encode_evt3(events))
    config = {
        "name": "left",
        "model": "SE1-S4",
        "filter": {"polarity": "on"},
        "output": {"fields": ["t", "x"]},
        "window": {"duration_ms": 50},
    }
    with opennve.open_raw(path, config) as raw:
        info = raw.info()
        assert info["serial"] == "05110634" and info["format"] == "EVT3" and raw.resolution == (1280, 720)
        packets = list(raw.events())
    assert all(packet.camera == "left" for packet in packets)
    on_events = np.concatenate([packet.polarity_events for packet in packets])
    assert np.all(on_events["on"]) and len(on_events) == int(np.count_nonzero(events["on"]))
    assert packets[0].events.dtype.names == ("t", "x")
    assert not raw.is_open

    with opennve.open_raw(path, config) as raw:
        windows = list(raw.windows())
    # 1.0 s .. 1.2 s in 50 ms windows; the last window is returned at the end of the file
    assert [w.start_t for w in windows] == [1_000_000, 1_050_000, 1_100_000, 1_150_000]
    assert sum(len(w) for w in windows) == len(on_events)
    assert windows[0].to_frame("count").shape == (720, 1280)


def test_reader_options(tmp_path):
    events = _random_events(7, count=500, start=5_000_000)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events))
    decoded, _ = opennve.read_raw(path, zero_time=True, filter={"background_activity": {"enabled": True}})
    assert len(decoded) < len(events)  # random events are mostly isolated noise
    decoded, _ = opennve.read_raw(path, zero_time=True)
    assert int(decoded["t"][0]) == 0 and int(decoded["t"][-1]) == int(events["t"][-1] - events["t"][0])
    with opennve.open_raw(path, name="replay", window=None) as raw:
        assert raw.name == "replay"
        with pytest.raises(ValueError, match="no window duration"):
            raw.read_windows()
    with pytest.raises(ValueError, match="speed"):
        opennve.open_raw(path, speed=0)


def test_reader_from_config_file(tmp_path):
    events = _random_events(8, count=100)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events))
    config_path = tmp_path / "cameras.json"
    opennve.save_configs(
        config_path,
        [opennve.CameraConfig(model="SE1-S4", name="left"), opennve.CameraConfig(model="SE1-S4", name="right")],
    )
    with opennve.open_raw(path, config_path=config_path, name="right") as raw:
        assert raw.name == "right"
        assert sum(len(packet.events) for packet in raw.events()) == 100


def test_cli_raw(capsys):
    from opennve.cli import main

    assert main(["raw", str(DATA / "faery_evt3.raw")]) == 0
    output = capsys.readouterr().out
    assert "EVT3, 1280 x 720" in output and "events:     10262" in output
    assert main(["raw", str(DATA / "faery_evt2.raw"), "--json"]) == 0
    import json

    info = json.loads(capsys.readouterr().out)
    assert info["events"] == 10262 and info["format"] == "EVT2" and info["first_t"] == 16700378


def test_realtime_playback(tmp_path, monkeypatch):
    from opennve.recordings import raw as raw_module

    events = _random_events(9, count=200, start=1_000_000, span=2_000_000)
    path = tmp_path / "a.raw"
    path.write_bytes(_header("EVT3;height=720;width=1280") + encode_evt3(events))
    clock = {"now": 100.0}
    sleeps = []
    monkeypatch.setattr(raw_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(raw_module.time, "sleep", lambda seconds: (sleeps.append(seconds), clock.__setitem__("now", clock["now"] + seconds)))
    with opennve.open_raw(path, speed=2.0, chunk_bytes=64) as raw:
        list(raw.events())
    # 2 s of events at twice the speed: about 1 s of waiting in total
    assert 0.9 < sum(sleeps) < 1.01
