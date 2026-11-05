import datetime as dt

import pytest

from eyesopenscan.events import Params
from eyesopenscan.live import LiveFlagger, SegmentRing, redact
from eyesopenscan.video import parse_until


def _fake_conn(ring: SegmentRing, offset: float, segs: list[tuple[str, float, float]], partial=None):
    c = ring.new_conn()
    ring.set_offset(c, offset)
    d = ring.dir / c.dir
    lines = []
    for name, s, e in segs:
        (d / name).write_bytes(b"x" * 10)
        lines.append(f"{name},{s:.6f},{e:.6f}\n")
    (d / "segments.csv").write_text("".join(lines))
    if partial:
        (d / partial).write_bytes(b"x")
    return c


@pytest.fixture
def ring(tmp_path):
    r = SegmentRing(tmp_path / "buffer", retention_s=300, segment_s=30)
    # connection 0: session time 0..120 ; connection 1 (after a 40 s outage): 160..
    _fake_conn(r, 0.0, [("a0.mkv", 0, 30), ("a1.mkv", 30, 60), ("a2.mkv", 60, 90), ("a3.mkv", 90, 120)])
    _fake_conn(r, 160.0, [("b0.mkv", 0, 30), ("b1.mkv", 30, 60)], partial="b2.mkv")
    return r


def test_segments_map_to_session_time(ring):
    segs = {s.path.name: s for s in ring.segments()}
    assert (segs["a2.mkv"].start, segs["a2.mkv"].end) == (60, 90)
    assert (segs["b1.mkv"].start, segs["b1.mkv"].end) == (190, 220)
    assert segs["b2.mkv"].start == 220 and not segs["b2.mkv"].complete


def test_covering_spans_segments_and_connections(ring):
    names = [s.path.name for s in ring.covering(50, 175)]
    assert names == ["a1.mkv", "a2.mkv", "a3.mkv", "b0.mkv"]


def test_prune_keeps_flagged_recent_and_partial(ring):
    ring.add_keep(65, 70)  # inside a2
    deleted = ring.prune(now=500)  # everything complete is older than 300 s except b1 (ends 220 > 200)
    left = sorted(s.path.name for s in ring.segments())
    assert left == ["a2.mkv", "b1.mkv", "b2.mkv"]
    assert deleted == 4  # a0, a1, a3, b0


def test_ring_state_survives_reload(ring):
    ring.add_keep(10, 20)
    ring.save()
    again = SegmentRing(ring.dir)
    assert [c.offset for c in again.conns] == [0.0, 160.0]
    assert again.keep == [(10, 20)]
    assert len(again.segments()) == 7


def test_flagger_warmup_then_thresholds():
    f = LiveFlagger(Params(close_thr=0.12, motion_min_abs=0.02), warmup_s=10)
    assert f.eye(5, False, float("nan"))  # warm-up keeps everything
    assert not f.eye(20, True, 0.08)
    assert not f.eye(21, True, 0.15)  # a single high sample is ignored...
    assert f.eye(21.2, True, 0.15)  # ...two in a row are flagged
    assert not f.eye(22, False, 0.5)  # no face -> never "eyes open"
    # motion must exceed half the min threshold for >= 1 s
    assert not f.motion(30.0, 0.015)
    assert not f.motion(30.5, 0.015)
    assert f.motion(31.0, 0.015)
    assert not f.motion(31.1, 0.001)


def test_flagger_auto_threshold_from_median():
    f = LiveFlagger(Params(), warmup_s=0)
    for i in range(200):
        f.eye(i, True, 0.08)
    assert f.eye_threshold() == pytest.approx(0.08 * Params().auto_close_ratio)
    f.eye(300, True, 0.2)
    assert f.eye(300.2, True, 0.2)


def test_parse_until():
    now = dt.datetime(2026, 11, 5, 22, 0)
    assert parse_until("07:30", now) == dt.datetime(2026, 11, 6, 7, 30)
    assert parse_until("23:15", now) == dt.datetime(2026, 11, 5, 23, 15)
    assert parse_until("+8h", now) == dt.datetime(2026, 11, 6, 6, 0)
    assert parse_until("+90m", now) == dt.datetime(2026, 11, 5, 23, 30)


def test_redact_password():
    assert redact("rtsp://admin:s3cret@192.168.1.5:554/stream1") == "rtsp://***@192.168.1.5:554/stream1"
    assert redact("rtsp://192.168.1.5/live") == "rtsp://192.168.1.5/live"
