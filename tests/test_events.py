import numpy as np

from eyesopenscan.events import (
    Params, analyze_eyes, analyze_motion, fuse, hysteresis, mask_to_intervals, merge_intervals,
)
from eyesopenscan.motion import rhythmicity


def _eye_signal(dur=600, fps=5, opens=((100, 130), (300, 301), (400, 410))):
    t = np.arange(0, dur, 1 / fps)
    rng = np.random.default_rng(0)
    ear = 0.12 + rng.normal(0, 0.005, len(t))
    for s, e in opens:
        ear[(t >= s) & (t < e)] = 0.30
    face = np.ones(len(t), bool)
    return t, ear, face


def test_hysteresis_holds_between_thresholds():
    x = np.array([0, 0.5, 0.35, 0.35, 0.2, 0.5, np.nan, 0.5])
    assert hysteresis(x, 0.4, 0.3).tolist() == [False, True, True, True, False, True, False, True]


def test_intervals_and_merge():
    t = np.arange(10.0)
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], bool)
    assert mask_to_intervals(t, m, 1.0) == [(1, 3), (5, 6), (8, 10)]
    assert merge_intervals([(1, 3), (5, 6), (8, 10)], 1.5) == [(1, 3), (5, 6), (8, 10)]
    assert merge_intervals([(1, 3), (5, 6), (9, 10)], 2.0) == [(1, 6), (9, 10)]


def test_eye_events_auto_threshold_ignores_blink():
    t, ear, face = _eye_signal()
    a = analyze_eyes(t, ear, face, Params())
    # the 1 s "blink" at 300 s is dropped (< 2 s min duration)
    assert len(a.intervals) == 2
    (s1, e1), (s2, e2) = a.intervals
    assert abs(s1 - 100) < 1 and abs(e1 - 130) < 1
    assert abs(s2 - 400) < 1 and abs(e2 - 410) < 1


def test_no_face_is_not_eyes_open():
    t, ear, face = _eye_signal(opens=())
    face[(t > 200) & (t < 300)] = False
    ear[~face] = 0.5  # garbage values while the face is missing must be ignored
    a = analyze_eyes(t, ear, face, Params())
    assert a.intervals == []


def test_motion_and_fusion():
    fps = 15
    t = np.arange(0, 1200, 1 / fps)
    rng = np.random.default_rng(1)
    energy = np.abs(rng.normal(0.002, 0.001, len(t)))
    seiz = (t >= 600) & (t < 640)
    energy[seiz] = 0.15 + 0.1 * np.sin(2 * np.pi * 4 * t[seiz])  # 4 Hz jerking
    m = analyze_motion(t, energy, Params())
    assert len(m.intervals) == 1 and abs(m.intervals[0][0] - 600) < 2

    te = np.arange(0, 1200, 0.2)
    ear = np.full(len(te), 0.12)
    ear[(te >= 605) & (te < 650)] = 0.3
    face = np.ones(len(te), bool)
    eyes = analyze_eyes(te, ear, face, Params())
    ev = fuse(eyes, m, te, face, t, energy, fps, Params())
    assert len(ev) == 1
    assert ev[0].kind == "both" and ev[0].confidence == "high"
    assert ev[0].rhythmicity > 0.5


def _bursty_session():
    """Like the real 2026-11-05 22:44 session: three ~1.2 s motion bursts with
    short dips, while the eyes are open (with a brief face dropout)."""
    fps = 15
    t = np.arange(0, 60, 1 / fps)
    energy = np.full(len(t), 0.001)
    for s in (18.7, 20.0, 22.7):
        energy[(t >= s) & (t < s + 1.2)] = 0.06
    te = np.arange(0, 60, 0.2)
    ear = np.full(len(te), 0.10)
    ear[(te >= 17.6) & (te < 26.6)] = 0.28
    face = np.ones(len(te), bool)
    face[(te >= 19.6) & (te < 20.0)] = False  # brief face dropout mid-event
    return fps, t, energy, te, ear, face


def test_motion_bursts_with_short_dips_are_one_episode():
    fps, t, energy, *_ = _bursty_session()
    m = analyze_motion(t, energy, Params())
    assert len(m.intervals) == 1
    s, e = m.intervals[0]
    assert abs(s - 18.7) < 0.5 and abs(e - 23.9) < 0.5


def test_single_short_burst_still_ignored():
    fps = 15
    t = np.arange(0, 60, 1 / fps)
    energy = np.full(len(t), 0.001)
    energy[(t >= 30) & (t < 31.2)] = 0.06  # one turn-over-like burst
    assert analyze_motion(t, energy, Params()).intervals == []


def test_bursty_motion_with_open_eyes_is_high_confidence():
    fps, t, energy, te, ear, face = _bursty_session()
    p = Params(open_thr=0.176, close_thr=0.148)
    eyes = analyze_eyes(te, ear, face, p)
    assert len(eyes.intervals) == 1  # face dropout bridged, not split
    ev = fuse(eyes, analyze_motion(t, energy, p), te, face, t, energy, fps, p)
    assert len(ev) == 1
    assert ev[0].kind == "both" and ev[0].confidence == "high"
    assert ev[0].motion_s > 4


def test_rhythmicity_low_for_single_movement():
    fps = 15
    t = np.arange(0, 10, 1 / fps)
    turn = np.exp(-((t - 5) ** 2))  # one smooth turn-over
    assert rhythmicity(turn, fps) < 0.3
