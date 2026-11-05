"""Turn per-frame signals into eyes-open / motion / fused events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .motion import rhythmicity


@dataclass
class Params:
    # eyes
    open_thr: float | None = None  # EAR above this -> eyes opening (None = auto)
    close_thr: float | None = None  # EAR below this -> eyes closing (None = auto)
    eye_smooth_s: float = 1.0  # rolling-median window over EAR
    eye_min_dur_s: float = 2.0  # ignore shorter openings (blinks, twitches)
    eye_join_gap_s: float = 1.0  # bridge brief dips/face dropouts before the min-duration check
    # motion
    motion_smooth_s: float = 0.5
    motion_baseline_s: float = 300.0  # rolling baseline window
    motion_k: float = 6.0  # threshold = baseline + k * robust sigma
    motion_min_abs: float = 0.02  # ...but at least this fraction of ROI pixels
    motion_min_dur_s: float = 3.0
    # Jerking comes in bursts with short pauses; bursts closer than this are one
    # movement episode, judged against motion_min_dur_s as a whole.
    motion_join_gap_s: float = 2.0
    # fusion
    merge_gap_s: float = 3.0  # join events separated by less than this
    min_event_s: float = 2.0
    auto_open_ratio: float = 1.6  # auto thresholds relative to typical (closed) EAR
    auto_close_ratio: float = 1.35

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- interval helpers

Interval = tuple[float, float]


def mask_to_intervals(t: np.ndarray, mask: np.ndarray, dt: float) -> list[Interval]:
    """Contiguous True runs -> [(start, end)], end = last True sample + dt."""
    t = np.asarray(t, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if len(t) == 0 or not m.any():
        return []
    d = np.diff(np.concatenate([[False], m, [False]]).astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(float(t[s]), float(t[e] + dt)) for s, e in zip(starts, ends)]


def merge_intervals(iv: list[Interval], gap: float) -> list[Interval]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def filter_min_duration(iv: list[Interval], min_dur: float) -> list[Interval]:
    return [(s, e) for s, e in iv if e - s >= min_dur]


def overlap(a: Interval, iv: list[Interval]) -> float:
    return float(sum(max(0.0, min(a[1], e) - max(a[0], s)) for s, e in iv))


def hysteresis(x: np.ndarray, high: float, low: float) -> np.ndarray:
    """True once x rises above `high`, until it falls below `low`. NaN -> False."""
    out = np.zeros(len(x), dtype=bool)
    state = False
    for i, v in enumerate(x):
        if not np.isfinite(v):
            state = False
        elif state:
            state = v >= low
        else:
            state = v > high
        out[i] = state
    return out


def _period(t: np.ndarray) -> float:
    return float(np.median(np.diff(t))) if len(t) > 1 else 1.0


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    return pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()


# ---------------------------------------------------------------- eyes


@dataclass
class EyeAnalysis:
    ear_smooth: np.ndarray
    open_mask: np.ndarray
    open_thr: float
    close_thr: float
    thr_source: str
    intervals: list[Interval]
    face_frac: float


def analyze_eyes(t: np.ndarray, ear: np.ndarray, face: np.ndarray, p: Params) -> EyeAnalysis:
    dt = _period(t)
    ear = np.where(face, ear, np.nan)
    sm = _rolling_median(ear, round(p.eye_smooth_s / dt) | 1)
    # Don't let smoothing paint values into long no-face stretches.
    sm = np.where(_near_face(face, int(round(p.eye_smooth_s / dt))), sm, np.nan)

    if p.open_thr is not None and p.close_thr is not None:
        hi, lo, src = p.open_thr, p.close_thr, "manual/calibrated"
    else:
        # A sleeper's eyes are closed most of the time the face is visible, so
        # the median EAR approximates the closed-eye level for this person/angle.
        vis = sm[np.isfinite(sm)]
        ref = float(np.median(vis)) if len(vis) else 0.15
        hi, lo, src = ref * p.auto_open_ratio, ref * p.auto_close_ratio, f"auto (median EAR {ref:.3f})"
    if lo > hi:
        lo = hi

    mask = hysteresis(sm, hi, lo)
    iv = merge_intervals(mask_to_intervals(t, mask, dt), p.eye_join_gap_s)
    iv = filter_min_duration(iv, p.eye_min_dur_s)
    face_frac = float(np.mean(face)) if len(face) else 0.0
    return EyeAnalysis(sm, mask, hi, lo, src, iv, face_frac)


def _near_face(face: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return face.astype(bool)
    k = np.ones(2 * radius + 1)
    return np.convolve(face.astype(float), k, mode="same") > 0


# ---------------------------------------------------------------- motion


@dataclass
class MotionAnalysis:
    energy_smooth: np.ndarray
    threshold: np.ndarray
    active_mask: np.ndarray
    intervals: list[Interval]


def analyze_motion(t: np.ndarray, energy: np.ndarray, p: Params) -> MotionAnalysis:
    dt = _period(t)
    s = pd.Series(energy).interpolate(limit=int(2 / dt), limit_area="inside")
    sm = s.rolling(max(1, int(round(p.motion_smooth_s / dt))), center=True, min_periods=1).mean()
    win = max(3, int(round(p.motion_baseline_s / dt)))
    med = sm.rolling(win, center=True, min_periods=1).median()
    mad = (sm - med).abs().rolling(win, center=True, min_periods=1).median()
    thr = np.maximum(med + p.motion_k * 1.4826 * mad, p.motion_min_abs).to_numpy()
    smv = sm.to_numpy()
    active = np.nan_to_num(smv, nan=0.0) > thr
    iv = merge_intervals(mask_to_intervals(t, active, dt), p.motion_join_gap_s)
    iv = filter_min_duration(iv, p.motion_min_dur_s)
    return MotionAnalysis(smv, thr, active, iv)


# ---------------------------------------------------------------- fusion


@dataclass
class Event:
    start: float
    end: float
    kind: str  # "both" | "eyes" | "motion"
    eyes_open_s: float
    motion_s: float
    face_visible: float  # fraction of eye samples in event with a face
    max_ear: float
    peak_motion: float
    rhythmicity: float
    confidence: str = field(default="")

    @property
    def duration(self) -> float:
        return self.end - self.start


def fuse(
    eyes: EyeAnalysis | None,
    motion: MotionAnalysis | None,
    t_eye: np.ndarray,
    face: np.ndarray,
    t_mot: np.ndarray,
    energy_raw: np.ndarray,
    mot_fps: float,
    p: Params,
) -> list[Event]:
    eye_iv = eyes.intervals if eyes else []
    mot_iv = motion.intervals if motion else []
    merged = filter_min_duration(merge_intervals(eye_iv + mot_iv, p.merge_gap_s), p.min_event_s)

    events: list[Event] = []
    for s, e in merged:
        eo = overlap((s, e), eye_iv)
        mo = overlap((s, e), mot_iv)
        kind = "both" if eo > 0 and mo > 0 else ("eyes" if eo > 0 else "motion")

        sel_e = (t_eye >= s) & (t_eye < e)
        fv = float(face[sel_e].mean()) if sel_e.any() else float("nan")
        max_ear = float(np.nanmax(eyes.ear_smooth[sel_e])) if eyes is not None and sel_e.any() and np.isfinite(eyes.ear_smooth[sel_e]).any() else float("nan")

        sel_m = (t_mot >= s) & (t_mot < e)
        if motion is not None and sel_m.any():
            peak = float(np.nanmax(motion.energy_smooth[sel_m])) if np.isfinite(motion.energy_smooth[sel_m]).any() else float("nan")
            rhy = rhythmicity(energy_raw[sel_m], mot_fps)
        else:
            peak, rhy = float("nan"), float("nan")

        if kind == "both":
            conf = "high"
        elif kind == "motion" and np.isfinite(rhy) and rhy >= 0.4:
            conf = "medium"
        elif kind == "eyes":
            conf = "medium"
        else:
            conf = "low"
        events.append(Event(s, e, kind, eo, mo, fv, max_ear, peak, rhy, conf))
    return events
