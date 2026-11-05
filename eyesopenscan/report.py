"""CSV, text summary and timeline plot."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .events import Event, EyeAnalysis, MotionAnalysis  # noqa: E402
from .video import fmt_clock, fmt_time  # noqa: E402

KIND_COLORS = {"both": "#d62728", "eyes": "#ff7f0e", "motion": "#1f77b4"}


def events_frame(events: list[Event], start_clock: dt.datetime | None) -> pd.DataFrame:
    rows = []
    for i, e in enumerate(events, 1):
        rows.append({
            "event": i,
            "start": fmt_time(e.start),
            "end": fmt_time(e.end),
            "start_s": round(e.start, 2),
            "end_s": round(e.end, 2),
            "duration_s": round(e.duration, 1),
            "clock_start": fmt_clock(start_clock, e.start),
            "type": e.kind,
            "confidence": e.confidence,
            "eyes_open_s": round(e.eyes_open_s, 1),
            "motion_s": round(e.motion_s, 1),
            "face_visible": round(e.face_visible, 2) if np.isfinite(e.face_visible) else "",
            "max_ear": round(e.max_ear, 3) if np.isfinite(e.max_ear) else "",
            "peak_motion": round(e.peak_motion, 3) if np.isfinite(e.peak_motion) else "",
            "rhythmicity": round(e.rhythmicity, 2) if np.isfinite(e.rhythmicity) else "",
        })
    cols = ["event", "start", "end", "start_s", "end_s", "duration_s", "clock_start", "type",
            "confidence", "eyes_open_s", "motion_s", "face_visible", "max_ear", "peak_motion",
            "rhythmicity"]
    return pd.DataFrame(rows, columns=cols)


def summary_text(
    events: list[Event],
    video: str,
    duration: float,
    eyes: EyeAnalysis | None,
    start_clock: dt.datetime | None,
    gaps: list | None = None,
    missing_clips: list[int] | None = None,
    reconnects: int = 0,
    note: str = "",
    status: str = "complete",
) -> str:
    L = [
        "eyesopenscan summary",
        "=" * 40,
        f"video:            {video}",
        f"recording length: {fmt_time(duration)}",
    ]
    if start_clock:
        L.append(f"recording start:  {start_clock:%Y-%m-%d %H:%M:%S}")
        L.append(f"recording end:    {start_clock + dt.timedelta(seconds=duration):%Y-%m-%d %H:%M:%S}")
    if status != "complete":
        L.append("status:           INCOMPLETE - session did not stop cleanly; data up to the last checkpoint")
    if gaps:
        offline = sum(e - s for s, e in gaps)
        L.append(f"stream offline:   {fmt_time(offline)} in {len(gaps)} gap(s), {reconnects} reconnect(s)")
        for s, e in gaps:
            when = f"{fmt_clock(start_clock, s)[11:]}-{fmt_clock(start_clock, e)[11:]}" if start_clock else f"{fmt_time(s)}-{fmt_time(e)}"
            L.append(f"    {when}  ({e - s:.0f} s, nothing analyzed)")
    if eyes is not None:
        L.append(f"face visible:     {eyes.face_frac * 100:.0f}% of the night")
        L.append(f"EAR thresholds:   open > {eyes.open_thr:.3f}, close < {eyes.close_thr:.3f}  [{eyes.thr_source}]")
    L.append("")

    if not events:
        L.append("No events detected.")
        return "\n".join(L) + "\n"

    durs = np.array([e.duration for e in events])
    starts = np.array([e.start for e in events])
    hours = duration / 3600 if duration > 0 else float("nan")
    L.append(f"events:           {len(events)}  ({len(events) / hours:.2f} per hour)")
    for k in ("both", "eyes", "motion"):
        n = sum(e.kind == k for e in events)
        if n:
            L.append(f"  {k:<7}         {n}")
    L.append(f"total duration:   {fmt_time(durs.sum())}  ({durs.sum():.0f} s)")
    L.append(f"mean duration:    {durs.mean():.1f} s   median {np.median(durs):.1f} s")
    longest = events[int(durs.argmax())]
    L.append(f"longest:          {durs.max():.1f} s at {fmt_time(longest.start)}"
             + (f" ({fmt_clock(start_clock, longest.start)})" if start_clock else ""))
    if len(events) > 1:
        gaps = np.diff(starts) / 60
        L.append(f"time between:     mean {gaps.mean():.1f} min, min {gaps.min():.1f} min, max {gaps.max():.1f} min")
    if missing_clips:
        L.append(f"no video for event(s): {', '.join(map(str, missing_clips))} "
                 "(outside the kept buffer)")
    if note:
        L.append(f"clips:            {note}")
    L.append("")
    L.append("High-confidence = eyes open AND strong movement at the same time.")
    L.append("Every event is a candidate for review, not a diagnosis.")
    return "\n".join(L) + "\n"


def _binmax(t: np.ndarray, y: np.ndarray, nbins: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    if len(t) <= nbins:
        return t, y
    edges = np.linspace(t[0], t[-1], nbins + 1)
    idx = np.clip(np.searchsorted(edges, t) - 1, 0, nbins - 1)
    out = np.full(nbins, np.nan)
    df = pd.DataFrame({"i": idx, "y": y}).groupby("i")["y"].max()
    out[df.index.to_numpy()] = df.to_numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, out


def timeline_png(
    path: Path,
    events: list[Event],
    t_eye: np.ndarray | None,
    eyes: EyeAnalysis | None,
    face: np.ndarray | None,
    t_mot: np.ndarray | None,
    motion: MotionAnalysis | None,
    duration: float,
    gaps: list | None = None,
) -> None:
    rows = (eyes is not None) + (motion is not None) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.4 * rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    h = lambda x: np.asarray(x) / 3600.0  # noqa: E731
    i = 0

    if eyes is not None and t_eye is not None:
        ax = axes[i]; i += 1
        x, y = _binmax(t_eye, eyes.ear_smooth)
        ax.plot(h(x), y, lw=0.6, color="#444")
        ax.axhline(eyes.open_thr, color=KIND_COLORS["eyes"], ls="--", lw=0.8, label="open threshold")
        ax.axhline(eyes.close_thr, color="#999", ls=":", lw=0.8, label="close threshold")
        if face is not None:
            nf = ~face.astype(bool)
            for s, e in _runs(t_eye, nf):
                if e - s > 30:
                    ax.axvspan(h(s), h(e), color="#bbb", alpha=0.35, lw=0)
        ax.set_ylabel("eye openness\n(EAR)")
        ax.legend(loc="upper right", fontsize=7)
        ax.text(0.003, 0.92, "grey = face not visible", transform=ax.transAxes, fontsize=7, va="top")

    if motion is not None and t_mot is not None:
        ax = axes[i]; i += 1
        x, y = _binmax(t_mot, motion.energy_smooth)
        ax.plot(h(x), y, lw=0.6, color="#444")
        xt, yt = _binmax(t_mot, motion.threshold)
        ax.plot(h(xt), yt, lw=0.8, ls="--", color=KIND_COLORS["motion"], label="threshold")
        ax.set_ylabel("motion\n(changed px)")
        ax.legend(loc="upper right", fontsize=7)

    ax = axes[i]
    # Short events would be invisible on an 8 h axis, so give each band a
    # minimum width relative to the recording length - never past its end.
    min_w = 0.0025 * duration if duration > 0 else 0.0
    for n, e in enumerate(events, 1):
        end = max(e.end, e.start + min_w)
        if duration > 0:
            end = min(end, duration)
        ax.axvspan(h(e.start), h(end), ymax=0.8, color=KIND_COLORS[e.kind], alpha=0.85, lw=0)
        ax.text(h((e.start + end) / 2), 0.9, f"#{n} {e.confidence}", ha="center", va="center",
                fontsize=7, transform=ax.get_xaxis_transform(), clip_on=True)
    ax.set_yticks([])
    ax.set_ylabel("events")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLORS.values()]
    ax.legend(handles, ["eyes + motion (high)", "eyes only (medium)", "motion only (medium if rhythmic)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7, ncol=3, frameon=False)
    for a in axes:
        for gs, ge in gaps or []:
            a.axvspan(h(gs), h(ge), color="#f2c4c4", alpha=0.5, lw=0, hatch="//")
    ax.set_xlabel("hours into recording" + ("   (hatched = stream offline)" if gaps else ""))
    ax.set_xlim(0, h(duration) if duration > 0 else None)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _runs(t: np.ndarray, m: np.ndarray):
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    return [(t[a], t[b]) for a, b in zip(s, e)]
