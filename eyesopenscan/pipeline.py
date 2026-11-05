"""Extraction pass over the video, and analysis of the saved signals."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import report
from .events import Params, analyze_eyes, analyze_motion, fuse
from .eyes import DEFAULT_MODEL, EyeDetector
from .motion import MotionMeter
from .video import export_clip, fmt_time, iter_frames, probe


@dataclass
class ExtractOptions:
    eye_fps: float = 5.0
    motion_fps: float = 15.0
    face_roi: tuple[int, int, int, int] | None = None
    body_roi: tuple[int, int, int, int] | None = None
    rotations: str = "auto"
    eyes: bool = True
    motion: bool = True
    start: float = 0.0
    end: float | None = None
    model: Path = DEFAULT_MODEL


def extract(video: Path, out: Path, o: ExtractOptions) -> None:
    """Single decode pass: eye metrics at eye_fps, motion energy at motion_fps."""
    info = probe(video)
    out.mkdir(parents=True, exist_ok=True)
    mot_fps = min(o.motion_fps, info.fps) if o.motion else None
    total = (o.end or info.duration or 0) - o.start

    eye_rows: list[tuple] = []
    mot_rows: list[tuple] = []
    det = EyeDetector(o.model, face_roi=o.face_roi, rotations=o.rotations) if o.eyes else None
    meter = MotionMeter(body_roi=o.body_roi) if o.motion else None

    t0 = time.time()
    last_print = 0.0
    t = o.start
    try:
        for fr in iter_frames(video, info.fps, o.eye_fps if o.eyes else 0, mot_fps, o.start, o.end):
            t = fr.t
            if det is not None and fr.want_eyes:
                r = det.detect(fr.image)
                eye_rows.append((fr.t, r.face, r.ear, r.ear_a, r.ear_b, r.blink, r.rotation))
            if meter is not None and fr.want_motion:
                mot_rows.append((fr.t, meter.update(fr.image)))
            now = time.time()
            if now - last_print > 10:
                last_print = now
                done = fr.t - o.start
                speed = done / max(1e-6, now - t0)
                eta = (total - done) / speed if total > 0 and speed > 0 else float("nan")
                pct = f"{100 * done / total:5.1f}%" if total > 0 else ""
                print(f"  {fmt_time(fr.t)} {pct}  {speed:5.1f}x realtime  eta {fmt_time(eta) if np.isfinite(eta) else '?'}",
                      file=sys.stderr, flush=True)
    finally:
        if det is not None:
            det.close()

    e = np.array(eye_rows, dtype=float).reshape(-1, 7)
    m = np.array(mot_rows, dtype=float).reshape(-1, 2)
    np.savez_compressed(
        out / "signals.npz",
        t_eye=e[:, 0], face=e[:, 1].astype(bool), ear=e[:, 2], ear_a=e[:, 3], ear_b=e[:, 4],
        blink=e[:, 5], rotation=e[:, 6], t_mot=m[:, 0], energy=m[:, 1],
    )
    meta = {
        "video": str(video.resolve()),
        "fps": info.fps,
        "duration": max(t, info.duration if o.end is None else 0.0),
        "analyzed_from": o.start,
        "analyzed_to": t,
        "creation_time": info.creation_time.isoformat() if info.creation_time else None,
        "eye_fps": o.eye_fps if o.eyes else None,
        "motion_fps": mot_fps,
        "face_roi": o.face_roi,
        "body_roi": o.body_roi,
        "processing_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"extracted {len(eye_rows)} eye samples, {len(mot_rows)} motion samples "
          f"in {fmt_time(time.time() - t0)}", file=sys.stderr)


def analyze(
    out: Path,
    p: Params,
    start_clock: dt.datetime | None = None,
    clips: bool = True,
    clip_pad: float = 10.0,
) -> list:
    """Events, CSV, summary and plot from out/signals.npz. Cheap to re-run."""
    meta = json.loads((out / "meta.json").read_text())
    s = np.load(out / "signals.npz")
    t_eye, face, ear = s["t_eye"], s["face"], s["ear"]
    t_mot, energy = s["t_mot"], s["energy"]
    if start_clock is None and meta.get("creation_time"):
        start_clock = dt.datetime.fromisoformat(meta["creation_time"])

    eyes = analyze_eyes(t_eye, ear, face, p) if len(t_eye) > 10 else None
    motion = analyze_motion(t_mot, energy, p) if len(t_mot) > 10 else None
    events = fuse(eyes, motion, t_eye, face, t_mot, energy, meta.get("motion_fps") or 1.0, p)

    df = report.events_frame(events, start_clock)
    df.to_csv(out / "events.csv", index=False)
    txt = report.summary_text(events, meta["video"], meta["duration"], eyes, start_clock)
    (out / "summary.txt").write_text(txt)
    (out / "params.json").write_text(json.dumps(p.to_dict(), indent=2))
    report.timeline_png(out / "timeline.png", events, t_eye, eyes, face, t_mot, motion, meta["duration"])

    if clips and events:
        cdir = out / "clips"
        cdir.mkdir(exist_ok=True)
        for old in cdir.glob("event_*.mp4"):
            old.unlink()
        video = Path(meta["video"])
        for i, e in enumerate(events, 1):
            name = f"event_{i:02d}_{fmt_time(e.start).replace(':', '-')}_{e.kind}.mp4"
            print(f"  clip {i}/{len(events)}: {name}", file=sys.stderr, flush=True)
            if not export_clip(video, cdir / name, e.start, e.end, clip_pad):
                print("  (ffmpeg missing or failed, skipping clips)", file=sys.stderr)
                break

    print(txt)
    return events
