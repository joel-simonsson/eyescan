"""Extraction pass over the video, and analysis of the saved signals."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from dataclasses import dataclass, field
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

    save_signals(out, eye_rows, mot_rows, {
        "source": "file",
        "status": "complete",
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
    })
    print(f"extracted {len(eye_rows)} eye samples, {len(mot_rows)} motion samples "
          f"in {fmt_time(time.time() - t0)}", file=sys.stderr)


def save_signals(out: Path, eye_rows: list, mot_rows: list, meta: dict) -> None:
    """Write signals.npz + meta.json atomically (also used for live checkpoints)."""
    out.mkdir(parents=True, exist_ok=True)
    e = np.array(eye_rows, dtype=float).reshape(-1, 7)
    m = np.array(mot_rows, dtype=float).reshape(-1, 2)
    tmp = out / "signals.tmp.npz"
    np.savez_compressed(
        tmp,
        t_eye=e[:, 0], face=e[:, 1].astype(bool), ear=e[:, 2], ear_a=e[:, 3], ear_b=e[:, 4],
        blink=e[:, 5], rotation=e[:, 6], t_mot=m[:, 0], energy=m[:, 1],
    )
    tmp.replace(out / "signals.npz")
    tmpm = out / "meta.tmp.json"
    tmpm.write_text(json.dumps(meta, indent=2))
    tmpm.replace(out / "meta.json")


@dataclass
class AnalyzeResult:
    events: list
    missing_clips: list[int] = field(default_factory=list)


def _clip_exporter(out: Path, meta: dict):
    """Returns export(start, end, pad, dst) -> bool, or None if no video exists."""
    if meta.get("source") == "live":
        from .live import SegmentRing

        buf = out / "buffer"
        if not (buf / "conns.json").exists():
            return None
        ring = SegmentRing(buf)
        return ring.export_clip
    video = Path(meta["video"])
    if not video.exists():
        return None
    return lambda s, e, pad, dst: export_clip(video, dst, s, e, pad)


def analyze(
    out: Path,
    p: Params,
    start_clock: dt.datetime | None = None,
    clips: bool = True,
    clip_pad: float = 10.0,
) -> AnalyzeResult:
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
    (out / "params.json").write_text(json.dumps(p.to_dict(), indent=2))
    report.timeline_png(out / "timeline.png", events, t_eye, eyes, face, t_mot, motion,
                        meta["duration"], gaps=meta.get("gaps") or [])

    missing: list[int] = []
    clip_note = ""
    if clips and events:
        exporter = _clip_exporter(out, meta)
        if exporter is None:
            clip_note = ("source video/buffer no longer exists - existing clips/ left untouched"
                         if (out / "clips").exists() else "source video/buffer not found - no clips")
            print(f"  ({clip_note})", file=sys.stderr)
        else:
            cdir = out / "clips"
            cdir.mkdir(exist_ok=True)
            for old in cdir.glob("event_*.mp4"):
                old.unlink()
            for i, e in enumerate(events, 1):
                name = f"event_{i:02d}_{fmt_time(e.start).replace(':', '-')}_{e.kind}.mp4"
                if start_clock is not None:
                    clock = (start_clock + dt.timedelta(seconds=e.start)).strftime("%H-%M-%S")
                    name = f"event_{i:02d}_{clock}_{e.kind}.mp4"
                print(f"  clip {i}/{len(events)}: {name}", file=sys.stderr, flush=True)
                if not exporter(e.start, e.end, clip_pad, cdir / name):
                    missing.append(i)

    txt = report.summary_text(events, meta["video"], meta["duration"], eyes, start_clock,
                              gaps=meta.get("gaps") or [], missing_clips=missing,
                              reconnects=meta.get("reconnects", 0), note=clip_note,
                              status=meta.get("status", "complete"))
    (out / "summary.txt").write_text(txt)
    print(txt)
    return AnalyzeResult(events, missing)
