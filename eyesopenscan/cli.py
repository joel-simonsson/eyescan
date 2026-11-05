"""Command line interface.

  eyesopenscan scan VIDEO [-o OUT]            full night -> events.csv, summary, plot, clips
  eyesopenscan analyze OUT [threshold opts]   re-run event detection on saved signals
  eyesopenscan calibrate VIDEO --closed A-B --open C-D
  eyesopenscan debug VIDEO --at T --duration S   overlay video for tuning
  eyesopenscan pick-roi VIDEO --at T          drag a box to get x,y,w,h
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from .events import Params
from .eyes import DEFAULT_MODEL, EyeDetector
from .pipeline import ExtractOptions, analyze, extract
from .video import fmt_time, iter_frames, parse_range, parse_roi, parse_time, probe, read_frame_at


def _add_threshold_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("event detection")
    g.add_argument("--calibration", type=Path, help="JSON from `calibrate` (sets EAR thresholds)")
    g.add_argument("--open-thr", type=float, help="EAR above which eyes count as open")
    g.add_argument("--close-thr", type=float, help="EAR below which eyes count as closed again")
    g.add_argument("--eye-min-dur", type=float, default=Params.eye_min_dur_s, help="seconds (default %(default)s)")
    g.add_argument("--motion-k", type=float, default=Params.motion_k, help="motion sensitivity, lower = more sensitive (default %(default)s)")
    g.add_argument("--motion-min", type=float, default=Params.motion_min_abs, help="min fraction of ROI pixels moving (default %(default)s)")
    g.add_argument("--motion-min-dur", type=float, default=Params.motion_min_dur_s, help="seconds (default %(default)s)")
    g.add_argument("--merge-gap", type=float, default=Params.merge_gap_s, help="join events closer than this, seconds (default %(default)s)")
    g.add_argument("--start-clock", help="wall-clock time the recording started, e.g. '2026-11-05 22:30:00'")
    g.add_argument("--no-clips", action="store_true", help="don't export per-event video clips")
    g.add_argument("--clip-pad", type=float, default=10.0, help="seconds before/after each event in clips")


def _params(a) -> Params:
    p = Params(
        eye_min_dur_s=a.eye_min_dur, motion_k=a.motion_k, motion_min_abs=a.motion_min,
        motion_min_dur_s=a.motion_min_dur, merge_gap_s=a.merge_gap,
    )
    if a.calibration:
        cal = json.loads(Path(a.calibration).read_text())
        p.open_thr, p.close_thr = cal["open_thr"], cal["close_thr"]
    if a.open_thr is not None:
        p.open_thr = a.open_thr
        if p.close_thr is None:
            p.close_thr = a.open_thr * 0.85
    if a.close_thr is not None:
        p.close_thr = a.close_thr
    return p


def _clock(a) -> dt.datetime | None:
    if not getattr(a, "start_clock", None):
        return None
    return dt.datetime.fromisoformat(a.start_clock)


# ---------------------------------------------------------------- commands


def cmd_scan(a) -> None:
    out = a.out or Path("out") / a.video.stem
    o = ExtractOptions(
        eye_fps=a.eye_fps, motion_fps=a.motion_fps, face_roi=parse_roi(a.face_roi),
        body_roi=parse_roi(a.body_roi), rotations=a.rotations, eyes=not a.no_eyes,
        motion=not a.no_motion, start=parse_time(a.start) if a.start else 0.0,
        end=parse_time(a.end) if a.end else None, model=a.model,
    )
    print(f"scanning {a.video} -> {out}", file=sys.stderr)
    extract(a.video, out, o)
    analyze(out, _params(a), _clock(a), clips=not a.no_clips, clip_pad=a.clip_pad)
    print(f"results in {out}/  (events.csv, summary.txt, timeline.png, clips/)")


def cmd_analyze(a) -> None:
    analyze(a.out, _params(a), _clock(a), clips=not a.no_clips, clip_pad=a.clip_pad)


def _sample_ear(video: Path, rng: tuple[float, float], det: EyeDetector, fps: float) -> np.ndarray:
    info = probe(video)
    vals = []
    for fr in iter_frames(video, info.fps, fps, None, rng[0], rng[1]):
        if fr.want_eyes:
            r = det.detect(fr.image)
            if r.face and np.isfinite(r.ear):
                vals.append(r.ear)
    return np.array(vals)


def cmd_calibrate(a) -> None:
    with EyeDetector(a.model, face_roi=parse_roi(a.face_roi), rotations=a.rotations) as det:
        closed = np.concatenate([_sample_ear(a.video, parse_range(r), det, 5) for r in a.closed])
        opened = np.concatenate([_sample_ear(a.video, parse_range(r), det, 5) for r in a.open])
    if len(closed) < 5 or len(opened) < 5:
        sys.exit(f"not enough frames with a detected face (closed={len(closed)}, open={len(opened)}). "
                 "Try --face-roi, or pick segments where the face is clearly visible.")
    c_med, o_med = float(np.median(closed)), float(np.median(opened))
    c_hi, o_lo = float(np.percentile(closed, 95)), float(np.percentile(opened, 5))
    span = o_med - c_med
    cal = {
        "closed_median": round(c_med, 4), "open_median": round(o_med, 4),
        "closed_p95": round(c_hi, 4), "open_p5": round(o_lo, 4),
        "open_thr": round(c_med + 0.55 * span, 4),
        "close_thr": round(c_med + 0.35 * span, 4),
        "n_closed": int(len(closed)), "n_open": int(len(opened)),
    }
    a.out.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    if span <= 0:
        print("WARNING: open EAR is not higher than closed EAR - check the segments.", file=sys.stderr)
    elif o_lo <= c_hi:
        print("WARNING: open/closed EAR ranges overlap - expect misses/false alarms. "
              "Improve camera angle/lighting or use --face-roi.", file=sys.stderr)
    print(f"saved {a.out}; use with: eyesopenscan scan VIDEO --calibration {a.out}")


def cmd_debug(a) -> None:
    import cv2

    info = probe(a.video)
    start = parse_time(a.at)
    end = start + a.duration
    face_roi = parse_roi(a.face_roi)
    thr = None
    if a.calibration:
        cal = json.loads(a.calibration.read_text())
        thr = (cal["open_thr"], cal["close_thr"])
    writer = None
    n = 0
    with EyeDetector(a.model, face_roi=face_roi, rotations=a.rotations, keep_images=True) as det:
        for fr in iter_frames(a.video, info.fps, a.fps, None, start, end):
            if not fr.want_eyes:
                continue
            r = det.detect(fr.image)
            img = cv2.cvtColor(r.prepared, cv2.COLOR_RGB2BGR)
            scale = 720 / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale)
            if writer is None:
                writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (img.shape[1], img.shape[0]))
                size = (img.shape[1], img.shape[0])
            if (img.shape[1], img.shape[0]) != size:  # rotation changed shape
                img = cv2.resize(img, size)
            if r.face and r.eye_px is not None:
                sx = size[0] / (r.prepared.shape[1])
                sy = size[1] / (r.prepared.shape[0])
                for x, y in r.eye_px:
                    cv2.circle(img, (int(x * sx), int(y * sy)), 2, (0, 255, 0), -1)
                state = ""
                if thr:
                    state = "OPEN" if r.ear > thr[0] else ("closed" if r.ear < thr[1] else "?")
                txt = f"{fmt_time(fr.t)}  EAR {r.ear:.3f}  blink {r.blink:.2f}  rot {r.rotation}  {state}"
                color = (0, 0, 255) if state == "OPEN" else (0, 255, 0)
            else:
                txt, color = f"{fmt_time(fr.t)}  no face", (0, 165, 255)
            cv2.rectangle(img, (0, 0), (size[0], 34), (0, 0, 0), -1)
            cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            writer.write(img)
            n += 1
            if r.face:
                print(f"{fr.t:9.2f}s  EAR {r.ear:.3f}  blink {r.blink:.2f}  rot {r.rotation}")
            else:
                print(f"{fr.t:9.2f}s  no face")
    if writer:
        writer.release()
    print(f"wrote {n} frames to {a.out}", file=sys.stderr)


def cmd_pick_roi(a) -> None:
    import cv2

    img = read_frame_at(a.video, parse_time(a.at))
    snap = a.video.with_suffix(".roi.jpg")
    cv2.imwrite(str(snap), img)
    print(f"saved reference frame {snap} ({img.shape[1]}x{img.shape[0]})", file=sys.stderr)
    disp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    x, y, w, h = cv2.selectROI("drag a box, then ENTER (c = cancel)", disp, showCrosshair=True)
    cv2.destroyAllWindows()
    if w and h:
        print(f"{x},{y},{w},{h}")


# ---------------------------------------------------------------- argparse


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eyesopenscan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, roi=True):
        sp.add_argument("video", type=Path)
        sp.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        sp.add_argument("--rotations", default="auto", help="auto | 0 | 90 | 180 | 270 (head orientation in image)")
        if roi:
            sp.add_argument("--face-roi", help="x,y,w,h crop around the pillow/face (see pick-roi)")

    s = sub.add_parser("scan", help="analyze a full recording")
    common(s)
    s.add_argument("-o", "--out", type=Path, help="output dir (default out/<video name>)")
    s.add_argument("--body-roi", help="x,y,w,h region for motion detection (default whole frame)")
    s.add_argument("--eye-fps", type=float, default=5.0)
    s.add_argument("--motion-fps", type=float, default=15.0)
    s.add_argument("--no-eyes", action="store_true")
    s.add_argument("--no-motion", action="store_true")
    s.add_argument("--start", help="only analyze from this time (e.g. 1:30:00)")
    s.add_argument("--end", help="only analyze up to this time")
    _add_threshold_args(s)
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("analyze", help="re-run event detection on a previous scan's signals")
    s.add_argument("out", type=Path)
    _add_threshold_args(s)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("calibrate", help="derive EAR thresholds from known closed/open segments")
    common(s)
    s.add_argument("--closed", action="append", required=True, help="START-END with eyes closed (repeatable)")
    s.add_argument("--open", action="append", required=True, help="START-END with eyes open (repeatable)")
    s.add_argument("-o", "--out", type=Path, default=Path("calibration.json"))
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("debug", help="write an overlay video showing what the eye detector sees")
    common(s)
    s.add_argument("--at", default="0", help="start time")
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--calibration", type=Path)
    s.add_argument("-o", "--out", type=Path, default=Path("debug.mp4"))
    s.set_defaults(fn=cmd_debug)

    s = sub.add_parser("pick-roi", help="drag a box on a frame and print x,y,w,h")
    s.add_argument("video", type=Path)
    s.add_argument("--at", default="0")
    s.set_defaults(fn=cmd_pick_roi)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
