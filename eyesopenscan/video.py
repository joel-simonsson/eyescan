"""Video decoding, sampling, time helpers and clip export."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# ---------------------------------------------------------------- time helpers


def parse_time(s: str) -> float:
    """Parse '90', '90.5', '01:30', '1:02:03' into seconds."""
    parts = [float(p) for p in s.strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def parse_range(s: str) -> tuple[float, float]:
    """Parse 'START-END' (each in parse_time format) into seconds."""
    a, b = s.split("-", 1)
    start, end = parse_time(a), parse_time(b)
    if end <= start:
        raise ValueError(f"range end must be after start: {s!r}")
    return start, end


def parse_roi(s: str | None) -> tuple[int, int, int, int] | None:
    """Parse 'x,y,w,h' into a tuple of ints."""
    if not s:
        return None
    vals = [int(round(float(v))) for v in s.split(",")]
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise ValueError(f"ROI must be x,y,w,h with positive w,h: {s!r}")
    return tuple(vals)  # type: ignore[return-value]


def fmt_time(secs: float) -> str:
    """Seconds -> 'H:MM:SS'."""
    secs = max(0.0, float(secs))
    h, rem = divmod(int(round(secs)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def fmt_clock(start: dt.datetime | None, secs: float) -> str:
    if start is None:
        return ""
    return (start + dt.timedelta(seconds=float(secs))).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- probing


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    creation_time: dt.datetime | None


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (1.0 <= fps <= 240.0):
        fps = 15.0  # some IR cams write garbage fps; timestamps from POS_MSEC still apply
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n / fps if n > 0 else 0.0
    return VideoInfo(path, fps, n, w, h, duration, _creation_time(path))


def _creation_time(path: Path) -> dt.datetime | None:
    """Best effort: read the container's creation_time tag (local time)."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout
        tag = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
        if not tag:
            return None
        t = dt.datetime.fromisoformat(tag.replace("Z", "+00:00"))
        if t.year < 2005:  # unset camera clocks
            return None
        return t.astimezone().replace(tzinfo=None) if t.tzinfo else t
    except Exception:
        return None


# ---------------------------------------------------------------- frame iteration


@dataclass
class Frame:
    index: int
    t: float  # seconds from start of file
    image: np.ndarray | None  # BGR, None if the frame was only grabbed (not wanted)
    want_eyes: bool
    want_motion: bool


def iter_frames(
    path: Path,
    fps: float,
    eye_fps: float,
    motion_fps: float | None,
    start: float = 0.0,
    end: float | None = None,
) -> Iterator[Frame]:
    """Decode the video, yielding frames flagged for eye and/or motion analysis.

    Frames that neither stage needs are grabbed (decoded) but not converted,
    which is the cheapest way to skip in OpenCV while keeping timing exact.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    eye_period = 1.0 / eye_fps if eye_fps > 0 else float("inf")
    mot_period = 1.0 / motion_fps if motion_fps else float("inf")
    next_eye = start
    next_mot = start
    idx = 0
    last_t = -1.0
    try:
        while True:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = pos if pos > 0 else start + idx / fps
            if t <= last_t:  # keep strictly monotonic even with odd containers
                t = last_t + 1.0 / fps
            last_t = t
            idx += 1
            if end is not None and t >= end:
                break
            want_eyes = t + 1e-6 >= next_eye
            want_motion = t + 1e-6 >= next_mot
            image = None
            if want_eyes or want_motion:
                ok, image = cap.retrieve()
                if not ok:
                    continue
            if want_eyes:
                next_eye += eye_period * max(1, int((t - next_eye) // eye_period) + 1)
            if want_motion:
                next_mot += mot_period * max(1, int((t - next_mot) // mot_period) + 1)
            yield Frame(idx - 1, t, image, want_eyes, want_motion)
    finally:
        cap.release()


def read_frame_at(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {t:.1f}s")
    return img


# ---------------------------------------------------------------- clips


def export_clip(src: Path, dst: Path, start: float, end: float, pad: float = 10.0) -> bool:
    """Cut [start-pad, end+pad] into a small H.264 MP4 for quick review."""
    if not shutil.which("ffmpeg"):
        return False
    s = max(0.0, start - pad)
    d = (end + pad) - s
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{s:.2f}", "-i", str(src), "-t", f"{d:.2f}",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart", str(dst),
    ]
    return subprocess.run(cmd).returncode == 0
