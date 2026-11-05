"""Live mode: analyze an RTSP stream overnight, keep only video around events.

One ffmpeg process per camera connection pulls the stream once and writes
two outputs:

  1. stream-copied 30 s .mkv segments into buffer/conn_NNN/  (rolling buffer)
  2. raw BGR frames at a fixed fps on stdout                   (analysis)

Because the pipe has a constant frame rate, frame n is at stream time n/fps,
which is the same timeline ffmpeg uses in segments.csv, so event times map
exactly onto buffered segments. Segments that are older than the retention
window and overlap no "keep" range are deleted; on a controlled stop the
normal offline analysis runs and clips are cut from what is left.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .events import Params, merge_intervals
from .eyes import DEFAULT_MODEL, EyeDetector
from .motion import MotionMeter
from .video import fmt_time

# ---------------------------------------------------------------- helpers

# Short cap: a probe costs almost nothing, and every second spent waiting
# after the camera is back is a second of the night not analyzed.
MAX_BACKOFF_S = 10.0


def redact(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://***@host/... (never log passwords)."""
    return re.sub(r"(//)[^/@]+@", r"\1***@", url)


def is_rtsp(url: str) -> bool:
    return url.lower().startswith(("rtsp://", "rtsps://"))


# ---------------------------------------------------------------- segment ring


@dataclass
class Conn:
    index: int
    dir: str  # relative to the buffer dir
    offset: float | None = None  # session time of stream time 0


@dataclass
class Segment:
    path: Path
    start: float  # session time
    end: float
    complete: bool


class SegmentRing:
    """Rolling buffer of stream-copied segments, across reconnections."""

    def __init__(self, buffer_dir: Path, retention_s: float = 300.0, segment_s: float = 30.0):
        self.dir = Path(buffer_dir)
        self.retention_s = retention_s
        self.segment_s = segment_s
        self.keep: list[tuple[float, float]] = []
        self.conns: list[Conn] = []
        self._lock = threading.Lock()
        idx = self.dir / "conns.json"
        if idx.exists():
            saved = json.loads(idx.read_text())
            self.conns = [Conn(**c) for c in saved.get("conns", [])]
            self.keep = [tuple(k) for k in saved.get("keep", [])]
            self.segment_s = saved.get("segment_s", segment_s)

    # bookkeeping ------------------------------------------------------------

    def new_conn(self) -> Conn:
        with self._lock:
            c = Conn(len(self.conns), f"conn_{len(self.conns):03d}")
            (self.dir / c.dir).mkdir(parents=True, exist_ok=True)
            self.conns.append(c)
        self.save()
        return c

    def set_offset(self, conn: Conn, offset: float) -> None:
        conn.offset = offset
        self.save()

    def add_keep(self, start: float, end: float) -> None:
        with self._lock:
            self.keep = merge_intervals(self.keep + [(start, end)], 0.0)

    def save(self) -> None:
        with self._lock:
            data = {"conns": [asdict(c) for c in self.conns], "keep": self.keep,
                    "segment_s": self.segment_s}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "conns.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(self.dir / "conns.json")

    # segments -----------------------------------------------------------------

    def segments(self) -> list[Segment]:
        """All segments still on disk, in session time, sorted.

        Finished segments come from ffmpeg's segments.csv. A segment still
        being written (or left behind by a crash) is not listed there; it is
        assumed to start where the last listed one ended.
        """
        out: list[Segment] = []
        for c in self.conns:
            if c.offset is None:
                continue
            d = self.dir / c.dir
            listed: dict[str, tuple[float, float]] = {}
            lst = d / "segments.csv"
            if lst.exists():
                with lst.open(newline="") as f:
                    for row in csv.reader(f):
                        if len(row) >= 3:
                            try:
                                listed[Path(row[0]).name] = (float(row[1]), float(row[2]))
                            except ValueError:
                                pass
            last_end = max((e for _, e in listed.values()), default=0.0)
            for p in sorted(d.glob("*.mkv")):
                if p.name in listed:
                    s, e = listed[p.name]
                    out.append(Segment(p, c.offset + s, c.offset + e, True))
                else:
                    s = c.offset + last_end
                    out.append(Segment(p, s, s + self.segment_s * 2, False))
        out.sort(key=lambda s: s.start)
        return out

    def _kept(self, s: float, e: float) -> bool:
        return any(min(e, ke) > max(s, ks) for ks, ke in self.keep)

    def prune(self, now: float) -> int:
        """Delete finished, unflagged segments older than the retention window."""
        n = 0
        with self._lock:
            keep_snapshot = list(self.keep)
        for seg in self.segments():
            if not seg.complete or seg.end > now - self.retention_s:
                continue
            if any(min(seg.end, ke) > max(seg.start, ks) for ks, ke in keep_snapshot):
                continue
            try:
                seg.path.unlink()
                n += 1
            except FileNotFoundError:
                pass
        return n

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.rglob("*.mkv") if p.exists())

    def covering(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments() if s.end > start and s.start < end]

    def export_clip(self, start: float, end: float, pad: float, dst: Path) -> bool:
        """Cut [start-pad, end+pad] from the buffered segments into an H.264 MP4."""
        a, b = start - pad, end + pad
        segs = self.covering(a, b)
        if not segs:
            return False
        lst = dst.with_suffix(".concat.txt")
        lst.write_text("".join(f"file '{s.path.resolve()}'\n" for s in segs))
        ss = max(0.0, a - segs[0].start)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "fatal",
            "-f", "concat", "-safe", "0", "-ss", f"{ss:.2f}", "-i", str(lst),
            "-t", f"{b - max(a, segs[0].start):.2f}",
            "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(dst),
        ]
        ok = subprocess.run(cmd).returncode == 0
        lst.unlink(missing_ok=True)
        return ok


# ---------------------------------------------------------------- flagger


class LiveFlagger:
    """Liberal online criteria deciding which buffered video to keep.

    These are deliberately more sensitive than the final offline analysis:
    a false keep only costs a little disk, a missed keep loses the clip.
    """

    def __init__(self, p: Params, warmup_s: float = 300.0):
        self.p = p
        self.warmup_s = warmup_s
        self._ears: deque[float] = deque(maxlen=6000)  # ~20 min at 5 fps
        self._ref: float | None = None
        self._n = 0
        self._motion_since: float | None = None
        self._eye_run = 0

    def eye_threshold(self) -> float | None:
        if self.p.close_thr is not None:
            return self.p.close_thr
        if self._ref is None:
            return None
        return self._ref * self.p.auto_close_ratio

    def eye(self, t: float, face: bool, ear: float) -> bool:
        if face and np.isfinite(ear):
            self._ears.append(ear)
            self._n += 1
            if self._n % 50 == 1:
                self._ref = float(np.median(self._ears))
        thr = self.eye_threshold()
        hit = bool(face and np.isfinite(ear) and thr is not None and ear > thr)
        self._eye_run = self._eye_run + 1 if hit else 0
        if t < self.warmup_s:
            return True
        return self._eye_run >= 2  # two samples in a row: ignore single noisy frames

    def motion(self, t: float, energy: float) -> bool:
        if t < self.warmup_s:
            return True
        if np.isfinite(energy) and energy > 0.5 * self.p.motion_min_abs:
            if self._motion_since is None:
                self._motion_since = t
            return t - self._motion_since >= 1.0
        self._motion_since = None
        return False


# ---------------------------------------------------------------- ffmpeg source


class FfmpegSource:
    def __init__(self, url: str, conn_dir: Path, fps: float, segment_s: float,
                 transport: str = "tcp", log: Path | None = None):
        self.url, self.dir, self.fps = url, conn_dir, fps
        self.segment_s, self.transport, self.log = segment_s, transport, log
        self.proc: subprocess.Popen | None = None
        self.w = self.h = 0

    def _input_args(self) -> list[str]:
        if is_rtsp(self.url):
            return ["-rtsp_transport", self.transport, "-timeout", "10000000"]
        return ["-re"]  # files (testing): play at real-time speed

    def probe(self) -> bool:
        cmd = ["ffprobe", "-v", "error", *(_rtsp_probe_args(self.url, self.transport)),
               "-select_streams", "v:0", "-show_entries", "stream=width,height",
               "-of", "json", self.url]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
            st = json.loads(out)["streams"][0]
            self.w, self.h = int(st["width"]), int(st["height"])
            return self.w > 0 and self.h > 0
        except Exception:
            return False

    def start(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
            *self._input_args(), "-i", self.url,
            # output 1: rolling buffer, video stream-copied (no re-encode)
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
            "-f", "segment", "-segment_time", str(self.segment_s), "-reset_timestamps", "1",
            "-strftime", "1", "-segment_list", str(self.dir / "segments.csv"),
            "-segment_list_type", "csv", str(self.dir / "%Y%m%d-%H%M%S.mkv"),
            # output 2: frames for analysis
            "-map", "0:v:0", "-vf", f"fps={self.fps}", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1",
        ]
        err = open(self.log, "ab") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=err, bufsize=0)

    def read(self) -> np.ndarray | None:
        assert self.proc and self.proc.stdout
        n = self.w * self.h * 3
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                return None
            got += k
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def stop(self, timeout: float = 15.0) -> None:
        """Ask ffmpeg to finish cleanly ('q') so the last segment is closed."""
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write(b"q")
            p.stdin.flush()
            p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            p.kill()

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def wait(self) -> None:
        if self.proc:
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def _rtsp_probe_args(url: str, transport: str) -> list[str]:
    return ["-rtsp_transport", transport, "-timeout", "10000000"] if is_rtsp(url) else []


# ---------------------------------------------------------------- latest-frame slot


class LatestSlot:
    """Single-item mailbox: put() overwrites, get() waits. Never blocks the reader."""

    def __init__(self):
        self._item = None
        self._cv = threading.Condition()

    def put(self, item) -> None:
        with self._cv:
            self._item = item
            self._cv.notify()

    def get(self, timeout: float):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


# ---------------------------------------------------------------- session


@dataclass
class LiveOptions:
    url: str
    out: Path
    until: dt.datetime | None = None
    eye_fps: float = 5.0
    motion_fps: float = 15.0
    face_roi: tuple[int, int, int, int] | None = None
    body_roi: tuple[int, int, int, int] | None = None
    rotations: str = "auto"
    segment_s: float = 30.0
    retention_s: float = 300.0
    keep_pad_s: float = 30.0
    warmup_s: float = 300.0
    transport: str = "tcp"
    model: Path = DEFAULT_MODEL
    status_every_s: float = 60.0
    checkpoint_every_s: float = 300.0
    stall_s: float = 20.0
    clip_pad: float = 10.0
    params: Params = field(default_factory=Params)


class LiveSession:
    def __init__(self, o: LiveOptions):
        self.o = o
        self.out = o.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.ring = SegmentRing(self.out / "buffer", o.retention_s, o.segment_s)
        self.flagger = LiveFlagger(o.params, o.warmup_s)
        self.eye_rows: list[tuple] = []
        self.mot_rows: list[tuple] = []
        self.gaps: list[tuple[float, float]] = []
        self.reconnects = 0
        self.stop_event = threading.Event()
        self.slot = LatestSlot()
        self.src: FfmpegSource | None = None
        self.t0 = time.monotonic()
        self.session_start = dt.datetime.now()
        self.last_frame_mono = time.monotonic()
        self.last_eye_t = 0.0
        self.last_t = 0.0
        self.src_done = False

    def now_t(self) -> float:
        return time.monotonic() - self.t0

    def log(self, msg: str) -> None:
        print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)

    # signals -----------------------------------------------------------------

    def request_stop(self, reason: str) -> None:
        if self.stop_event.is_set():
            return
        self.log(f"stopping ({reason}) - finishing recording and building the report...")
        self.stop_event.set()
        src = self.src
        if src is not None:
            threading.Thread(target=src.stop, daemon=True).start()

    def _on_signal(self, signum, _frame) -> None:
        self.request_stop("Ctrl+C" if signum == signal.SIGINT else "SIGTERM")

    # threads -----------------------------------------------------------------

    def _eye_worker(self, det: EyeDetector) -> None:
        while not (self.stop_event.is_set() and self.src_done):
            item = self.slot.get(0.5)
            if item is None:
                continue
            t, frame = item
            r = det.detect(frame)
            self.eye_rows.append((t, r.face, r.ear, r.ear_a, r.ear_b, r.blink, r.rotation))
            self.last_eye_t = t
            if self.flagger.eye(t, r.face, r.ear):
                self.ring.add_keep(t - self.o.keep_pad_s, t + self.o.keep_pad_s)

    def _monitor(self) -> None:
        last_status = last_prune = last_ckpt = time.monotonic()
        status_from = 0
        while not self.stop_event.wait(1.0):
            now = time.monotonic()
            if self.o.until and dt.datetime.now() >= self.o.until:
                self.request_stop(f"reached --until {self.o.until:%H:%M}")
                break
            if self.src is not None and now - self.last_frame_mono > self.o.stall_s:
                self.log(f"no frames for {self.o.stall_s:.0f} s - reconnecting")
                self.last_frame_mono = now
                self.src.kill()
            if now - last_prune >= 30:
                last_prune = now
                self.ring.prune(self.now_t())
                self.ring.save()
            if now - last_status >= self.o.status_every_s:
                last_status = now
                rows = self.eye_rows[status_from:]
                status_from = len(self.eye_rows)
                face = 100 * np.mean([r[1] for r in rows]) if rows else 0.0
                kept = sum(e - s for s, e in self.ring.keep)
                lag = max(0.0, self.last_t - self.last_eye_t)
                self.log(f"up {fmt_time(self.now_t())} | face {face:3.0f}% | kept {fmt_time(kept)} "
                         f"| buffer {self.ring.size_bytes() / 1e6:5.0f} MB | eye lag {lag:4.1f} s "
                         f"| reconnects {self.reconnects}")
            if now - last_ckpt >= self.o.checkpoint_every_s:
                last_ckpt = now
                self.checkpoint("running")

    # main loop ---------------------------------------------------------------

    def run(self) -> None:
        o = self.o
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        if shutil.which("caffeinate"):  # keep the Mac from idle-sleeping overnight
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

        self.log(f"live: {redact(o.url)} -> {self.out}")
        if o.until:
            self.log(f"will stop automatically at {o.until:%Y-%m-%d %H:%M}; Ctrl+C stops earlier")
        else:
            self.log("press Ctrl+C to stop and build the report")
        self.ring.add_keep(0.0, o.warmup_s)  # keep everything until baselines exist
        self.src_done = False

        det = EyeDetector(o.model, face_roi=o.face_roi, rotations=o.rotations)
        worker = threading.Thread(target=self._eye_worker, args=(det,), daemon=True)
        monitor = threading.Thread(target=self._monitor, daemon=True)
        worker.start()
        monitor.start()
        try:
            self._supervise()
        finally:
            self.src_done = True
            self.stop_event.set()
            worker.join(timeout=30)
            det.close()
        self.finalize()

    def _supervise(self) -> None:
        o = self.o
        backoff = 2.0
        gap_start: float | None = None
        while not self.stop_event.is_set():
            src = FfmpegSource(o.url, Path(), o.motion_fps, o.segment_s,
                               o.transport, log=self.out / "ffmpeg.log")
            if not src.probe():
                self.log(f"camera not reachable ({redact(o.url)}), retrying in {backoff:.0f} s")
                if gap_start is None:
                    gap_start = self.last_t
                self.stop_event.wait(backoff)
                backoff = min(MAX_BACKOFF_S, backoff * 2)
                continue
            conn = self.ring.new_conn()
            src.dir = self.ring.dir / conn.dir
            src.start()
            self.src = src
            self.last_frame_mono = time.monotonic()
            meter = MotionMeter(body_roi=o.body_roi)
            eye_period = 1.0 / o.eye_fps
            next_eye = 0.0
            n = 0
            while True:
                frame = src.read()
                if frame is None:
                    break
                if n == 0:
                    offset = self.now_t()
                    self.ring.set_offset(conn, offset)
                    if gap_start is not None:
                        self.gaps.append((gap_start, offset))
                        self.log(f"stream back after {offset - gap_start:.0f} s offline")
                        gap_start = None
                    self.log(f"receiving {src.w}x{src.h} (connection {conn.index})")
                t = conn.offset + n / o.motion_fps
                n += 1
                self.last_t = t
                self.last_frame_mono = time.monotonic()
                e = meter.update(frame)
                self.mot_rows.append((t, e))
                if self.flagger.motion(t, e):
                    self.ring.add_keep(t - o.keep_pad_s, t + o.keep_pad_s)
                if t >= next_eye:
                    next_eye = t + eye_period
                    self.slot.put((t, frame))
            src.wait()
            self.src = None
            if self.stop_event.is_set():
                break
            self.reconnects += 1
            gap_start = self.last_t
            self.log(f"stream ended unexpectedly (see ffmpeg.log), reconnecting in {backoff:.0f} s")
            self.stop_event.wait(backoff)
            backoff = 2.0 if n > o.motion_fps * 60 else min(MAX_BACKOFF_S, backoff * 2)
        if gap_start is not None and self.stop_event.is_set():
            self.gaps.append((gap_start, self.now_t()))

    # output ------------------------------------------------------------------

    def meta(self, status: str) -> dict:
        o = self.o
        return {
            "source": "live",
            "status": status,
            "video": redact(o.url),
            "session_start": self.session_start.isoformat(timespec="seconds"),
            "creation_time": self.session_start.isoformat(timespec="seconds"),
            "duration": self.last_t,
            "fps": o.motion_fps,
            "eye_fps": o.eye_fps,
            "motion_fps": o.motion_fps,
            "face_roi": o.face_roi,
            "body_roi": o.body_roi,
            "gaps": self.gaps,
            "reconnects": self.reconnects,
            "segment_s": o.segment_s,
        }

    def checkpoint(self, status: str) -> None:
        from .pipeline import save_signals

        save_signals(self.out, list(self.eye_rows), list(self.mot_rows), self.meta(status))
        self.ring.save()

    def finalize(self) -> None:
        from .pipeline import analyze

        # From here a second Ctrl+C aborts clip cutting but keeps all data.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        self.checkpoint("complete")
        try:
            # Cut clips before deleting anything: segments from the last few
            # minutes (inside the retention window) are still available too.
            missing = analyze(self.out, self.o.params, self.session_start,
                              clips=True, clip_pad=self.o.clip_pad).missing_clips
        except KeyboardInterrupt:
            self.log("aborted. Signals and buffered video are kept; finish later with:\n"
                     f"  eyesopenscan analyze {self.out}")
            return
        if missing:
            self.log(f"{len(missing)} event(s) had no buffered video; buffer kept for inspection")
        else:
            shutil.rmtree(self.ring.dir, ignore_errors=True)
        self.log(f"done. Results in {self.out}/")
