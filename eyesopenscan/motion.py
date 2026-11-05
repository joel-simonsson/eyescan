"""Body-motion energy from frame differencing, plus rhythmicity analysis."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import signal as sps


class MotionMeter:
    """Fraction of pixels in the body ROI that changed since the previous sample.

    Works on a downscaled, blurred grayscale copy so IR sensor noise does not
    register as movement. Frames where nearly the whole picture changes at once
    (IR/day-mode switch, auto exposure jump) are returned as NaN.
    """

    def __init__(
        self,
        body_roi: tuple[int, int, int, int] | None = None,
        width: int = 320,
        diff_thresh: int = 12,
        global_change_frac: float = 0.85,
    ):
        self.body_roi = body_roi
        self.width = width
        self.diff_thresh = diff_thresh
        self.global_change_frac = global_change_frac
        self._prev: np.ndarray | None = None
        self._roi_small: tuple[slice, slice] | None = None

    def update(self, bgr: np.ndarray) -> float:
        H, W = bgr.shape[:2]
        scale = self.width / W
        small = cv2.resize(bgr, (self.width, max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._roi_small is None:
            if self.body_roi:
                x, y, w, h = self.body_roi
                self._roi_small = (
                    slice(max(0, int(y * scale)), max(1, int((y + h) * scale))),
                    slice(max(0, int(x * scale)), max(1, int((x + w) * scale))),
                )
            else:
                self._roi_small = (slice(None), slice(None))

        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return float("nan")

        changed = cv2.absdiff(gray, prev) > self.diff_thresh
        if changed.mean() > self.global_change_frac:
            return float("nan")
        return float(changed[self._roi_small].mean())


def rhythmicity(energy: np.ndarray, fps: float, band: tuple[float, float] = (2.0, 8.0)) -> float:
    """Share of motion-signal power inside `band` Hz (0..1).

    Clonic seizure jerking tends to be rhythmic at roughly 2-8 Hz, whereas
    turning over is a slow, one-off change. Returns NaN when the sample rate
    is too low to see the band or the segment is too short.
    """
    x = np.asarray(energy, dtype=float)
    x = x[np.isfinite(x)]
    nyq = fps / 2.0
    if fps < 2 * band[0] * 1.5 or len(x) < int(fps * 2):
        return float("nan")
    x = sps.detrend(x)
    if np.allclose(x, 0):
        return 0.0
    nper = min(len(x), int(fps * 4))
    f, p = sps.welch(x, fs=fps, nperseg=nper)
    keep = f >= 0.3  # ignore DC / very slow drift
    total = p[keep].sum()
    if total <= 0:
        return 0.0
    hi = min(band[1], nyq * 0.95)
    inband = p[(f >= band[0]) & (f <= hi)].sum()
    return float(inband / total)
