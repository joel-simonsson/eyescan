"""Per-frame eye openness from MediaPipe Face Landmarker on IR frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# MediaPipe 478-point mesh, ordered p1..p6 for the classic EAR formula:
# p1/p4 = eye corners, (p2,p6) and (p3,p5) = upper/lower lid pairs.
EYE_A = [362, 385, 387, 263, 373, 380]
EYE_B = [33, 160, 158, 133, 153, 144]

# A sleeping head is often lying on its side; the face detector copes badly
# with faces rotated more than ~45 deg, so we try rotated copies of the frame.
ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
}


Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def _clip(b: Box, W: int, H: int) -> Box:
    x0, y0, x1, y1 = b
    x0, y0 = max(0, min(W - 2, x0)), max(0, min(H - 2, y0))
    return (x0, y0, max(x0 + 2, min(W, x1)), max(y0 + 2, min(H, y1)))


def _steps(lo: int, hi: int, stride: int) -> list[int]:
    """Positions lo..hi (inclusive) with roughly `stride` spacing."""
    if hi <= lo:
        return [lo]
    n = max(1, round((hi - lo) / max(1, stride)))
    return sorted({lo + round(i * (hi - lo) / n) for i in range(n + 1)})


def _unrotate(pts: np.ndarray, rot: int, w: int, h: int) -> np.ndarray:
    """Map points from a rotated image (size w x h) back to the unrotated one."""
    x, y = pts[:, 0], pts[:, 1]
    if rot == 90:  # image was rotated clockwise
        return np.stack([y, w - 1 - x], 1)
    if rot == 270:  # counter-clockwise
        return np.stack([h - 1 - y, x], 1)
    if rot == 180:
        return np.stack([w - 1 - x, h - 1 - y], 1)
    return pts


def eye_aspect_ratio(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return float("nan")
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


@dataclass
class EyeResult:
    face: bool
    ear: float = float("nan")  # mean of both eyes
    ear_a: float = float("nan")
    ear_b: float = float("nan")
    blink: float = float("nan")  # mean MediaPipe eyeBlink blendshape (1 = closed)
    rotation: int = -1
    eye_px: np.ndarray | None = None  # (12,2) eye landmarks in the prepared image
    prepared: np.ndarray | None = None  # the image actually fed to the model (RGB)


class EyeDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        face_roi: tuple[int, int, int, int] | None = None,
        rotations: str = "auto",
        clahe: bool = True,
        min_side: int = 480,
        keep_images: bool = False,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}\nDownload it with:\n"
                f"  curl -L -o {model_path} {MODEL_URL}"
            )
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.face_roi = face_roi
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        self.min_side = min_side
        self.keep_images = keep_images
        if rotations == "auto":
            self._order = [0, 90, 270, 180]
            self._auto = True
        else:
            self._order = [int(rotations) % 360]
            self._auto = False
        self._misses = 0
        self._track: Box | None = None

    def close(self) -> None:
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -------------------------------------------------------------------------

    def _enhance(self, bgr: np.ndarray, region: Box) -> tuple[np.ndarray, float]:
        """Crop `region`, boost IR contrast, upscale small crops. Returns (rgb, scale)."""
        x0, y0, x1, y1 = region
        img = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        f = 1.0
        short = min(gray.shape[:2])
        if short < self.min_side:
            f = self.min_side / short
            gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), f

    def _base_region(self, H: int, W: int) -> Box:
        if self.face_roi:
            x, y, w, h = self.face_roi
            return _clip((x, y, x + w, y + h), W, H)
        return (0, 0, W, H)

    def _regions(self, H: int, W: int, search: bool) -> list[Box]:
        """Where to look, cheapest/most likely first.

        The face detector squashes its input into a small *square*, so a face
        in a wide 16:9 bedroom shot ends up too small to find. We therefore (1)
        follow a square crop around the last known face, and (2) when
        searching, try overlapping square tiles at two scales, centre first.
        """
        base = self._base_region(H, W)
        out: list[Box] = []
        if self._track is not None:
            out.append(self._track)
        bx0, by0, bx1, by1 = base
        bw, bh = bx1 - bx0, by1 - by0
        if max(bw, bh) / min(bw, bh) < 1.2 or not search:
            out.append(base)
        if search:
            ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2
            for frac in (1.0, 0.6):
                side = int(min(bw, bh) * frac)
                tiles = [
                    (x, y, x + side, y + side)
                    for x in _steps(bx0, bx1 - side, side // 2)
                    for y in _steps(by0, by1 - side, side // 2)
                ]
                tiles.sort(key=lambda b: (b[0] + side / 2 - ccx) ** 2 + (b[1] + side / 2 - ccy) ** 2)
                out.extend(t for t in tiles if t != base)
        return out

    def _should_search(self) -> bool:
        # Full search (all rotations x all tiles, ~1 s CPU) is expensive. Do it
        # every 5th sample right after losing the face, then every 25th once
        # the face has been gone a while (e.g. under a blanket for hours).
        every = 5 if self._misses < 50 else 25
        return self._misses % every == 0

    def detect(self, bgr: np.ndarray) -> EyeResult:
        H, W = bgr.shape[:2]
        search = self._should_search()
        rots = self._order if (self._auto and search) else self._order[:1]
        first_rgb = None

        for region in self._regions(H, W, search):
            rgb, scale = self._enhance(bgr, region)
            if first_rgb is None:
                first_rgb = rgb
            for rot in rots:
                code = ROTATIONS[rot]
                img = np.ascontiguousarray(rgb if code is None else cv2.rotate(rgb, code))
                res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
                if not res.face_landmarks:
                    continue
                if self._auto and self._order[0] != rot:
                    self._order.remove(rot)
                    self._order.insert(0, rot)
                self._misses = 0
                self._update_track(res, img, rot, scale, region, W, H)
                return self._measure(res, img, rot)

        self._misses += 1
        if self._misses > 25:
            self._track = None
        return EyeResult(face=False, prepared=first_rgb if self.keep_images else None)

    def _update_track(self, res, img, rot, scale, region, W, H) -> None:
        """Remember a square crop (in full-frame pixels) around the face."""
        h, w = img.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        pts = _unrotate(pts, rot, w, h) / scale + np.array(region[:2])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(200.0, 2.2 * max(x1 - x0, y1 - y0))
        self._track = _clip((int(cx - side / 2), int(cy - side / 2),
                             int(cx + side / 2), int(cy + side / 2)), W, H)

    def _measure(self, res, img: np.ndarray, rot: int) -> EyeResult:
        h, w = img.shape[:2]
        lms = res.face_landmarks[0]
        pts = np.array([[lms[i].x * w, lms[i].y * h] for i in EYE_A + EYE_B])
        ea = eye_aspect_ratio(pts[:6])
        eb = eye_aspect_ratio(pts[6:])
        blink = float("nan")
        if res.face_blendshapes:
            scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
            if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
                blink = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0
        return EyeResult(
            face=True,
            ear=float(np.nanmean([ea, eb])),
            ear_a=ea,
            ear_b=eb,
            blink=blink,
            rotation=rot,
            eye_px=pts,
            prepared=img if self.keep_images else None,
        )
