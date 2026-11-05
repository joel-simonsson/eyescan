import cv2
import numpy as np
import pytest

from eyesopenscan.eyes import ROTATIONS, _unrotate, eye_aspect_ratio


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_unrotate_inverts_cv2_rotate(rot):
    img = np.zeros((40, 70), np.uint8)
    img[7, 23] = 255  # (x=23, y=7)
    r = img if ROTATIONS[rot] is None else cv2.rotate(img, ROTATIONS[rot])
    y, x = np.argwhere(r == 255)[0]
    h, w = r.shape
    back = _unrotate(np.array([[x, y]], float), rot, w, h)[0]
    assert back.tolist() == [23, 7]


def test_ear_open_vs_closed():
    open_eye = np.array([[0, 0], [3, -3], [7, -3], [10, 0], [7, 3], [3, 3]], float)
    closed = open_eye * [1, 0.1]
    assert eye_aspect_ratio(open_eye) == pytest.approx(0.6)
    assert eye_aspect_ratio(closed) < 0.1
