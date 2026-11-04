"""Chunk visibility facts of the vendored H3 control hybrid: a wrist view that shows a white or pale task object (a
white kitchen scale) must not be measured as empty just because nothing in it is saturated.

The chunkvis prompt states an empty wrist as "shows only the tabletop, the background and its own gripper fingers,
with no task object in it", so a false "empty" contradicts the carrier.  Synthetic white-mesh-like views: flat wall and table, straight seams, dark fingers, soft shadow; a pale scale
(outline, buttons, display; HSV S < 70) and a saturated apple.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / 'src/datafarm/render/h3'


@pytest.fixture(scope='module')
def hcp():
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    if str(HYBRID) not in sys.path:
        sys.path.insert(0, str(HYBRID))
    import importlib
    return importlib.import_module('h3_package')


def wrist_geometry(hcp):
    s = hcp.SPATIAL
    hy = int(round(s.canvas_height * 720 / 1080))
    return hy, s.canvas_height - hy, (s.canvas_active_x1 - s.canvas_active_x0) // 2


def room(h, w, corner=False, seed=0):
    """Empty wrist view: wall above a slanted horizon (or a table corner polyline), table, soft shadow, two fingers."""
    import cv2
    import numpy as np
    img = np.full((h, w, 3), (230, 231, 228), np.uint8)
    if corner:  # table corner in front of two wall planes meeting at a vertical corner
        wall = np.array([[0, 0], [w, 0], [w, 95], [int(0.7 * w), 60], [int(0.25 * w), 62], [0, 110]], np.int32)
        cv2.fillPoly(img, [wall], (196, 190, 200))
        cv2.fillPoly(img, [np.array([[int(0.55 * w), 0], [w, 0], [w, 95], [int(0.7 * w), 60], [int(0.55 * w), 61]],
                                    np.int32)], (205, 200, 210))
    else:
        cv2.fillPoly(img, [np.array([[0, 0], [w, 0], [w, 40], [0, 72]], np.int32)], (148, 150, 149))
    yy, xx = np.mgrid[:h, :w]
    shade = 22 * np.exp(-(((xx - 0.6 * w) / (0.25 * w)) ** 2 + ((yy - 0.75 * h) / (0.3 * h)) ** 2))
    img = np.clip(img.astype(np.float32) - shade[..., None], 0, 255)
    img += np.random.default_rng(seed).normal(0, 1.2, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for x in (int(0.18 * w), int(0.8 * w)):  # own gripper fingers at the bottom edge
        cv2.fillPoly(img, [np.array([[x - 45, h], [x - 5, h - 70], [x + 15, h - 40], [x + 40, h]], np.int32)],
                     (58, 60, 61))
    return img


def with_scale(img):
    """A white kitchen scale close up: rounded body outline, inner platform outline, three buttons, pale display."""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    body = (int(0.12 * w), int(0.25 * h), int(0.9 * w), int(1.1 * h))
    cv2.rectangle(out, body[:2], body[2:], (236, 237, 234), -1)
    cv2.rectangle(out, body[:2], body[2:], (204, 206, 202), 2)
    cv2.rectangle(out, (body[0] + 18, body[1] + 16), (body[2] - 18, body[1] + 120), (214, 215, 212), 2)
    for k in range(3):
        x = int(0.55 * w) + k * 34
        cv2.rectangle(out, (x, int(0.78 * h)), (x + 24, int(0.78 * h) + 20), (198, 199, 196), 2)
    cv2.rectangle(out, (int(0.2 * w), int(0.8 * h)), (int(0.45 * w), int(0.93 * h)), (222, 214, 206), -1)
    cv2.rectangle(out, (int(0.2 * w), int(0.8 * h)), (int(0.45 * w), int(0.93 * h)), (196, 192, 186), 2)
    return out


def with_apple(img):
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    cv2.ellipse(out, (w // 2, int(0.8 * h)), (60, 50), 0, 0, 360, (45, 40, 200), -1)
    return out


def saturated_fraction(img):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] >= 70) & (hsv[..., 2] >= 40)).mean())


def test_white_scale_is_structure_not_saturation(hcp):
    _, h, w = wrist_geometry(hcp)
    empty, corner, scale = room(h, w), room(h, w, corner=True), with_scale(room(h, w))
    # saturation alone cannot see the white scale
    assert saturated_fraction(scale) <= 0.004
    # empty rooms (one horizon; a table-corner polyline with a wall corner) stay below the object threshold,
    # the pale scale is well above it
    assert hcp.wrist_structure(empty) < 0.25 * hcp.WRIST_MIN_STRUCTURE
    assert hcp.wrist_structure(corner) < hcp.WRIST_MIN_STRUCTURE
    assert hcp.wrist_structure(scale) >= 2 * hcp.WRIST_MIN_STRUCTURE


def write_clip(path, frames):
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{frames[0].shape[1]}x{frames[0].shape[0]}', '-r', '24', '-i', '-', '-an', '-c:v',
                          'libx264', '-preset', 'fast', '-crf', '10', '-pix_fmt', 'yuv420p', str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')
def test_chunk_facts_keep_a_white_scale_wrist_non_empty(hcp, tmp_path):
    import numpy as np
    s = hcp.SPATIAL
    hy, h, w = wrist_geometry(hcp)
    x0, xm = s.canvas_active_x0, s.canvas_active_x0 + w
    frames = []
    for i in range(48):  # 12 sampled frames; the right wrist sees the scale from frame 8 on (10/12 samples)
        canvas = np.zeros((s.canvas_height, s.canvas_width, 3), np.uint8)
        canvas[:hy, x0:s.canvas_active_x1] = 225
        canvas[hy:, x0:xm] = room(h, w, seed=i)
        right = room(h, w, corner=True, seed=100 + i)
        canvas[hy:, xm:xm + w] = with_scale(right) if i >= 8 else right
        frames.append(canvas)
    clip = tmp_path / 'white.mp4'
    write_clip(clip, frames)
    job = {'white': str(clip)}
    assert hcp.chunk_facts(job)['empty_wrists'] == ['left_wrist']
    assert hcp.chunk_facts(job, {'detector': 'wrist-structure-v1'})['empty_wrists'] == ['left_wrist']
    with pytest.raises(SystemExit, match='detector'):
        hcp.chunk_facts(job, {'detector': 'guess'})


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')
def test_chunk_facts_saturated_object_and_empty_room_unchanged(hcp, tmp_path):
    import numpy as np
    s = hcp.SPATIAL
    hy, h, w = wrist_geometry(hcp)
    x0, xm = s.canvas_active_x0, s.canvas_active_x0 + w
    frames = []
    for i in range(24):
        canvas = np.zeros((s.canvas_height, s.canvas_width, 3), np.uint8)
        canvas[:hy, x0:s.canvas_active_x1] = 225
        canvas[hy:, x0:xm] = with_apple(room(h, w, seed=i))
        canvas[hy:, xm:xm + w] = room(h, w, seed=50 + i)
        frames.append(canvas)
    clip = tmp_path / 'white.mp4'
    write_clip(clip, frames)
    for detector in hcp.CHUNK_FACTS_DETECTORS:
        assert hcp.chunk_facts({'white': str(clip)}, {'detector': detector})['empty_wrists'] == ['right_wrist']
