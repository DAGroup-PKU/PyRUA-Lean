#!/usr/bin/env python3
"""Canvas geometry, window plan, background anchoring, state hand-over and stitch schedule of the SolarRenderer route.

Pure numpy/opencv helpers shared by the chain, clip and first-frame tools (no model, no GPU), so the frame bookkeeping is
unit-tested on CPU.

Canvas: SolarRenderer renders a 1344x768 frame. The 1280x1080 three-camera T layout (head 1280x720 on top, left and
right wrist 640x360 below) is scaled into the 910x768 active box at x = 217 (the same active area the H3 control canvas
uses) with black bars on both sides.

Windows: one bidirectional window renders ``n`` carrier frames with n = 17k + 5 (Solar encodes 17k + 5 frames; a shorter
last window is padded by repeating its last frame and the padding is never written). Window w + 1 starts ``overlap``
frames before window w ends, s(w+1) = s(w) + n(w) - 1 - overlap, and is conditioned on window w's rendered frame at
s(w+1). The ``overlap`` frames after that shared frame are cross-faded linearly when stitched; every other output frame
is one rendered frame, so output frame i is carrier frame start + i.
"""
from __future__ import annotations

import numpy as np

CANVAS_W, CANVAS_H = 1344, 768
ACTIVE_X0, ACTIVE_W = 217, 910
T3_W, T3_H = 1280, 1080
HEAD_ROWS = 512                      # head panel rows on the canvas (720 of 1080 T-layout rows scaled to 768)
SOLAR_FPS = 24                       # frame-rate label of Solar clips; frames map 1:1 to carrier frames


def legal_length(n: int) -> bool:
    """Solar encodes 17k + 5 frames."""
    return n >= 22 and (n - 5) % 17 == 0


def pad_to_legal(n: int) -> int:
    """Smallest legal length >= n."""
    m = max(n, 22)
    return m + (-(m - 5)) % 17


def plan_windows(total: int, win: int, overlap: int, wins=(), start: int = 0, max_windows: int = 0) -> list[tuple[int, int]]:
    """[(carrier_start, frames)] for a carrier of ``total`` frames.

    ``wins`` gives explicit lengths for the first windows (to end a window just after a state change); later windows use
    ``win``. The chain stops when fewer than overlap + 2 frames remain or after ``max_windows`` windows (0 = no limit).
    """
    if not legal_length(win):
        raise ValueError(f"window length {win} is not 17k+5")
    for n in wins:
        if n <= overlap + 1:
            raise ValueError(f"explicit window length {n} must exceed overlap + 1 = {overlap + 1}")
    out: list[tuple[int, int]] = []
    s, w = start, 0
    while True:
        remaining = total - s
        if remaining < overlap + 2 or (max_windows and w >= max_windows):
            break
        n = min(wins[w] if w < len(wins) else win, remaining)
        out.append((s, n))
        if s + n >= total:
            break
        s += n - 1 - overlap
        w += 1
    return out


def stitch_schedule(windows: list[tuple[int, int]], overlap: int) -> list[tuple[int, list[tuple[int, int, float]]]]:
    """[(carrier_frame, [(window_index, local_frame, weight), ...])] in output order.

    Window i > 0 skips its frame 0 (the shared conditioning frame, written by window i - 1). Its frames 1..overlap are
    blended with window i - 1's frames at the same carrier frame, the weight of window i rising as k / (overlap + 1).
    Window i - 1's frames after the shared frame are written only through that blend.
    """
    out = []
    for i, (s0, n) in enumerate(windows):
        nxt = windows[i + 1][0] if i + 1 < len(windows) else None
        for k in range(n):
            cf = s0 + k
            if i > 0 and k == 0:
                continue
            if nxt is not None and cf > nxt:
                continue
            if i > 0 and 1 <= k <= overlap:
                alpha = k / (overlap + 1)
                ps = windows[i - 1][0]
                out.append((cf, [(i - 1, cf - ps, 1.0 - alpha), (i, k, alpha)]))
            else:
                out.append((cf, [(i, k, 1.0)]))
    return out


def t3_to_canvas(t3: np.ndarray) -> np.ndarray:
    import cv2
    c = np.zeros((CANVAS_H, CANVAS_W) + t3.shape[2:], t3.dtype)
    c[:, ACTIVE_X0:ACTIVE_X0 + ACTIVE_W] = cv2.resize(t3, (ACTIVE_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    return c


def canvas_to_t3(canvas: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(canvas[:, ACTIVE_X0:ACTIVE_X0 + ACTIVE_W], (T3_W, T3_H), interpolation=cv2.INTER_CUBIC)


def blank_bars(canvas: np.ndarray) -> np.ndarray:
    canvas[:, :ACTIVE_X0] = 0
    canvas[:, ACTIVE_X0 + ACTIVE_W:] = 0
    return canvas


def head_zone() -> np.ndarray:
    """Canvas pixels that may be re-anchored: the head view only (the wrist cameras move with the grippers)."""
    z = np.zeros((CANVAS_H, CANVAS_W), bool)
    z[:HEAD_ROWS, ACTIVE_X0:ACTIVE_X0 + ACTIVE_W] = True
    return z


# ------------------------------------------------------------------ background anchoring
class MovedCounter:
    """Pixels the task moves over the episode: the carrier differs from its first frame by >= ``diff`` in more than
    ``frac`` of the sampled frames, dilated ``dilate`` px.

    A plain "ever changed" test excludes almost everything (arm shadows sweep the whole head view over a long episode);
    a per-frame test alone pastes a closed white drawer onto an open one when white slides over white. The frequency
    rule keeps walls and furniture the task never touches and excludes the work area.
    """

    def __init__(self, car0_canvas: np.ndarray, diff: int = 24):
        self.car0 = car0_canvas.astype(np.int16)
        self.diff = diff
        self.count = np.zeros(car0_canvas.shape[:2], np.int32)
        self.n = 0

    def add(self, carrier_canvas: np.ndarray) -> None:
        self.count += np.abs(carrier_canvas.astype(np.int16) - self.car0).max(2) >= self.diff
        self.n += 1

    def mask(self, frac: float = 0.01, dilate: int = 41) -> np.ndarray:
        import cv2
        moved = (self.count > frac * max(self.n, 1)).astype(np.uint8)
        return cv2.dilate(moved, np.ones((dilate, dilate), np.uint8)).astype(bool)


def anchor_blend(frame_c: np.ndarray, car_c: np.ndarray, car0_c: np.ndarray, real_c: np.ndarray, no_paste: np.ndarray,
                 still: int = 8, erode: int = 31, feather: float = 8.0) -> tuple[np.ndarray, float]:
    """Replace the static background of a hand-over frame by the real frame 0.

    A pixel is static when the carrier at this frame equals its frame 0 (max channel difference < ``still``) and it is
    not in ``no_paste`` (moved mask | outside the head view). The mask is eroded ``erode`` px and feathered (Gaussian
    ``feather`` px). Returns the frame and the mean mask weight over the active box.
    """
    import cv2
    static = (np.abs(car_c.astype(np.int16) - car0_c.astype(np.int16)).max(2) < still) & ~no_paste
    static = cv2.erode(static.astype(np.uint8), np.ones((erode, erode), np.uint8))
    m = cv2.GaussianBlur(static.astype(np.float32), (0, 0), feather)[..., None]
    out = np.clip(frame_c.astype(np.float32) * (1 - m) + real_c.astype(np.float32) * m, 0, 255).astype(np.uint8)
    return blank_bars(out), float(m[:, ACTIVE_X0:ACTIVE_X0 + ACTIVE_W].mean())


# ------------------------------------------------------------------ state hand-over
def lit_display_mask(carrier_t3: np.ndarray, min_area: int = 30, max_area: int = 6000) -> np.ndarray:
    """uint8 mask (255) of lit displays in a white carrier frame: near-white, low-saturation, wide, filled rectangles
    (V > 245, S < 40; area in [min_area, max_area]; height >= 3; width > 1.6 height; fill > 60 %)."""
    import cv2
    hsv = cv2.cvtColor(carrier_t3, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(m.shape, np.uint8)
    for c in range(1, k):
        _x, _y, w, h, area = st[c]
        if min_area <= area <= max_area and h >= 3 and w > 1.6 * h and area > 0.6 * w * h:
            keep[lab == c] = 255
    return keep


def handover_paste(frame_c: np.ndarray, src_t3: np.ndarray, mask_t3: np.ndarray, dilate: int = 7,
                   feather: float = 1.5) -> tuple[np.ndarray, int]:
    """Blend ``src_t3`` (an accepted render of the same carrier frame, 1280x1080) into the canvas hand-over frame inside
    ``mask_t3`` (dilated ``dilate`` px, feathered ``feather`` px). Returns the frame and the mask's T-layout pixel count.

    Solar keeps a window's first-frame look for fixed objects, so an appearance that changes mid-window (a display
    lighting up, a lamp turning on) stays in its old state. Ending the window shortly after the change and handing the
    next window a first frame with the new look fixes it; only that conditioning frame is edited.
    """
    import cv2
    keep = (mask_t3 > 0).astype(np.uint8) * 255
    if dilate > 1:
        keep = cv2.dilate(keep, np.ones((dilate, dilate), np.uint8))
    mc = t3_to_canvas(keep).astype(np.float32) / 255
    if feather > 0:
        mc = cv2.GaussianBlur(mc, (0, 0), feather)
    mc = mc[..., None]
    out = frame_c.astype(np.float32) * (1 - mc) + t3_to_canvas(src_t3).astype(np.float32) * mc
    return np.clip(out, 0, 255).astype(np.uint8), int((keep > 0).sum())


def fill_lit_displays(carrier_t3: np.ndarray, bgr=(255, 178, 92)) -> tuple[np.ndarray, int]:
    """Carrier frame with every lit display (``lit_display_mask``, 3 px rim) filled with a flat backlight colour, so the
    prompt can name the lit state by a colour the carrier really shows. Returns the frame and the component count."""
    import cv2
    m = lit_display_mask(carrier_t3)
    if not m.any():
        return carrier_t3, 0
    k = cv2.connectedComponents(m)[0] - 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0
    out = carrier_t3.copy()
    out[m] = bgr
    return out, k
