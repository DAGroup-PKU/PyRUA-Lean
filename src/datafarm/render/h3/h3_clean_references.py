#!/usr/bin/env python3
"""Object-free real scene reference from the real episode itself (references only).

Assumes a dark robot and saturated warm task objects; adjust the colour masks for other scenes.

A single-frame inpainted plate leaves smeared blocks where the objects were.  Every task object moves during the real
episode, so each head-view table pixel is visible as plain table in some frames: the head view of the scene reference is
the per-pixel temporal median of the real frames in which that pixel is neither robot (dark), task object (saturated warm,
grown 30 px) nor in a shadow (darker than the pixel's own bright cluster).  Wrist views are single real frames chosen from
the episode where the view shows only table / wall / fingertips (`--wrist-frames`); small object remnants listed in
`--wrist-erase` boxes are Telea-filled (labelled in the provenance).  The robot sheet keeps the base bundle's robot sheet
with task-object-coloured fragments set to the neutral sheet grey.

The default recipe's references_policy uses no isolated object sheets, because they make the generator re-insert the
object on the table.  A spec with `"object_sheet": true` also crops whole real objects from given real frames with a
background-deviation matte and adds that sheet to the references.

  python h3_clean_references.py <task> --base bundle.json --variant clean --spec <run>/tasks/<task>/clean-references.json
  (reads/writes <run>/tasks/<task>/; run directory = --run-dir or env H3_RUN)
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import h3_media
import h3_paths
from h3_media import sha256

GREY = 128
RIM_REACH_PX = 90  # pale rims and openings of a warm-coloured object sit up to this far above its coloured part


def read_frames(path: str, crop: list[int] | None, step: int = 1, only: list[int] | None = None) -> dict[int, np.ndarray]:
    """Frames of a real source video; `crop` = [x, y, w, h] of the real picture inside letterboxed source frames."""
    out = h3_media.read_frames(path, step=step, only=only)
    if crop:
        x, y, w, h = crop
        out = {k: cv2.resize(f[y:y + h, x:x + w], (1280, 720), interpolation=cv2.INTER_LANCZOS4) for k, f in out.items()}
    return out


def poly_features(h: int, w: int, deg: int = 3) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx / w - 0.5, yy / h - 0.5
    f = [np.ones_like(xx)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            f.append(xx ** (d - i) * yy ** i)
    return np.stack(f, -1)


def robust_fit(img: np.ndarray, m: np.ndarray, deg: int = 3) -> np.ndarray:
    """Per-channel polynomial colour field fitted on m, 3 reweighting rounds (residual < 6 levels)."""
    h, w = m.shape
    A = poly_features(h, w, deg)
    out = np.zeros(img.shape, np.float32)
    sub = np.zeros_like(m)
    sub[::3, ::3] = True
    for c in range(img.shape[2]):
        v = img[..., c].astype(np.float32)
        mm = m.copy()
        for _ in range(3):
            sel = mm & sub
            coef = np.linalg.lstsq(A[sel], v[sel], rcond=None)[0]
            fit = A @ coef
            mm = m & (np.abs(v - fit) < 6)
        out[..., c] = fit
    return out


def table_edge_rows(gray: np.ndarray) -> np.ndarray:
    """Row of the wall->table step per column (largest positive vertical step of the smoothed image)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    dy = g[6:] - g[:-6]
    rows = np.argmax(dy[: int(gray.shape[0] * 0.8)], axis=0) + 3
    k = 15
    pad = np.pad(rows, (k, k), mode="edge")
    return np.array([int(np.median(pad[i:i + 2 * k + 1])) for i in range(rows.size)])


def warm_mask(im: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (v >= 40) & (((s >= 45) & (h <= 35)) | ((s >= 30) & (h >= 140)))


def temporal_plate(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    st = np.stack(frames).astype(np.float32)                     # T,H,W,3
    gray = st.mean(axis=3)
    excl = np.zeros(gray.shape, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    for t, f in enumerate(frames):
        w = cv2.dilate(warm_mask(f).astype(np.uint8), k)
        up = np.zeros_like(w)
        up[:-RIM_REACH_PX] = w[RIM_REACH_PX:]
        obj = (w | up | cv2.dilate(up, k)) > 0
        dark = cv2.dilate((gray[t] < 110).astype(np.uint8), kd) > 0  # robot, cable, deep shadow
        excl[t] = obj | dark
    g = np.where(excl, np.nan, gray)
    hi = np.nanpercentile(g, 85, axis=0)
    use = ~excl & (gray >= hi[None] - 6)                          # bright (unshadowed) background cluster
    cnt = use.sum(0)
    st[~use] = np.nan
    plate = np.nanmedian(st, axis=0)
    hole = cnt < 3
    plate = np.nan_to_num(plate).clip(0, 255).astype(np.uint8)
    if hole.any():
        plate = cv2.inpaint(plate, hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    return plate, hole


def swap_plate(base: np.ndarray, fill: np.ndarray, check_fill: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base frame with its task objects (+ rims above, + contact shadow) replaced by the same pixels of a fill frame in
    which that area is plain table; per-channel gain matched on a ring around the region, 15 px feather; region = objects
    + rims above + 120 px right/40 px down shadow, clipped below the straight table edge.
    Returns (plate, swapped region, table mask)."""
    w0 = cv2.morphologyEx(warm_mask(base).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(w0)
    w = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 2000]).astype(np.uint8)
    up = np.zeros_like(w)
    for d in range(10, 130, 10):
        up[:-d] |= w[d:]
    reg = cv2.dilate(w | up, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    sh = np.zeros_like(reg)
    for d in range(10, 130, 10):                                # contact shadow falls right / below
        sh[:, d:] |= reg[:, :-d]
    sh[40:] |= reg[:-40]
    reg = (reg | sh) > 0
    edge = table_edge_rows(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)).astype(np.float64)
    xs = np.arange(edge.size)
    keep = np.ones(edge.size, bool)
    for _ in range(4):                                          # the wall/table edge is a straight line
        k, b0 = np.polyfit(xs[keep], edge[keep], 1)
        r = np.abs(edge - (k * xs + b0))
        keep = r <= max(3.0, np.percentile(r[keep], 60))
    edge = k * xs + b0
    table = np.arange(base.shape[0])[:, None] > (edge[None, :] + 8)
    reg &= table                                                # never repaint the wall / table edge
    fw = cv2.morphologyEx(warm_mask(fill).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fw)
    big = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 300])
    if check_fill and big[reg].any():
        raise SystemExit("fill frame has task objects inside the swap region")
    ring = (cv2.dilate(reg.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~reg
    gain = base[ring].reshape(-1, 3).astype(np.float64).mean(0) / (fill[ring].reshape(-1, 3).astype(np.float64).mean(0) + 1e-6)
    filled = np.clip(fill.astype(np.float32) * gain[None, None], 0, 255)
    a = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 7.5)[..., None]
    out = base.astype(np.float32) * (1 - a) + filled * a
    return out.clip(0, 255).astype(np.uint8), reg, table


def polyfill_plate(base: np.ndarray, reg: np.ndarray, table: np.ndarray, seed: int = 0) -> np.ndarray:
    """Region filled with a robust quadratic colour field fitted to the surrounding real table pixels of the same
    frame (window = region bbox + 120 px, excluding robot/object/shadow outliers) plus matched sensor noise."""
    ys, xs = np.nonzero(reg)
    y0, y1 = max(0, ys.min() - 120), min(base.shape[0], ys.max() + 121)
    x0, x1 = max(0, xs.min() - 120), min(base.shape[1], xs.max() + 121)
    win = base[y0:y1, x0:x1]
    r = reg[y0:y1, x0:x1]
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    tb = table[y0:y1, x0:x1]
    ok = ~r & tb & (g > np.percentile(g[~r & tb], 5)) & ~warm_mask(win)
    fit = robust_fit(win, ok, deg=2)
    resid = (win.astype(np.float32) - fit)[ok]
    noise = np.random.default_rng(seed).normal(0, float(np.median(np.abs(resid - np.median(resid))) * 1.4826 * 0.6),
                                               win.shape).astype(np.float32)
    a = cv2.GaussianBlur(r.astype(np.float32), (0, 0), 7.5)[..., None]
    out = base.astype(np.float32).copy()
    out[y0:y1, x0:x1] = win * (1 - a) + (fit + noise) * a
    return out.clip(0, 255).astype(np.uint8)


def object_crop(im: np.ndarray, box: list[int], matte: str = "deviation") -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = im[y0:y1, x0:x1].copy()
    if matte == "warm":  # large close-up objects: the median background is itself the object, use its colour instead
        dev = warm_mask(crop).astype(np.uint8)
    else:
        bg = cv2.medianBlur(im, 151)[y0:y1, x0:x1]
        dev = (np.abs(crop.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 12).astype(np.uint8)
    dev = cv2.morphologyEx(dev, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dev)
    if n > 1:
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        dev = (lab == keep).astype(np.uint8)
    pts = np.column_stack(np.nonzero(dev))[:, ::-1].astype(np.int32)
    m = np.zeros_like(dev)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
    return (crop * m + GREY * (1 - m)).astype(np.uint8)


def head_plate(spec: dict, source: str, crop) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Head view of the scene sheet: (plate, never-visible pixels, provenance frame, provenance description)."""
    hp = spec.get("head_plate", {"mode": "temporal"})
    if hp["mode"] == "frame":  # an unmodified real frame that shows no task object and no robot in the head view
        frame = int(hp["frame"])
        plate = read_frames(source, crop, only=[frame])[frame]
        return plate, np.zeros(plate.shape[:2], bool), frame, \
            f"head view = unmodified real frame {frame} (task objects out of view, arms out of view)"
    if hp["mode"] == "swap":
        base_frame, fill_frame = int(hp["base_frame"]), int(hp["fill_frame"])
        fr = read_frames(source, crop, only=[base_frame, fill_frame])
        poly = hp.get("fill") == "poly"
        plate, region, table = swap_plate(fr[base_frame], fr[fill_frame], check_fill=not poly)
        if poly:
            plate = polyfill_plate(fr[base_frame], region, table)
        filler = ("a robust quadratic colour field fitted to the surrounding real table pixels of the same frame plus "
                  "matched sensor noise (" if poly else
                  f"the same pixels of real frame {fill_frame} where that area is plain table (per-channel gain matched "
                  "on a ring, ")
        return plate, np.zeros_like(region), base_frame, (
            f"head view = real frame {base_frame} (arms out of view) with the task-object region "
            f"({float(region.mean()):.2%} of the view: objects, rims, contact shadows) replaced by {filler}"
            "15 px feather; region = objects + rims above + 120 px right/40 px down shadow)")
    stride = int(spec.get("head_stride", 3))
    frames = read_frames(source, crop, step=stride)
    plate, hole = temporal_plate(list(frames.values()))
    return plate, hole, 0, (
        f"head view = per-pixel temporal median of the real head-camera frames (stride {stride}, {len(frames)} frames) "
        "restricted to pixels that are not robot (dark, dilated), not task object (saturated warm, dilated 30 px) and in "
        f"the pixel's bright unshadowed cluster; {float(hole.mean()):.3%} never-visible pixels Telea-filled")


def wrist_frame(spec: dict, view: str, source: str, crop) -> tuple[int, np.ndarray]:
    index = int(spec["wrist_frames"][view])
    im = read_frames(source, crop, only=[index])[index]
    if view in spec.get("wrist_poly", []):
        # task object removed like the head plate: object (+ rims, contact shadow) region, clipped below the
        # straight table edge, filled with a robust colour field fitted to the surrounding real table pixels
        _, region, table = swap_plate(im, im, check_fill=False)
        im = polyfill_plate(im, region, table)
    boxes = spec.get("wrist_erase", {}).get(view, [])
    if boxes:
        m = np.zeros(im.shape[:2], np.uint8)
        for x0, y0, x1, y1 in boxes:
            m[y0:y1, x0:x1] = 255
        m[cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 60] = 0      # keep the dark fingertips
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        im = cv2.inpaint(im, m, 15, cv2.INPAINT_TELEA)
    return index, im


def object_sheet(spec: dict, sources: dict, crop) -> np.ndarray:
    tiles = []
    for o in spec["objects"]:
        index = int(o["frame"])
        t = object_crop(read_frames(sources[o["view"]], crop, only=[index])[index], o["box"], o.get("matte", "deviation"))
        s = min(340 / t.shape[0], 400 / t.shape[1])
        tiles.append(cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4))
    sheet = np.full((720, 1280, 3), GREY, np.uint8)
    x, y, rowh = 20, 20, 0
    for t in tiles:
        if x + t.shape[1] > 1260:
            x, y, rowh = 20, y + rowh + 20, 0
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 20
        rowh = max(rowh, t.shape[0])
    return sheet


def robot_sheet(spec: dict, rob_ref: dict | None) -> np.ndarray:
    rob = cv2.imread(rob_ref["path"]) if rob_ref else np.full((720, 1280, 3), GREY, np.uint8)
    frag = cv2.dilate(warm_mask(rob).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    rob[frag] = GREY
    if spec.get("robot_drop_thin"):
        kd = 25 if spec["robot_drop_thin"] is True else int(spec["robot_drop_thin"])
        # drop loose cables / stray fragments: keep only the thick robot bodies (opening with a 25 px disc by default,
        # components >= 0.3 % of the sheet), grown back by 6 px so link edges are not eroded
        fg = (np.abs(rob.astype(np.int16) - GREY).max(axis=2) > 6).astype(np.uint8)
        body = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(body)
        keep = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.003 * fg.size])
        keep = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        rob[~(keep & (fg > 0))] = GREY
    return rob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task")
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--spec", required=True)
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    tdir = h3_paths.run_dir(args.run_dir) / "tasks" / args.task
    spec = json.loads(Path(args.spec).read_text())
    crop = spec.get("source_crop")
    src = spec["sources"]                                         # view -> real mp4
    refs = tdir / "refs"
    plate, hole, head_frame, head_desc = head_plate(spec, src["head"], crop)
    wrist = {v: wrist_frame(spec, v, src[v], crop) for v in ("left_wrist", "right_wrist")}
    canvas = np.zeros((1080, 1280, 3), np.uint8)
    canvas[:720] = cv2.resize(plate, (1280, 720), interpolation=cv2.INTER_AREA)
    canvas[720:, :640] = cv2.resize(wrist["left_wrist"][1], (640, 360), interpolation=cv2.INTER_LANCZOS4)
    canvas[720:, 640:] = cv2.resize(wrist["right_wrist"][1], (640, 360), interpolation=cv2.INTER_LANCZOS4)
    scene = refs / f"{args.variant}-scene-3view.png"
    cv2.imwrite(str(scene), canvas)
    b = json.loads((tdir / args.base).read_text())
    rob_ref = next((r for r in b["real_references"] if "robot" in Path(r["path"]).name), None)
    if rob_ref is None and not spec.get("scene_keeps_robot"):
        raise SystemExit("base bundle has no robot sheet: set scene_keeps_robot only if the scene shows robot parts")
    robp = refs / f"{args.variant}-robot-sheet.png"
    cv2.imwrite(str(robp), robot_sheet(spec, rob_ref))
    full = next(r for r in b["real_references"] if r["scope"] == "full-view")
    nf = copy.deepcopy(full)
    nf["path"] = str(scene)
    nf["coverage"] = [c for c in full["coverage"] if c not in ("task_objects",)
                      + (() if spec.get("scene_keeps_robot") else ("robot_body",))]
    wrist_desc = (f". Wrist views = single unmodified real frames (left {wrist['left_wrist'][0]}, right "
                  f"{wrist['right_wrist'][0]}) showing table/wall/fingertips only")
    if spec.get("wrist_poly"):
        wrist_desc += ("; task object in " + ", ".join(spec["wrist_poly"]) +
                       " replaced by a robust colour field fitted to the surrounding real table (as the head)")
    if spec.get("wrist_erase"):
        wrist_desc += "; small object remnants Telea-filled in boxes " + json.dumps(spec["wrist_erase"])
    nf["provenance"] = {**full["provenance"], "source": src["head"], "frame": head_frame, "derived": True,
                        "all_sources": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in src.items()},
                        "description": "real scene without task objects or robot: " + head_desc + wrist_desc}
    nr = copy.deepcopy(rob_ref) if rob_ref else {"path": "", "provenance": {}}
    nr["path"] = str(robp)
    base_desc = (rob_ref or {}).get("provenance", {}).get("description", "")
    nr["provenance"] = {**(rob_ref or {}).get("provenance", {}), "description": (
        base_desc + " — task-object-coloured fragments set to the neutral sheet grey"
        + ("; loose cables and stray fragments removed (only thick robot bodies kept)" if spec.get("robot_drop_thin") else ""))}
    new_refs = [nf, nr] if rob_ref else [nf]
    record = {"base": args.base, "spec": spec, "scene": str(scene), "robot_sheet": str(robp),
              "never_visible_fraction": float(hole.mean())}
    if spec.get("object_sheet"):
        objp = refs / f"{args.variant}-object-sheet.png"
        cv2.imwrite(str(objp), object_sheet(spec, src, crop))
        objr = next(r for r in b["real_references"] if "object" in Path(r["path"]).name)
        no = copy.deepcopy(objr)
        no["path"] = str(objp)
        no["provenance"] = {**objr["provenance"], "source": src[spec["objects"][0]["view"]],
                            "frame": int(spec["objects"][0]["frame"]), "derived": True,
                            "description": "whole real task objects cropped from real frames with a background-deviation "
                                           "convex-hull matte on neutral grey: " + json.dumps(spec["objects"])}
        new_refs.append(no)
        record["object_sheet"] = str(objp)
    nb = copy.deepcopy(b)
    nb["real_references"] = new_refs
    nb.pop("per_episode_references", None)
    nb["clean_references"] = record
    dst = tdir / f"bundle-{args.variant}.json"
    dst.write_text(json.dumps(nb, indent=2, ensure_ascii=False))
    print(dst, scene, robp, round(float(hole.mean()), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
