#!/usr/bin/env python3
"""Frozen automatic evaluator for H3 renders against their simulator carrier (EVAL-v1).

Scores one generated render against its simulator carrier (the sole geometry/timing
authority) and a pool of real-camera frames (the appearance target).  It is
renderer-agnostic: H3 and Seedance outputs are scored by the same code.

Subcommands
  realpool  embed real frames per view with DINOv2 -> npz cache
  score     score one generated video vs carrier (+ real pool) -> score.json
  pack      build a neutral lite review pack (carrier|generated grids, refs)
  table     aggregate score.json files into a CSV/JSON summary

Metric definitions (all per view; views are cut from a canonical layout after the
generated video is resized to the carrier frame size):
  motion_corr   mean over sampled t of Pearson(|C[t+2]-C[t]|, |G[t+2]-G[t]|) on blurred
                gray 256x144 maps, weighted by carrier motion std; frames where the
                carrier is static are skipped.  1.0 = identical motion pattern.
  lag_frames    lag (generated minus carrier, frames) maximising the correlation of
                16x9 block motion-energy sequences over all frames, searched in [-16,16].
  struct_corr   mean Pearson between Sobel-magnitude maps of C[t] and G[t] (blurred
                gray 256x144).  Higher = same edges/silhouettes in the same places.
  flicker       mean |G[t+2]-G[t]| (0..255) on pixels where the carrier is static.
  kid           KID x1000 (poly kernel, unbiased) between DINOv2-base CLS embeddings of
                generated frames and the real pool of the same view.  Lower = closer to
                the real camera distribution.
  nn_cos        mean over generated frames of max cosine to any real-pool frame.
  lab_de        CIE76 distance between mean Lab of generated frames and of real pool.
  jumps_per_1k  generated-only jumps per 1000 frames: t where the whole-view mean |G[t+1]-G[t]|
                exceeds 4/255, is > 3x its local (+-12 frame) median and > 2x the carrier change + 2.
  flips_per_1k  whole-view exposure flips per 1000 frames: |mean luma G[t+1]-G[t]| > 5 while the
                carrier's |mean luma change| < 1.5.
  freeze_frames frames inside runs (>= 6) where G is static (< 0.15) but the carrier moves (> 1.0).
  white_cos     mean cosine between DINO embeddings of G[t] and C[t] (diagnostic of
                residual simulator look; not a gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL_VERSION = "EVAL-v1"

LAYOUTS = {
    # name: (canvas_w, canvas_h, {view: (x, y, w, h)})
    "t3": (1280, 1080, {"head": (0, 0, 1280, 720), "left_wrist": (0, 720, 640, 360),
                         "right_wrist": (640, 720, 640, 360)}),
    "single": (None, None, {"view": None}),
}
SMALL = (256, 144)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n, w, h, fps


def iter_frames(path: Path, size: tuple[int, int] | None = None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if size is not None and (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        yield frame
    cap.release()


def views_of(layout: str, frame: np.ndarray) -> dict[str, np.ndarray]:
    _, _, spec = LAYOUTS[layout]
    if layout == "single":
        return {"view": frame}
    return {v: frame[y:y + h, x:x + w] for v, (x, y, w, h) in spec.items()}


def gray_small(view: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(view, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g.astype(np.float32), (5, 5), 1.2)


def block_energy(diff: np.ndarray) -> np.ndarray:
    return cv2.resize(diff, (16, 9), interpolation=cv2.INTER_AREA).ravel()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else float("nan")


def sobel_mag(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class Dino:
    def __init__(self, model_dir: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval().to(device)
        self.device = device

    def embed(self, bgr_images: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(bgr_images), batch):
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in bgr_images[i:i + batch]]
            inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                cls = self.model(**inputs).last_hidden_state[:, 0]
            cls = self.torch.nn.functional.normalize(cls.float(), dim=-1)
            out.append(cls.cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 768), np.float32)


def kid(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased KID (poly kernel degree 3) between row-embedding sets, x1000."""
    d = x.shape[1]
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sxy = kxy.mean()
    return float((sxx + syy - 2 * sxy) * 1000.0)


def mean_lab(images: list[np.ndarray]) -> np.ndarray:
    labs = [cv2.cvtColor(cv2.resize(im, (128, 72)), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
            for im in images]
    lab = np.concatenate(labs, 0)
    # OpenCV 8-bit Lab: L*255/100, a+128, b+128
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab.mean(0)


# ----------------------------------------------------------------------------- realpool
def cmd_realpool(args) -> int:
    """Embed real frames. Sources: JSON list of {video|image, view, layout?, stride?}."""
    spec = json.loads(Path(args.spec).read_text())
    dino = Dino(args.dino, args.device)
    per_view: dict[str, list[np.ndarray]] = {}
    sources = []
    for item in spec["sources"]:
        view = item["view"]
        if "image" in item:
            im = cv2.imread(item["image"])
            if im is None:
                raise RuntimeError(item["image"])
            region = item.get("region")
            if region:
                x, y, w, h = region
                im = im[y:y + h, x:x + w]
            per_view.setdefault(view, []).append(im)
            sources.append({"image": item["image"], "sha256": sha256(Path(item["image"])), "view": view})
        else:
            stride = int(item.get("stride", 15))
            region = item.get("region")
            k = 0
            for i, fr in enumerate(iter_frames(Path(item["video"]))):
                if i % stride:
                    continue
                if region:
                    x, y, w, h = region
                    fr = fr[y:y + h, x:x + w]
                per_view.setdefault(view, []).append(fr)
                k += 1
            sources.append({"video": item["video"], "sha256": sha256(Path(item["video"])), "view": view,
                            "stride": stride, "frames_used": k})
    arrays, labs = {}, {}
    for view, ims in per_view.items():
        arrays[f"emb_{view}"] = dino.embed(ims)
        arrays[f"lab_{view}"] = mean_lab(ims)
    np.savez(args.out, **arrays)
    Path(args.out + ".json").write_text(json.dumps({"eval_version": EVAL_VERSION, "task": spec.get("task"),
                                                     "sources": sources,
                                                     "counts": {v: len(i) for v, i in per_view.items()}},
                                                    indent=2))
    print(json.dumps({v: len(i) for v, i in per_view.items()}))
    return 0


# ----------------------------------------------------------------------------- score
def cmd_score(args) -> int:
    gen, car = Path(args.generated), Path(args.carrier)
    ng, gw, gh, gfps = video_info(gen)
    nc, cw, ch, cfps = video_info(car)
    layout = args.layout
    size = (cw, ch)
    n = min(ng, nc)
    stride, gap = args.stride, 2
    keep = set()
    for t in range(0, n - gap, stride):
        keep.add(t)
        keep.add(t + gap)
    emb_stride = args.emb_stride
    emb_idx = set(range(0, n, emb_stride))

    def collect(path: Path):
        grays: dict[str, dict[int, np.ndarray]] = {}
        energy: dict[str, list[np.ndarray]] = {}
        prev: dict[str, np.ndarray] = {}
        mad: dict[str, list[float]] = {}
        luma: dict[str, list[float]] = {}
        emb_frames: dict[str, dict[int, np.ndarray]] = {}
        for i, fr in enumerate(iter_frames(path, size)):
            if i >= n:
                break
            for v, im in views_of(layout, fr).items():
                g = gray_small(im)
                if i in keep:
                    grays.setdefault(v, {})[i] = g
                luma.setdefault(v, []).append(float(g.mean()))
                if v in prev:
                    diff = np.abs(g - prev[v])
                    energy.setdefault(v, []).append(block_energy(diff))
                    mad.setdefault(v, []).append(float(diff.mean()))
                prev[v] = g
                if i in emb_idx:
                    emb_frames.setdefault(v, {})[i] = cv2.resize(im, (448, 252), interpolation=cv2.INTER_AREA)
        return grays, energy, emb_frames, mad, luma

    cg, ce, cemb, cmad, cluma = collect(car)
    gg, ge, gemb, gmad, gluma = collect(gen)
    pool = np.load(args.realpool) if args.realpool else None
    dino = Dino(args.dino, args.device) if (pool is not None or args.white_cos) else None

    result = {"eval_version": EVAL_VERSION, "generated": str(gen), "generated_sha256": sha256(gen),
              "carrier": str(car), "carrier_sha256": sha256(car), "layout": layout,
              "frames": {"generated": ng, "carrier": nc, "scored": n},
              "generated_size": [gw, gh], "carrier_size": [cw, ch], "fps": {"generated": gfps, "carrier": cfps},
              "realpool": args.realpool, "views": {}}
    if ng != nc:
        result["frame_count_mismatch"] = True
    for v in cg:
        mc_vals, mc_w, sc_vals, fl_vals = [], [], [], []
        for t in range(0, n - gap, stride):
            c0, c1, g0, g1 = cg[v][t], cg[v][t + gap], gg[v][t], gg[v][t + gap]
            dc, dg = np.abs(c1 - c0), np.abs(g1 - g0)
            sd = float(dc.std())
            if sd > 1.0:
                r = pearson(dc, dg)
                if not math.isnan(r):
                    mc_vals.append(r)
                    mc_w.append(sd)
            static = cv2.dilate((dc > 3.0).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
            if static.mean() > 0.05:
                fl_vals.append(float(dg[static].mean()))
            r2 = pearson(sobel_mag(c0), sobel_mag(g0))
            if not math.isnan(r2):
                sc_vals.append(r2)
        ec, eg = np.stack(ce[v]), np.stack(ge[v])
        best = (None, -2.0)
        lag_curve = {}
        for lag in range(-16, 17):
            if lag >= 0:
                a, b = ec[:len(ec) - lag], eg[lag:]
            else:
                a, b = ec[-lag:], eg[:len(eg) + lag]
            m = min(len(a), len(b))
            r = pearson(a[:m], b[:m])
            lag_curve[lag] = r
            if not math.isnan(r) and r > best[1]:
                best = (lag, r)
        dg_, dc_ = np.array(gmad[v]), np.array(cmad[v])
        m = min(len(dg_), len(dc_))
        dg_, dc_ = dg_[:m], dc_[:m]
        jumps = []
        for t in range(m):
            lo, hi = max(0, t - 12), min(m, t + 13)
            local = np.median(np.r_[dg_[lo:t], dg_[t + 1:hi]]) if hi - lo > 1 else 0.0
            if dg_[t] > 4.0 and dg_[t] > 3 * local and dg_[t] > 2 * dc_[t] + 2:
                jumps.append(t)
        lg, lc = np.diff(np.array(gluma[v])[:m + 1]), np.diff(np.array(cluma[v])[:m + 1])
        flips = [int(t) for t in np.where((np.abs(lg) > 5) & (np.abs(lc) < 1.5))[0]]
        static_g, moving_c = dg_ < 0.15, dc_ > 1.0
        freeze, run = 0, 0
        for a, b in zip(static_g, moving_c):
            run = run + 1 if (a and b) else 0
            if run == 6:
                freeze += 6
            elif run > 6:
                freeze += 1
        vr = {
            "jumps_per_1k": 1000.0 * len(jumps) / max(1, m), "jump_frames": jumps[:50],
            "flips_per_1k": 1000.0 * len(flips) / max(1, m), "flip_frames": flips[:50],
            "freeze_frames": freeze,
            "motion_corr": float(np.average(mc_vals, weights=mc_w)) if mc_vals else None,
            "motion_frames": len(mc_vals),
            "struct_corr": float(np.mean(sc_vals)) if sc_vals else None,
            "flicker": float(np.mean(fl_vals)) if fl_vals else None,
            "lag_frames": best[0], "lag_peak_corr": best[1], "lag0_corr": lag_curve.get(0),
        }
        if dino is not None:
            idx = sorted(gemb[v])
            ge_emb = dino.embed([gemb[v][i] for i in idx])
            if args.white_cos:
                ce_emb = dino.embed([cemb[v][i] for i in idx])
                vr["white_cos"] = float((ge_emb * ce_emb).sum(1).mean())
            if pool is not None and f"emb_{v}" in pool:
                real = pool[f"emb_{v}"]
                vr["kid"] = kid(ge_emb, real)
                vr["nn_cos"] = float((ge_emb @ real.T).max(1).mean())
                vr["lab_de"] = float(np.linalg.norm(mean_lab([gemb[v][i] for i in idx]) - pool[f"lab_{v}"]))
                vr["emb_frames"] = len(idx)
        result["views"][v] = vr
    keys = ["motion_corr", "struct_corr", "flicker", "kid", "nn_cos", "lab_de", "white_cos", "jumps_per_1k",
            "flips_per_1k", "freeze_frames"]
    result["summary"] = {k: (float(np.mean([r[k] for r in result["views"].values() if r.get(k) is not None]))
                             if any(r.get(k) is not None for r in result["views"].values()) else None)
                         for k in keys}
    result["summary"]["abs_lag_max"] = max(abs(r["lag_frames"]) for r in result["views"].values()
                                           if r["lag_frames"] is not None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]))
    return 0


# ----------------------------------------------------------------------------- pack
def _label(im: np.ndarray, text: str) -> np.ndarray:
    im = im.copy()
    cv2.rectangle(im, (0, 0), (min(im.shape[1], 12 + 9 * len(text)), 20), (0, 0, 0), -1)
    cv2.putText(im, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def cmd_pack(args) -> int:
    """Neutral lite pack: per view grids of C|G pairs at K evenly spaced frames, detail, refs."""
    gen, car = Path(args.generated), Path(args.carrier)
    ng, _, _, _ = video_info(gen)
    nc, cw, ch, _ = video_info(car)
    n = min(ng, nc)
    k = args.samples
    idx = sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))
    detail_idx = idx[:: max(1, len(idx) // 6)][:6]
    want = set(idx)
    cfr, gfr = {}, {}
    for i, fr in enumerate(iter_frames(car)):
        if i in want:
            cfr[i] = fr
        if i >= n:
            break
    for i, fr in enumerate(iter_frames(gen, (cw, ch))):
        if i in want:
            gfr[i] = fr
        if i >= n:
            break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    pw, ph = 400, 225
    for v in views_of(args.layout, cfr[idx[0]]):
        tiles = []
        for i in idx:
            c = cv2.resize(views_of(args.layout, cfr[i])[v], (pw, ph), interpolation=cv2.INTER_AREA)
            g = cv2.resize(views_of(args.layout, gfr[i])[v], (pw, ph), interpolation=cv2.INTER_AREA)
            tiles.append(np.hstack([_label(c, f"C {i}"), np.full((ph, 4, 3), 255, np.uint8), _label(g, f"G {i}")]))
        per_sheet = 10
        for s in range(0, len(tiles), per_sheet):
            group = tiles[s:s + per_sheet]
            while len(group) % 2:
                group.append(np.zeros_like(tiles[0]))
            rows = [np.hstack([group[j], np.full((ph, 12, 3), 128, np.uint8), group[j + 1]])
                    for j in range(0, len(group), 2)]
            sheet = np.vstack([np.vstack([r, np.full((6, r.shape[1], 3), 128, np.uint8)]) for r in rows])
            name = f"grid-{v}-{s // per_sheet:02d}.jpg"
            cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            files.append(name)
    det = [cv2.resize(gfr[i], (960, int(960 * ch / cw)), interpolation=cv2.INTER_AREA) for i in detail_idx]
    det = [_label(d, f"G {i}") for d, i in zip(det, detail_idx)]
    while len(det) % 2:
        det.append(np.zeros_like(det[0]))
    sheet = np.vstack([np.hstack(det[j:j + 2]) for j in range(0, len(det), 2)])
    cv2.imwrite(str(out / "detail.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    files.append("detail.jpg")
    if args.ref:
        refs = []
        for r in args.ref:
            im = cv2.imread(r)
            refs.append(cv2.resize(im, (960, int(960 * im.shape[0] / im.shape[1]))))
        width = max(r.shape[1] for r in refs)
        refs = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in refs]
        cv2.imwrite(str(out / "refs.jpg"), np.vstack(refs), [cv2.IMWRITE_JPEG_QUALITY, 88])
        files.append("refs.jpg")
    manifest = {"eval_version": EVAL_VERSION, "generated_sha256": sha256(gen), "carrier_sha256": sha256(car),
                "frames": {"generated": ng, "carrier": nc}, "sampled_frames": idx, "detail_frames": detail_idx,
                "files": {f: sha256(out / f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"files": len(files), "frames": idx[:3] + ["..."]}))
    return 0


# ----------------------------------------------------------------------------- table
def cmd_table(args) -> int:
    rows = []
    for p in args.scores:
        d = json.loads(Path(p).read_text())
        row = {"path": p, **{k: v for k, v in d["summary"].items()}}
        for v, r in d["views"].items():
            for k in ("motion_corr", "struct_corr", "kid", "lag_frames"):
                row[f"{v}.{k}"] = r.get(k)
        rows.append(row)
    print(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    # DINOv2-base directory: --dino, else the H3_EVAL_DINO deployment binding; no built-in default
    dino = os.environ.get("H3_EVAL_DINO") or os.environ.get("DATAFARM_H3_EVAL_DINO")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("realpool")
    rp.add_argument("--spec", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--dino", default=dino)
    rp.add_argument("--device", default="cpu")
    sc = sub.add_parser("score")
    sc.add_argument("--generated", required=True)
    sc.add_argument("--carrier", required=True)
    sc.add_argument("--layout", default="t3", choices=list(LAYOUTS))
    sc.add_argument("--realpool")
    sc.add_argument("--out", required=True)
    sc.add_argument("--stride", type=int, default=4)
    sc.add_argument("--emb-stride", type=int, default=12)
    sc.add_argument("--white-cos", action="store_true")
    sc.add_argument("--dino", default=dino)
    sc.add_argument("--device", default="cpu")
    pk = sub.add_parser("pack")
    pk.add_argument("--generated", required=True)
    pk.add_argument("--carrier", required=True)
    pk.add_argument("--layout", default="t3", choices=list(LAYOUTS))
    pk.add_argument("--samples", type=int, default=30)
    pk.add_argument("--ref", action="append")
    pk.add_argument("--out", required=True)
    tb = sub.add_parser("table")
    tb.add_argument("scores", nargs="+")
    args = ap.parse_args()
    if args.cmd in ("realpool", "score") and not args.dino:
        ap.error("--dino is required: pass the DINOv2-base model directory or set H3_EVAL_DINO")
    import torch  # noqa: F401  (fail early if env is wrong)
    torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "16")))
    return {"realpool": cmd_realpool, "score": cmd_score, "pack": cmd_pack, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
