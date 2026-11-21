#!/usr/bin/env python3
"""Chained bidirectional SolarRenderer render of one whole episode (one GPU).

    python solar_chain.py --carrier T3.mp4 --first FIRST_CANVAS.png --real REAL_F0_T3.png --prompt PROMPT.txt --root OUT
                          [--fps 30] [--win 532] [--overlap 16] [--seed 0] [--no-anchor] [--wins 141,447]
                          [--handover F:SRC_T3.png[:MASK_T3.png] ...] [--handover-mask-carrier WHITE_T3.mp4]
                          [--plan-only]

A whole-episode block-causal stream drifts after ~5-8 s (duplicated objects, colour noise), while one bidirectional
window renders 11-22 s cleanly. So the episode is cut into bidirectional windows (solar_canvas.plan_windows): window
w + 1 starts ``--overlap`` frames before window w ends and is conditioned on window w's rendered frame there.

Hand-over frame edits (only the conditioning frame of the next window; rendered frames are never modified):
  anchor     head-view pixels where the carrier equals its own frame 0 at that frame and that the task rarely moves
             over the episode are replaced by the real frame 0 (--real), so the static background does not drift from
             window to window (solar_canvas.anchor_blend; moved mask cached in OUT/anchor-exclude.png). --no-anchor
             disables it.
  handover   when the hand-over frame is carrier frame F, the pixels of MASK_T3 (default: the lit displays detected in
             --handover-mask-carrier, else --carrier, at F) are taken from SRC_T3, an accepted render of the same
             carrier frame. Use it with --wins so a window ends a few frames after an appearance change (a display
             lighting up): Solar keeps a fixed object's first-frame look for a whole window.

Windows run SOLAR_PYTHON solar_window.py; finished windows are reused, so a failed run resumes. The overlap frames are
cross-faded when stitched (solar_canvas.stitch_schedule). Outputs: OUT/stitched-t3.mp4 (1280x1080, --fps, output frame
i <-> carrier frame start + i), OUT/chain.json (plan, per-window records, hand-over edits, seams), OUT/w<k>.log.
--plan-only prints the window plan as JSON and exits (no GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_canvas as sc  # noqa: E402


def parse_handover(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not parts[0].isdigit():
        raise argparse.ArgumentTypeError(f"--handover wants F:SRC_T3.png[:MASK_T3.png], got {spec!r}")
    return {"at": int(parts[0]), "src": parts[1], "mask": parts[2] if len(parts) == 3 else None}


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, im = cap.read()
    if not ok:
        raise SystemExit(f"{path}: cannot read frame {index}")
    return im


def frame_count(path: str) -> int:
    return int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--carrier", required=True, help="1280x1080 T-layout white carrier")
    ap.add_argument("--first", help="canvas-sized first frame of window 0 (solar_first_frame.py)")
    ap.add_argument("--real", help="real frame 0, 1280x1080 T layout (background anchor)")
    ap.add_argument("--prompt", help="SolarRenderer prompt (solar_prompt.py)")
    ap.add_argument("--root", help="output directory")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--windows", type=int, default=0, help="render at most this many windows (0 = all)")
    ap.add_argument("--win", type=int, default=532, help="window length, 17k+5 (532 ~ 155 GB, 447 fits 141 GB)")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--wins", default="", help="explicit lengths of the first windows, comma separated")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--handover", action="append", type=parse_handover, default=[])
    ap.add_argument("--handover-mask-carrier", default="", help="white carrier the lit-display mask is detected in")
    ap.add_argument("--mode", choices=["bidir"], default="bidir")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)

    total = frame_count(a.carrier)
    wins = [int(x) for x in a.wins.split(",") if x]
    plan = sc.plan_windows(total, a.win, a.overlap, wins, a.start, a.windows)
    handover_at = {h["at"]: h for h in a.handover}
    hand_frames = {s for s, _ in plan[1:]}
    missed = sorted(set(handover_at) - hand_frames)
    if a.plan_only:
        print(json.dumps({"carrier_frames": total, "windows": plan, "handover_frames": sorted(hand_frames),
                          "handover_not_on_a_handover_frame": missed}))
        return 0
    for need in ("first", "real", "prompt", "root"):
        if not getattr(a, need):
            ap.error(f"--{need} is required unless --plan-only")
    if missed:
        ap.error(f"--handover frames {missed} are not hand-over frames of the plan {sorted(hand_frames)}; adjust --wins")

    import solar_paths
    solar_py = solar_paths.python_exe()
    K = a.overlap
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "datafarm.solar-chain/1", "seed": a.seed, "fps": a.fps, "carrier": a.carrier, "carrier_frames": total,
           "first": a.first, "real": a.real, "prompt": a.prompt, "start": a.start, "win": a.win, "wins": wins,
           "overlap": K, "anchor": not a.no_anchor, "handover": a.handover, "plan": plan, "windows": []}

    def save():
        (root / "chain.json").write_text(json.dumps(rec, indent=1))

    car0_c = sc.t3_to_canvas(read_frame(a.carrier, a.start))
    real_c = sc.t3_to_canvas(cv2.imread(a.real))
    no_paste = None

    def moved_mask() -> np.ndarray:
        f = root / "anchor-exclude.png"
        if f.exists():
            return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0
        counter = sc.MovedCounter(car0_c)
        cap = cv2.VideoCapture(a.carrier)
        k = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if k % 4 == 0:
                counter.add(sc.t3_to_canvas(im))
            k += 1
        m = counter.mask()
        cv2.imwrite(str(f), m.astype(np.uint8) * 255)
        return m

    first = a.first
    for w, (s, n) in enumerate(plan):
        clip = f"w{w}"
        d = root / "clips" / clip
        out_mp4 = root / f"{clip}-s{a.seed}.mp4"
        run_json = root / f"{clip}-s{a.seed}.run.json"
        info = {"window": w, "carrier_start": s, "carrier_frames": n}
        if out_mp4.exists() and run_json.exists():
            info.update(json.loads(run_json.read_text()))
            info["reused"] = True
        else:
            if d.exists():
                shutil.rmtree(d)
            subprocess.run([sys.executable, str(HERE / "solar_clip.py"), "--carrier", a.carrier, "--start", str(s),
                            "--n", str(n), "--layout", "t3", "--out", str(d), *([] if sc.legal_length(n) else ["--pad17"])],
                           check=True, stdout=subprocess.DEVNULL)
            im = sc.blank_bars(cv2.imread(first))
            cv2.imwrite(str(d / "first.png"), im)
            shutil.copy(a.prompt, d / "prompt.txt")
            b = time.time()
            with open(root / f"w{w}.log", "w") as log:
                r = subprocess.run([solar_py, str(HERE / "solar_window.py"), "--clip-dir", str(d), "--out-dir", str(root),
                                    "--name", clip, "--mode", a.mode, "--seed", str(a.seed)],
                                   stdout=log, stderr=subprocess.STDOUT)
            if run_json.exists():
                info.update(json.loads(run_json.read_text()))
            info.update({"rc": r.returncode, "wall_s": round(time.time() - b, 1)})
            if r.returncode or not out_mp4.exists():
                rec["windows"].append(info)
                save()
                print("window", w, "FAILED rc", r.returncode, "see", root / f"w{w}.log", flush=True)
                return 1
        rec["windows"].append(info)
        save()
        print("window", w, "start", s, "frames", n, "wall", info.get("wall_s"), "reused", info.get("reused", False),
              flush=True)
        if w + 1 == len(plan):
            break
        local = n - 1 - K                       # window w's frame that becomes window w+1's frame 0
        cf = s + local
        fr = read_frame(str(out_mp4), local)
        if not a.no_anchor:
            if no_paste is None:
                no_paste = moved_mask() | ~sc.head_zone()
            fr, frac = sc.anchor_blend(fr, sc.t3_to_canvas(read_frame(a.carrier, cf)), car0_c, real_c, no_paste)
            info["next_first_anchor_fraction"] = round(frac, 3)
        h = handover_at.get(cf)
        if h:
            if h["mask"]:
                mask = cv2.imread(h["mask"], cv2.IMREAD_GRAYSCALE)
            else:
                mask = sc.lit_display_mask(read_frame(a.handover_mask_carrier or a.carrier, cf))
            cv2.imwrite(str(root / f"w{w}-next-first-before-handover.png"), fr)
            fr, npx = sc.handover_paste(fr, cv2.imread(h["src"]), mask)
            info["next_first_handover"] = {"at": cf, "src": h["src"], "mask": h["mask"] or "lit-display detector",
                                           "pixels_t3": npx}
            if npx == 0:
                print("warning: hand-over mask at", cf, "is empty", flush=True)
        first = str(root / f"w{w}-next-first.png")
        cv2.imwrite(first, fr)
        save()

    # stitch: cross-fade over the overlap frames after each shared first frame
    windows = [(i["carrier_start"], i["carrier_frames"]) for i in rec["windows"]]
    sched = sc.stitch_schedule(windows, K)
    tmp = root / "stitched-t3.tmp.mp4"
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{sc.T3_W}x{sc.T3_H}", "-r", f"{a.fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "14",
                           "-preset", "medium", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    caps = {}
    cache: dict[tuple[int, int], np.ndarray] = {}

    def frame(wi: int, k: int) -> np.ndarray:
        key = (wi, k)
        if key not in cache:
            cap = caps.get(wi)
            if cap is None:
                cap = caps[wi] = [cv2.VideoCapture(str(root / f"w{wi}-s{a.seed}.mp4")), 0]
            while cap[1] <= k:              # sequential read; keep only frames still needed by a later blend
                ok, im = cap[0].read()
                if not ok:
                    raise SystemExit(f"window {wi}: rendered video ends before frame {k}")
                if cap[1] >= windows[wi][1] - 1 - K:
                    cache[(wi, cap[1])] = im
                elif cap[1] == k:
                    cache[key] = im
                cap[1] += 1
        return cache.pop(key)

    seams, nout = [], 0
    for _cf, parts in sched:
        if len(parts) == 1:
            im = frame(parts[0][0], parts[0][1]).astype(np.float32)
        else:
            (wa, ka, ua), (wb, kb, ub) = parts
            if kb == 1:
                seams.append(nout)
            im = frame(wa, ka).astype(np.float32) * ua + frame(wb, kb).astype(np.float32) * ub
        im = np.clip(im, 0, 255).astype(np.uint8)
        ff.stdin.write(sc.canvas_to_t3(im).tobytes())
        nout += 1
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("stitch: ffmpeg failed")
    os.replace(tmp, root / "stitched-t3.mp4")
    rec.update({"stitched_frames": nout, "seam_output_frames": seams,
                "crossfade_carrier_ranges": [[s + 1, s + K] for s, _ in windows[1:]],
                "stitched": str(root / "stitched-t3.mp4")})
    save()
    print("STITCHED", nout, "seams (first blended output frame) at", seams, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
