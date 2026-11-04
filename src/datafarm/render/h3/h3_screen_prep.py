#!/usr/bin/env python3
"""Producer-side per-chunk seed screen: composites for a screening model, and the merge of its answers.

`composites` writes, per chunk and seed, 4 carrier|generated frame pairs from the chunk's keep range on one image, for a
separate screening model that picks the cleanest seed per chunk.  `merge` combines per-part answers (part-*.json in the
screen directory) into choice.json, schema h3.seed-screen/1.  The screen is NOT the acceptance review: the final package
still goes to the frozen, independent blind review (EVAL-v1).

  python h3_screen_prep.py composites --cands <pkg_s0> <pkg_s1> ... --out <run>/screen/<task>/<episode>
  python h3_screen_prep.py merge <screen dir> --chunks <n> --screener <label, e.g. the model that ran the screen>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from h3_media import read_frames
from h3_spatial import SPATIAL


def grab(path: str, idx: list[int]) -> dict:
    x0, x1 = SPATIAL.canvas_active_x0, SPATIAL.canvas_active_x1
    return read_frames(path, only=idx, transform=lambda frame: frame[:, x0:x1])


def composites(args) -> int:
    cands = [Path(c).resolve() for c in args.cands]
    mans = [json.loads((c / "job-manifest.json").read_text()) for c in cands]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"cands": [str(c) for c in cands], "chunks": []}
    for k, job in enumerate(mans[0]["jobs"]):
        a, b = job["keep_local_range"]
        frames = [int(a + (b - a) * q) for q in (0.12, 0.37, 0.62, 0.87)]
        white = grab(job["white"], frames)
        entry = {"chunk": k, "chunk_id": job["chunk_id"], "local_frames": frames,
                 "global_frames": [int(job["source_frame_range"][0]) + f for f in frames], "images": []}
        for s, (c, m) in enumerate(zip(cands, mans)):
            j = m["jobs"][k]
            if j["white_sha256"] != job["white_sha256"]:
                raise SystemExit(f"candidate {c} renders another white clip at chunk {k}")
            gen = grab(str(c / "render/chunks" / j["chunk_id"] / "chunk-render.mp4"), frames)
            rows = []
            for f, g in zip(frames, entry["global_frames"]):
                pair = [cv2.resize(white[f], (480, 405)), cv2.resize(gen[f], (480, 405))]
                cv2.putText(pair[0], f"SIM f{g}", (6, 22), 0, 0.7, (0, 0, 255), 2)
                cv2.putText(pair[1], f"CANDIDATE {s} f{g}", (6, 22), 0, 0.7, (0, 0, 255), 2)
                rows.append(np.hstack(pair))
            p = out / f"c{k:02d}-cand{s}.jpg"
            cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            entry["images"].append(str(p))
        index["chunks"].append(entry)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(out, len(index["chunks"]), "chunks x", len(cands), "candidates")
    return 0


def merge(args) -> int:
    chunks = {}
    for f in sorted(Path(args.screen_dir).glob("part-*.json")):
        for c in json.loads(f.read_text())["chunks"]:
            chunks[int(c["chunk"])] = c
    if sorted(chunks) != list(range(args.chunks)):
        raise SystemExit(f"parts cover chunks {sorted(chunks)}, expected 0..{args.chunks - 1}")
    rows = [chunks[i] for i in range(args.chunks)]
    seeds = range(len(rows[0]["scores"]))
    print("per-chunk choice:", [c["choice"] for c in rows])
    # a seed that is clean in every chunk is the single-seed alternative, which keeps one look across the episode
    print("per-seed worst:", [max(c["scores"][k] for c in rows) for k in seeds],
          "total:", [sum(c["scores"][k] for c in rows) for k in seeds])
    (Path(args.screen_dir) / "choice.json").write_text(json.dumps(
        {"schema": "h3.seed-screen/1", "screener": args.screener, "chunks": rows}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("composites")
    c.add_argument("--cands", nargs="+", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("screen_dir")
    m.add_argument("--chunks", type=int, required=True)
    m.add_argument("--screener", required=True)
    args = ap.parse_args()
    return {"composites": composites, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
