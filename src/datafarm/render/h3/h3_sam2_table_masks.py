#!/usr/bin/env python3
"""Track the TABLETOP through one panel of a generated T-layout video with SAM2.1 video (CPU ok).

Prompts: positive points on bright, low-saturation, low-texture pixels in the lower panel + negatives on the wall band, at
frame 0 and re-prompted every --every frames (recovers after long occlusions).  Output: uint8 masks (frames, h, w) .npy,
consumed by h3_ood_restyle.py --masks.

  python h3_sam2_table_masks.py --video final.mp4 --panel head --out masks_head.npy [--max-frames N] [--threads N]
SAM2.1 weights: directory of facebook/sam2.1-hiera-large from the H3_SAM2 binding (h3_paths.py; offline load).
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import h3_paths
from h3_spatial import T_LAYOUT_PANELS as PANELS


def prompts(p):
    h, w = p.shape[:2]
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    var = cv2.blur(g * g, (15, 15)) - cv2.blur(g, (15, 15)) ** 2
    pos = [
        [int(fx * w), int(fy * h)]
        for fy in (0.9, 0.78, 0.66)
        for fx in np.linspace(0.12, 0.88, 7)
        if hsv[int(fy * h), int(fx * w), 1] < 70
        and hsv[int(fy * h), int(fx * w), 2] > 110
        and var[int(fy * h), int(fx * w)] < 60
    ]
    neg = [[int(fx * w), int(0.04 * h)] for fx in np.linspace(0.15, 0.85, 5)]
    return pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--panel", required=True, choices=list(PANELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--every", type=int, default=96)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_dir = h3_paths.sam2_model()
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    torch.set_num_threads(a.threads)
    y0, y1, x0, x1 = PANELS[a.panel]
    cap = cv2.VideoCapture(a.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (a.max_frames and len(frames) >= a.max_frames):
            break
        frames.append(f[y0:y1, x0:x1])
    H, W = frames[0].shape[:2]
    proc = Sam2VideoProcessor.from_pretrained(model_dir)
    model = Sam2VideoModel.from_pretrained(model_dir).eval()
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    sess = proc.init_video_session(video=rgb, inference_device="cpu", dtype=torch.float32)
    keys = list(range(0, len(frames), a.every))
    for k in keys:
        pos, neg = prompts(frames[k])
        if not pos:
            continue
        proc.add_inputs_to_inference_session(
            inference_session=sess,
            frame_idx=k,
            obj_ids=1,
            input_points=[[pos + neg]],
            input_labels=[[[1] * len(pos) + [0] * len(neg)]],
        )
        with torch.no_grad():
            model(inference_session=sess, frame_idx=k)
    masks = np.zeros((len(frames), H, W), np.uint8)
    t0 = time.time()
    with torch.no_grad():
        for out in model.propagate_in_video_iterator(sess, start_frame_idx=0):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            masks[out.frame_idx] = m.reshape(-1, H, W)[0].numpy().astype(np.uint8)
    np.save(a.out, masks)
    print(
        json.dumps(
            {
                "video": a.video,
                "panel": a.panel,
                "frames": len(frames),
                "keys": keys,
                "mean_fraction": round(float(masks.mean()), 3),
                "sec": round(time.time() - t0, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
