#!/usr/bin/env python3
"""Render one SolarRenderer window (run with SOLAR_PYTHON on one GPU).

    SOLAR_PYTHON solar_window.py --clip-dir CLIP --out-dir OUT --name NAME [--seed 0] [--mode bidir|causal] [--decode-only]

CLIP holds structure.mp4 (solar_clip.py), prompt.txt and optionally first.png (canvas-sized conditioning frame). The
encode, LoRA merge and sampling are those of SolarRenderer scripts/render.py (5-step shifted schedule, 4 evaluations
with renoise for bidir; block-causal stream with exit index 3 for causal). The latents are saved before the decode, the
transformer is released before the VAE decode, and the decode is converted to uint8 in 48-frame slices, so a ~450-frame
window fits one 141 GB GPU and a ~530-frame window one 180 GB GPU.

Output: OUT/NAME-s<seed>.mp4 (1344x768, 24 fps label, frame k <-> structure frame k, padding cut) and
OUT/NAME-s<seed>.run.json (timings, peak memory, frame counts, LoRA pair). --decode-only re-decodes saved latents.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solar_paths  # noqa: E402


def gb(x) -> float:
    return round(x / 2**30, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--mode", choices=["causal", "bidir"], default="bidir")
    a = ap.parse_args(argv)

    sr = solar_paths.solar_repo()
    h3 = str(solar_paths.h3_weights())
    control, fewstep = solar_paths.lora_pair(a.mode)
    sys.path[:0] = [str(sr / "scripts"), str(sr / "src"), str(sr / "tools")]
    import render  # SolarRenderer scripts/render.py
    import torch

    d, out = Path(a.clip_dir).resolve(), Path(a.out_dir).resolve()   # SolarRenderer reads the member relative to "/"
    out.mkdir(parents=True, exist_ok=True)
    lat_path = out / f"{a.name}-s{a.seed}.latents.pt"
    mp4 = out / f"{a.name}-s{a.seed}.mp4"
    rec_path = out / f"{a.name}-s{a.seed}.run.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    rec.update({"clip": a.name, "seed": a.seed, "host": os.uname().nodename, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "mode": a.mode, "pair": [control.stem, fewstep.stem]})

    def save():
        rec_path.write_text(json.dumps(rec, indent=1))

    t0 = time.perf_counter()
    dev = torch.device("cuda:0")
    if not a.decode_only:
        st = str(d / "structure.mp4")
        first = d / "first.png"
        with tempfile.TemporaryDirectory(dir=str(out)) as work:
            member = f"{work}/{a.name}.safetensors"
            os.environ["OMNI_MAX_SECONDS"] = "600"
            enc = render.load_encoder(h3, "cuda:0")
            pre, _ = enc
            held = int(pre.frame_count(st) * render.FPS / pre.frame_rate(st))
            frames, keep = render.clip_frames(held / render.FPS, held)
            b = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            render.encode_sample(enc, render.sample_of(a.name, (d / "prompt.txt").read_text(), st,
                                                       str(first) if first.exists() else None), member, frames)
            rec.update({"held": held, "frames": frames, "keep": keep, "encode_s": round(time.perf_counter() - b, 1),
                        "encode_peak_alloc_gb": gb(torch.cuda.max_memory_allocated())})
            save()
            del enc, pre, _
            render.free_cuda()
            b = time.perf_counter()
            core, conditioners, device = render.load_renderer(h3, str(control), str(fewstep), "cuda:0",
                                                              causal=a.mode == "causal", frames=frames)
            rec["renderer_load_s"] = round(time.perf_counter() - b, 1)
            save()
            from solarwm.backends.minimax_h3.flow import make_shifted_schedule
            from solarwm.backends.minimax_h3.omni_data import H3OmniStream
            stream = object.__new__(H3OmniStream)
            stream.root, stream.current_plan_fingerprint, stream.aligned_control = "/", "render", False
            batch = stream.read({"id": a.name, "member": member})
            torch.cuda.reset_peak_memory_stats()
            b = time.perf_counter()
            with torch.no_grad():
                torch.manual_seed(a.seed)
                if a.mode == "causal":
                    latents, *_rest = core.stream(batch, generator=torch.Generator(device=device).manual_seed(a.seed),
                                                  schedule=make_shifted_schedule(5, shift=12.0), exit_index=3,
                                                  cache_structure=False)
                else:  # every frame at once, as render.render_member does (4 evaluations, renoise)
                    latents = core.generate(batch, noise_seed=a.seed, num_inference_steps=5, renoise=True)
            rec.update({"sample_s": round(time.perf_counter() - b, 1),
                        "sample_peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                        "sample_peak_reserved_gb": gb(torch.cuda.max_memory_reserved()), "latent_shape": list(latents.shape)})
            torch.save(latents.detach().cpu(), lat_path)
            save()
            vae = conditioners.video_vae
            del core, batch, stream, latents
            gc.collect()
            torch.cuda.empty_cache()
    else:
        from solarwm.backends.minimax_h3.optional import load_conditioners
        model_cfg = {"checkpoint_path": h3, "transformer_subfolder": "transformer_ref", "torch_dtype": "bfloat16",
                     "attention_backend": "flash", "adapter": {"rank": render.RANK, "alpha": render.ALPHA, "dropout": 0.0}}
        vae = load_conditioners(model_cfg, device=dev, qwen=False, video_vae=True, audio_vae=False,
                                schedulers=False).video_vae

    # decode: the normalisation of solarwm inference.decode_h3_latents, uint8 conversion in slices
    latents = torch.load(lat_path)
    keep = rec.get("keep") or 0
    torch.cuda.reset_peak_memory_stats()
    b = time.perf_counter()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        decoded = vae.decode(latents.to(dev, torch.float32) * std + mean, return_dict=False)[0]
    pm = torch.tensor((0.485, 0.456, 0.406), device=dev).view(-1, 1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225), device=dev).view(-1, 1, 1, 1)
    T, H, W = decoded.shape[2], decoded.shape[3], decoded.shape[4]
    n = min(T, keep) if keep else T
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt",
                           "yuv420p", str(mp4) + ".tmp.mp4"], stdin=subprocess.PIPE)
    for s in range(0, n, 48):
        sl = decoded[0, :, s:min(n, s + 48)].float()
        sl = (sl * ps + pm).clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu().numpy()
        ff.stdin.write(sl.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("solar_window: ffmpeg failed")
    os.replace(str(mp4) + ".tmp.mp4", mp4)
    rec.update({"decode_s": round(time.perf_counter() - b, 1), "decode_peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                "decoded_frames": T, "written_frames": n, "out": str(mp4), "wall_s": round(time.perf_counter() - t0, 1)})
    save()
    print("DONE", json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
