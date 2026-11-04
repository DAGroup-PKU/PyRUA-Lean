#!/usr/bin/env python3
"""Resident Ref2VA + H3-Fun ControlNet (soft-HED) + inpaint hybrid worker, one process on one GPU.

Renders packages built by h3_package.py.  It loads the VideoX-Fun MiniMaxH3ControlPipeline once (Ref2VA base
transformer + MiniMax-H3-Fun-Controlnet-Union branch), claims <queue>/pending/<profile>/*.json by atomic rename into
<queue>/running, and renders each package's chunks in order:

  * render-policy gate first: datafarm.render.policy.check_h3_payload(manifest, contract)
    (prompt file == compiled contract prompt, contract real reference images first), plus the
    control-input audit (files, sha256, canvas, legal 17n+5 shapes);
  * Ref2VA image references = the contract real references (images-first);
  * control rows = VAE-encoded soft-HED chunk clip at control_context_scale, inpaint rows = white
    chunk clip behind the active-panel repaint mask; both cover generated target rows only;
  * chunks render independently; output: <output_root>/chunks/<chunk_id>/chunk-render.mp4 (full 1024x768 guttered
    canvas, 24 fps, video only) + segment-manifest.json, so `h3_assemble.py <pkg>` applies.

Exits after --idle-exit seconds without work or when <queue>/STOP exists, releasing the lease.

Deployment bindings (h3_paths.py, no built-in machine paths): H3_VIDEOX_FUN (VideoX-Fun checkout with
vendor/videox_fun applied; required at start), H3_MODELS (default --base-model/--control-checkpoint),
--turbo-lora from H3_TURBO_LORA in launch_control_worker.sh.  Arguments are parsed before torch/VideoX-Fun are imported,
so `--help` needs neither a GPU nor the bindings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import h3_package
import h3_paths
from h3_media import sha256

h3_paths.datafarm_src()
from datafarm.common import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="resident MiniMax-H3 Ref2VA + ControlNet-Union hybrid queue worker")
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--profile", choices=list(h3_package.PROFILES), default="control-turbo")
    ap.add_argument("--idle-exit", type=int, default=900)
    ap.add_argument("--max-tickets", type=int, default=10**6)
    ap.add_argument("--memory-mode", choices=["full", "group_offload", "cpu_offload"], default="full")
    ap.add_argument("--base-model", default=None, help="default: ${H3_MODELS}/MiniMax-H3/Ref2VA")
    ap.add_argument("--control-checkpoint", default=None,
                    help="default: ${H3_MODELS}/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors")
    ap.add_argument("--control-config", default=None,
                    help="default: ${H3_VIDEOX_FUN}/config/minimax_h3/minimax_h3_control.yaml")
    ap.add_argument("--turbo-lora", default=None, help="fuse the Sol-H3 four-step Ref2VA LoRA (turbo profiles only)")
    return ap


# Parse before the runtime imports below: --help and argument errors need no torch, GPU or deployment binding.
ARGS = build_parser().parse_args() if __name__ == "__main__" else None


def default(key: str) -> str:
    """Model/repo default from the deployment bindings (fails closed when the binding is absent)."""
    return h3_paths.default_model_path(key)


VIDEOX_FUN_REPO = str(h3_paths.videox_fun_repo())
os.environ.setdefault("VIDEOX_ATTENTION_TYPE", "SDPA")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, VIDEOX_FUN_REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from videox_fun.models import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,  # noqa: E402
                               MiniMaxH3ControlTransformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
from videox_fun.pipeline import (MiniMaxH3ControlPipeline, MiniMaxH3ImageReference,  # noqa: E402
                                 MiniMaxH3VideoReference)
from videox_fun.utils import MiniMaxH3Scheduler  # noqa: E402
from videox_fun.utils.utils import get_video_to_video_latent  # noqa: E402

H, W, FPS = h3_package.SPATIAL.canvas_height, h3_package.SPATIAL.canvas_width, 24
OUTPUT_NAME = "chunk-render.mp4"  # the name h3_assemble.py reads


def write_json(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload, sort_keys=False, mode=0o644, fsync=False)


def videox_provenance(repo: Path) -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout
    diff = git("diff", "HEAD")
    return {"repo": str(repo), "head": git("rev-parse", "HEAD").strip(),
            "dirty_files": [ln[3:] for ln in git("status", "--porcelain", "--untracked-files=no").splitlines()],
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


class HybridEngine:
    """VideoX-Fun MiniMaxH3ControlPipeline, loaded once (same construction as predict_v2v_control.py)."""

    def __init__(self, base_model: str, control_checkpoint: str, control_config: str, memory_mode: str, log,
                 turbo_lora: str | None = None):
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        t0 = time.perf_counter()
        self.base_model, self.control_checkpoint = base_model, control_checkpoint
        self.memory_mode = memory_mode
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        kwargs = OmegaConf.to_container(OmegaConf.load(control_config)["transformer_additional_kwargs"], resolve=True)
        transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
            base_model, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, **kwargs)
        state = load_file(control_checkpoint)
        state = state.get("state_dict", state)
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"control checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}")
        self.control_tensors = len(state)
        del state
        self.turbo = None
        vae = AutoencoderKLMiniMaxH3.from_pretrained(base_model, subfolder="vae", low_cpu_mem_usage=True,
                                                     torch_dtype=torch.bfloat16)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(base_model, subfolder="audio_vae",
                                                                low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(base_model, "tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(os.path.join(base_model, "processor"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(base_model, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
        self.pipe = MiniMaxH3ControlPipeline(
            vae=vae, audio_vae=audio_vae, text_encoder=text_encoder, tokenizer=tokenizer, processor=processor,
            transformer=transformer, scheduler=MiniMaxH3Scheduler.from_pretrained(base_model, subfolder="scheduler"),
            audio_scheduler=MiniMaxH3Scheduler.from_pretrained(base_model, subfolder="audio_scheduler"))
        t_cpu = time.perf_counter() - t0
        if memory_mode == "full":
            self.pipe.to(device=self.device)
        elif memory_mode == "group_offload":  # single-GPU offload (~7.7 s/step)
            from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
            register_auto_device_hook(self.pipe.transformer)
            safe_enable_group_offload(self.pipe, onload_device=self.device, offload_device="cpu",
                                      offload_type="leaf_level", use_stream=True)
        elif memory_mode == "cpu_offload":
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            raise ValueError(memory_mode)
        if turbo_lora:  # Sol-H3 four-step Ref2VA adapter fused into the resident main transformer (alpha 8, scale 1, as Sol)
            if memory_mode != "full":
                raise ValueError("--turbo-lora needs memory_mode=full (the fuser runs on the GPU-resident transformer)")
            from h3_lora import fuse_lora
            rep = fuse_lora(self.pipe.transformer, turbo_lora, alpha=8, scale=1.0)
            self.turbo = {"path": str(turbo_lora), "pairs": rep.pairs, "rank": rep.rank, "alpha": rep.alpha,
                          "effective_scale": rep.effective_scale, "fuse_s": rep.fuse_s}
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.load_info = {"load_seconds": round(self.load_seconds, 1), "cpu_load_seconds": round(t_cpu, 1),
                          "memory_mode": memory_mode,
                          "control_tensors": self.control_tensors,
                          "missing_keys_after_control_load": len(missing),
                          "gpu_allocated_gb_after_load": round(torch.cuda.memory_allocated(self.device) / 2**30, 2),
                          "gpu_name": torch.cuda.get_device_name(self.device),
                          "gpu_total_gb": round(torch.cuda.get_device_properties(self.device).total_memory / 2**30, 1)}
        log(f"loaded {json.dumps(self.load_info)}")

    # -- inputs -------------------------------------------------------------------------------------------
    @staticmethod
    def read_video(path: str, frames: int) -> torch.Tensor:
        video, _, _, _ = get_video_to_video_latent(str(path), video_length=frames, sample_size=[H, W], fps=FPS,
                                                   ref_image=None, keep_aspect_ratio=True)
        if video.shape[2] < frames or tuple(video.shape[-2:]) != (H, W):
            raise ValueError(f"{path}: decoded {tuple(video.shape)}, need {frames} frames at {H}x{W}")
        return video[:, :, :frames]

    # -- generation ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, job: dict, prompt: str, white_video_reference) -> tuple[np.ndarray, dict]:
        n = int(job["generated_frames"])
        control = self.read_video(job["control_video"], n)
        inpaint = self.read_video(job["inpaint_video"], n)
        mask = (self.read_video(job["inpaint_mask"], n)[:, :1] > 0.5).to(inpaint.dtype)
        references = [MiniMaxH3ImageReference.from_file(p) for p in job["reference_images"]]
        if white_video_reference:  # the chunk's white carrier as a Ref2VA video reference, for task semantics
            ref = MiniMaxH3VideoReference.from_file(job["white"])
            if not isinstance(white_video_reference, bool):
                # int = evenly decimated to that many frames; the full clip doubles the sequence length and can
                # exhaust the GPU memory of a worker that fits the recipe's minimum free memory
                fr = np.asarray(ref.frames)
                idx = np.linspace(0, len(fr) - 1, int(white_video_reference)).round().astype(int)
                ref = MiniMaxH3VideoReference(frames=fr[idx], fps=float(ref.fps))
            references.append(ref)
        stats = {"control_nonzero": float((control > 1 / 255).float().mean()),
                 "control_mean": float(control.mean()),
                 "mask_regenerate_fraction": float(mask.mean())}
        step_t = []

        def on_step(_pipe, i, _t, _kw):
            torch.cuda.synchronize()
            step_t.append(time.perf_counter())
            return {}

        torch.cuda.reset_peak_memory_stats(self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(job["seed"]))
        t0 = time.perf_counter()
        out = self.pipe(prompt=prompt, references=references,
                        control_video=control, control_context_scale=float(job["control_context_scale"]),
                        mask_video=mask, inpaint_video=inpaint, height=H, width=W, num_frames=n,
                        num_inference_steps=int(job["steps"]), guidance_scale=float(job["guidance_scale"]),
                        generator=generator, output_type="pt", callback_on_step_end=on_step)
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        video = out.videos
        if video.ndim != 5:
            raise RuntimeError(f"unexpected output shape {tuple(video.shape)}")
        frames = video[0] if video.shape[2] == 3 else video[0].permute(1, 0, 2, 3)  # -> T,C,H,W
        if tuple(frames.shape) != (n, 3, H, W):
            raise RuntimeError(f"output {tuple(frames.shape)} != ({n},3,{H},{W})")
        rgb = frames.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        diffs = np.diff(np.array([t0] + step_t))
        stats.update({"generation_seconds": round(gen_s, 2), "steps_run": len(step_t),
                      "first_step_seconds": round(float(diffs[0]), 2) if len(diffs) else None,
                      "step_seconds_mean": round(float(diffs[1:].mean()), 3) if len(diffs) > 1 else None,
                      "post_denoise_seconds": round(time.perf_counter() - step_t[-1], 2) if step_t else None,
                      "peak_allocated_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2),
                      "peak_reserved_gb": round(torch.cuda.max_memory_reserved(self.device) / 2**30, 2)})
        del out, video, frames, control, inpaint, mask, references
        return rgb, stats


def encode_mp4(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.mp4")
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", str(tmp)],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(rgb).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")
    tmp.replace(path)


def render_ticket(engine: HybridEngine, ticket: dict, prov: dict, log) -> dict:
    manifest_path = Path(ticket["manifest"])
    payload = json.loads(manifest_path.read_text())
    os.environ["DATAFARM_RENDER_CONTRACT"] = ticket["contract"]
    policy = h3_package.audit(Path(ticket["package"]))  # check_h3_payload + control-input audit
    ctl = payload["control"]
    for key in ("base_model", "control_checkpoint"):
        loaded = getattr(engine, key)
        if str(Path(ctl.get(key) or default(key))) != str(Path(loaded)):
            raise ValueError(f"ticket wants {key}={ctl.get(key)} but this worker loaded {loaded}")
    out_root = Path(ticket["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "policy-receipt.json", policy)
    prompt_file = Path(payload["prompt"])
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    completed = []
    for job in sorted(payload["jobs"], key=lambda j: int(j["chunk_index"])):
        chunk_root = out_root / "chunks" / str(job["chunk_id"])
        output, seg = chunk_root / OUTPUT_NAME, chunk_root / "segment-manifest.json"
        if output.exists() and seg.exists():
            completed.append(str(job["chunk_id"]))
            continue  # resume
        t0 = time.perf_counter()
        job_prompt = prompt
        if job.get("prompt") and Path(job["prompt"]) != prompt_file:  # chunk-local lowering (gate-checked above)
            job_prompt = Path(job["prompt"]).read_text(encoding="utf-8").strip()
        rgb, stats = engine.generate(job, job_prompt, ctl.get("white_video_reference"))
        encode_mp4(rgb, output)
        write_json(seg, {
            "schema_version": "h3_control_chunk/1", "status": "complete", "backend": "control-hybrid",
            "source_manifest": str(manifest_path), "chunk_id": str(job["chunk_id"]),
            "white": job["white"], "control_video": job["control_video"], "inpaint_video": job["inpaint_video"],
            "inpaint_mask": job["inpaint_mask"], "prompt_source": str(prompt_file),
            "prompt_sha256": hashlib.sha256(job_prompt.encode()).hexdigest(), "chunk_facts": job.get("chunk_facts"),
            "reference_images": job["reference_images"], "seed": int(job["seed"]), "steps": int(job["steps"]),
            "denoise_forwards": int(job["steps"]), "guidance_scale": float(job["guidance_scale"]),
            "control_context_scale": float(job["control_context_scale"]),
            **stats, "wall_seconds": round(time.perf_counter() - t0, 2),
            "width": W, "height": H, "frames": int(job["generated_frames"]),
            "base_model": engine.base_model, "control_checkpoint": engine.control_checkpoint,
            "memory_mode": engine.memory_mode, "videox_fun": prov, "output": str(output),
            "output_sha256": sha256(output)})
        log(f"chunk {job['chunk_id']} done in {time.perf_counter() - t0:.1f}s (gen {stats['generation_seconds']}s, "
            f"{stats['steps_run']} steps, {stats['step_seconds_mean']} s/step, peak {stats['peak_allocated_gb']} GB)")
        del rgb
        gc.collect()
        torch.cuda.empty_cache()
        completed.append(str(job["chunk_id"]))
    return {"completed": completed, "policy": policy}


def claim(queue: Path, profile: str, tag: str) -> Path | None:
    """First claimable ticket of this profile."""
    running = queue / "running"
    running.mkdir(parents=True, exist_ok=True)
    for t in sorted((queue / "pending" / profile).glob("*.json")):
        target = running / f"{t.stem}@{tag}.json"
        try:
            os.rename(t, target)
            return target
        except OSError:
            continue
    return None


def main() -> int:
    args = ARGS if ARGS is not None else build_parser().parse_args()
    args.base_model = args.base_model or default("base_model")
    args.control_checkpoint = args.control_checkpoint or default("control_checkpoint")
    args.control_config = args.control_config or default("control_config")
    # <host>-<pid>-<profile>: h3_requeue_orphans.py parses the host and pid from this tag and matches them to the
    # live leases; never append anything after the profile.
    tag = f"{socket.gethostname()}-{os.getpid()}-{args.profile}"
    logf = args.queue / "logs" / f"{tag}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with logf.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
        print(msg, flush=True)

    log(f"starting cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} memory_mode={args.memory_mode}")
    prov = videox_provenance(Path(VIDEOX_FUN_REPO))
    if bool(args.turbo_lora) != args.profile.endswith("-turbo"):
        raise SystemExit("turbo profiles (suffix -turbo) need --turbo-lora and only they may fuse it")
    engine = HybridEngine(args.base_model, args.control_checkpoint, args.control_config, args.memory_mode, log,
                          turbo_lora=args.turbo_lora)
    log(f"turbo {json.dumps(engine.turbo)}")
    log(f"videox_fun {json.dumps(prov)}")
    idle_since, done = time.time(), 0
    while done < args.max_tickets:
        if (args.queue / "STOP").exists():
            log("STOP file seen")
            break
        path = claim(args.queue, args.profile, tag)
        if path is None:
            if time.time() - idle_since > args.idle_exit:
                log("idle exit")
                break
            time.sleep(15)
            continue
        ticket = json.loads(path.read_text())
        log(f"claimed {path.name}")
        status, err, res = "done", None, None
        try:
            res = render_ticket(engine, ticket, prov, log)
        except torch.OutOfMemoryError as e:
            # another process may share this GPU: hand the ticket back unchanged, release this lease and let the next
            # worker start on a GPU with enough free memory instead of failing the rest of the queue here
            back = args.queue / "pending" / str(ticket.get("profile") or args.profile)
            back.mkdir(parents=True, exist_ok=True)
            (back / (path.name.split("@")[0] + ".json")).write_text(json.dumps(ticket, indent=2))
            path.unlink(missing_ok=True)
            log(f"OOM on {path.name}: {e!r}; ticket returned to {back.name}, worker exits to free the GPU")
            break
        except Exception as e:  # noqa: BLE001 - keep the resident worker alive
            status, err = "failed", traceback.format_exc()
            print(f"FAILED {path.name}:\n{err}", file=sys.stderr, flush=True)
            log(f"FAILED {path.name}: {e!r}")
            gc.collect()
            torch.cuda.empty_cache()
        dest = args.queue / status
        dest.mkdir(parents=True, exist_ok=True)
        rec = {"ticket": ticket, "worker": tag, "finished": time.time(), "status": status, "error": err,
               "load": engine.load_info, "completed": (res or {}).get("completed")}
        (dest / path.name).write_text(json.dumps(rec, indent=2))
        path.unlink(missing_ok=True)
        done += 1
        idle_since = time.time()
    log("exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
