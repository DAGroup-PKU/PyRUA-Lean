#!/usr/bin/env python3
"""Task bundle -> render intent/overlay -> compiled render contract (sim2real | ood).

A task bundle (<run>/tasks/<task>/bundle.json, schema h3.task-bundle/1) is a TASK ARTIFACT the agent generates for
the current task; see BUNDLE-SCHEMA.md next to this file.  This module builds contracts with the repository's own
render-policy compiler; it never calls a provider.  The mode and any OOD axes come only from the bundle's
`mode_confirmations` (the current user's in-scope instruction, quoted with its source); this module carries no
authorization of its own and fails closed without it.

  python h3_bundle.py check  <run>/tasks/<task>/bundle.json [--run-dir <run>]
  python h3_bundle.py compile <run>/tasks/<task>/bundle.json --mode sim2real \
        [--episode <id>] [--variant <ood-variant-name>] --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h3_carrier
import h3_keep_regions
import h3_paths
from h3_media import probe, sha256

h3_paths.datafarm_src()
from datafarm.render.policy import compile_prompt  # noqa: E402

AREAS = ["robot_body", "attached_cameras_and_cables", "task_objects", "table_and_background",
         "photographic_character"]
SLOTS = ["TASK_DESCRIPTION", "SOURCE_VIDEO_ROLE", "CAMERA_TIMING_CONTRACT", "TASK_EVENT_CONTRACT",
         "CONTINUITY_CONTRACT"]


def load(path: str) -> dict:
    b = json.loads(Path(path).read_text())
    if b.get("schema") != "h3.task-bundle/1":
        raise SystemExit("bundle schema must be h3.task-bundle/1")
    return b


def references_for(bundle: dict, episode: str | None) -> list[dict]:
    per = bundle.get("per_episode_references") or {}
    refs = per.get(episode) if episode and episode in per else bundle["real_references"]
    out = []
    for r in refs:
        r = copy.deepcopy(r)
        r["sha256"] = sha256(Path(r["path"]))
        src = r["provenance"]["source"]
        if isinstance(src, str):  # render-policy binds the provenance source as a hashed file ref
            r["provenance"]["source"] = {"path": src, "sha256": sha256(Path(src))}
        out.append(r)
    return out


def mode_confirmation(bundle: dict, mode: str) -> tuple[dict, list]:
    """The bundle's record of the current user's in-scope mode instruction (quote + source) and, for OOD, the
    authorized axes.  Never defaulted: a missing record stops before any contract is compiled."""
    conf = (bundle.get("mode_confirmations") or {}).get(mode)
    if not isinstance(conf, dict) or not str(conf.get("quote") or "").strip() or not str(conf.get("source") or "").strip():
        raise SystemExit(f"bundle {bundle.get('task')!r} lacks mode_confirmations.{mode} (quote and source of the "
                         f"current user's in-scope instruction); resolve the mode with the user before rendering")
    axes = [str(a) for a in conf.get("authorized_ood_axes") or []]
    if mode == "ood" and not axes:
        raise SystemExit("mode_confirmations.ood needs the user-authorized authorized_ood_axes")
    if mode != "ood" and axes:
        raise SystemExit("sim2real has no authorized OOD axes")
    return conf, axes


def build_intent(bundle: dict, mode: str, episode: str | None) -> dict:
    if not bundle.get("run_id"):
        raise SystemExit("bundle lacks run_id (the run that owns this render)")
    run_id = bundle["run_id"]
    task_id = bundle.get("task_id") or f"{bundle['task']}-h3-control-hybrid"
    conf, axes = mode_confirmation(bundle, mode)
    return {
        "schema": "datafarm.render-intent/1",
        "run_id": run_id,
        "task_id": task_id,
        "mode": mode,
        "real_baseline": bundle["real_baseline"],
        "mode_confirmation": {"run_id": run_id, "task_id": task_id, "mode": mode,
                              "quote": conf["quote"], "source": conf["source"],
                              "authorized_ood_axes": axes},
        "authorized_ood_axes": axes,
        "appearance_views": bundle["appearance_views"],
        "appearance_requirements": bundle["appearance_requirements"],
        "real_references": references_for(bundle, episode),
        "appearance_invariants": bundle["appearance_invariants"],
    }


def build_overlay(bundle: dict, mode: str, variant: dict | None) -> dict:
    overlay = dict(bundle["overlay"])
    if mode == "ood":
        if not variant:
            raise SystemExit("ood compile needs --variant")
        overlay["TASK_DESCRIPTION"] = (overlay["TASK_DESCRIPTION"].rstrip(". ") +
                                       f". OOD appearance for this sample (authorized axes only): tabletop = "
                                       f"{variant['tabletop']}; lighting = {variant['lighting']}")
    return overlay


def compile_contract(bundle: dict, mode: str, episode: str | None, variant_name: str | None, out: Path) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    variant = None
    if mode == "ood":
        variant = next(v for v in bundle["ood_variants"] if v["name"] == variant_name)
    intent = build_intent(bundle, mode, episode)
    overlay = build_overlay(bundle, mode, variant)
    (out / "intent.json").write_text(json.dumps(intent, ensure_ascii=False, indent=2))
    (out / "overlay.json").write_text(json.dumps(overlay, ensure_ascii=False, indent=2))
    contract = compile_prompt(out, intent, overlay, assets=h3_paths.REPO)
    (out / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (out / "prompt.txt").write_text(contract["prompt"])
    return contract


def cmd_check(args) -> int:
    b = load(args.bundle)
    problems = []
    for key in ["task", "run_id", "mode_confirmations", "layout", "appearance_views", "appearance_requirements",
                "real_references", "overlay", "real_baseline", "appearance_invariants", "episodes", "realpool"]:
        if key not in b:
            problems.append(f"missing {key}")
    if set(b.get("overlay", {})) != set(SLOTS):
        problems.append(f"overlay slots must be exactly {SLOTS}")
    if set(b.get("appearance_requirements", {})) != set(AREAS):
        problems.append(f"appearance_requirements must classify {AREAS}")
    episodes = b.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        problems.append("episodes must be a non-empty list")
        episodes = []
    report = {"task": b.get("task"), "episodes": {}}
    warnings = []
    for ep in episodes:
        row = {}
        if not Path(ep["carrier"]).is_file():
            problems.append(f"{ep['id']}: carrier missing {ep['carrier']}")
        else:
            row["carrier"] = probe(ep["carrier"])
            fps = h3_carrier.fps_of(ep["carrier"])
            if fps < h3_carrier.MIN_WHITE_FPS:
                problems.append(f"{ep['id']}: carrier at {fps:g} fps; re-render the white densely (white_replay "
                                "image_stride), never interpolate it")
            elif abs(fps - float(b.get("carrier_fps") or 0)) > 0.5:
                problems.append(f"{ep['id']}: carrier is {fps:g} fps but carrier_fps is {b.get('carrier_fps')}")
            try:
                measured = h3_carrier.check(Path(ep["carrier"]))
                row["room_seam_frames"] = [measured["frames_with_room_seam"], measured["sampled_frames"]]
                warnings += [f"{ep['id']}: {w}" for w in measured["warnings"] if "fps" not in w]
            except Exception as e:  # noqa: BLE001 - a measurement, not a gate
                warnings.append(f"{ep['id']}: carrier check failed: {e}")
        fr = ep.get("frame_range")
        if fr and row.get("carrier") and not (0 <= fr[0] < fr[1] <= row["carrier"].frames):
            problems.append(f"{ep['id']}: frame_range {fr} outside carrier {row['carrier'].frames}")
        if ep.get("keep_regions"):
            kp = Path(ep["keep_regions"])
            if not kp.is_file():
                problems.append(f"{ep['id']}: keep_regions spec missing {kp}")
            else:
                problems += [f"{ep['id']}: keep_regions {p}" for p in
                             h3_keep_regions.validate(json.loads(kp.read_text()))]
        ww = ep.get("wide_wrist")
        if ww is not None:
            pps = (ww or {}).get("principal_points") or {}
            if not (isinstance(ww, dict) and isinstance(ww.get("factor"), (int, float))
                    and 0.3 <= float(ww["factor"]) < 1.0):
                problems.append(f"{ep['id']}: wide_wrist.factor must be in [0.3, 1)")
            if not all(isinstance(pps.get(s), list) and len(pps[s]) == 2 for s in ("left", "right")):
                problems.append(f"{ep['id']}: wide_wrist.principal_points needs left and right [cx, cy] "
                                "(pixels of the 1280x720 wrist camera)")
        report["episodes"][ep["id"]] = row
    for src in b.get("realpool", {}).get("sources", []):
        p = src.get("video") or src.get("image")
        if not p or not Path(p).is_file():
            problems.append(f"realpool source missing {p}")
    modes = [m for m in ("sim2real", "ood") if m in (b.get("mode_confirmations") or {})]
    if not modes:
        problems.append("mode_confirmations must record at least one user-confirmed mode")
    if "ood" in modes and len(b.get("ood_variants", [])) < 2:
        problems.append("need >= 2 ood_variants for a confirmed ood mode")
    if not problems:
        tmp = h3_paths.run_dir() / "tasks" / b["task"] / "_check"
        for mode in modes:
            try:
                c = compile_contract(b, mode, episodes[0]["id"],
                                     b["ood_variants"][0]["name"] if mode == "ood" else None, tmp / mode)
                report[f"{mode}_prompt_chars"] = len(c["prompt"])
            except (ValueError, KeyError, TypeError, OSError, SystemExit) as e:
                problems.append(f"{mode} compile failed: {e}")
    report["problems"] = problems
    report["warnings"] = warnings
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 1 if problems else 0


def cmd_compile(args) -> int:
    b = load(args.bundle)
    c = compile_contract(b, args.mode, args.episode, args.variant, Path(args.out).resolve())
    print(json.dumps({"prompt_sha256": c["prompt_sha256"], "chars": len(c["prompt"])}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("bundle")
    h3_paths.add_run_dir_arg(c)
    k = sub.add_parser("compile")
    k.add_argument("bundle")
    k.add_argument("--mode", choices=["sim2real", "ood"], required=True)
    k.add_argument("--episode")
    k.add_argument("--variant")
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    if getattr(args, "run_dir", None) is not None:
        h3_paths.set_run_dir(args.run_dir)
    return {"check": cmd_check, "compile": cmd_compile}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
