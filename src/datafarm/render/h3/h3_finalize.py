#!/usr/bin/env python3
"""Finish one rendered episode: seed choice -> assembly -> frozen EVAL-v1 score -> first-chunk exposure lock -> score.

  python h3_finalize.py select --render-recipe <name> --task <task> --episode-tag <tag> --name <selection name> \
      --choice-file <choice.json> --seeds s0 s1 ... [--run-dir <run>]
  python h3_finalize.py lock --package <assembled package dir> --name <selection name> [--run-dir <run>]

`select` builds <run>/recipes/<name>.json and <run>/pkgs/<name>/<task>/<tag> from the per-chunk choice (a separate
producer-side VLM screen; h3_screen_prep.py), assembles it with min-diff seams, scores it, then applies `lock`.
`lock` writes <run>/pkgs/exposure-lock-<name>/<task>/<tag> and scores it.  Acceptance stays the independent blind review.
Relative paths resolve against the run directory.  Run with H3_PYTHON (numpy, opencv); bindings: H3_RUN, H3_EVAL_DINO.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h3_paths
from h3_assemble import assemble
from h3_exposure_lock import lock_package
from h3_score_and_pack import score
from h3_select import select_seeds


def score_package(pkg_dir: Path, task: str, run: Path) -> None:
    assembled = json.loads((pkg_dir / "assembled.json").read_text())
    score(assembled["final"], assembled["carrier_clip"], pkg_dir / "score.json", task, run)


def lock(package: Path, name: str, run: Path) -> Path:
    lock_name = f"exposure-lock-{name}"
    recipe_path = run / "recipes" / f"{lock_name}.json"
    if not recipe_path.exists():
        recipe_path.write_text(json.dumps({
            "name": lock_name, "parent": name,
            "single_variable": "output-side first-chunk photometric anchoring of every chunk (h3_exposure_lock.py) "
                               "before hard-cut assembly at the recorded seams; gains only, no geometry/timing change"},
            indent=2))
    dst = lock_package(package, json.loads(recipe_path.read_text()), run)
    score_package(dst, package.parts[-2], run)
    return dst


def select(args, run: Path) -> Path:
    choice_file = run / args.choice_file
    screener = json.loads(choice_file.read_text())["screener"]
    recipe = json.loads((run / "recipes" / f"{args.render_recipe}.json").read_text())
    recipe.update(name=args.name, parent=args.render_recipe, render_from=args.render_recipe,
                  selection={"mode": "screen", "seeds": "available offsets",
                             "screen": f"{screener} seed screen (h3_screen_prep.py composites + choice.json, "
                                       "h3.seed-screen/1)"},
                  single_variable=f"per-chunk seed choice among {args.render_recipe} seeds by a separate producer-side "
                                  "VLM screen (content errors vs the carrier); acceptance stays the independent blind "
                                  f"review; assembly = {args.render_recipe}")
    (run / "recipes" / f"{args.name}.json").write_text(json.dumps(recipe, indent=2))
    base = run / "pkgs" / args.render_recipe / args.task / args.episode_tag
    cands = [base if s == "s0" else base.with_name(f"{base.name}-{s}") for s in args.seeds]
    dst = run / "pkgs" / args.name / args.task / args.episode_tag
    if dst.exists():
        shutil.rmtree(dst)
    select_seeds(cands, recipe, choice_file, run)
    assemble(dst)
    score_package(dst, args.task, run)
    return lock(dst, args.name, run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--render-recipe", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--episode-tag", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--choice-file", required=True, type=Path)
    s.add_argument("--seeds", nargs="+", required=True, help="seed suffixes of the candidates: s0 s1 ...")
    h3_paths.add_run_dir_arg(s)
    k = sub.add_parser("lock")
    k.add_argument("--package", required=True, type=Path)
    k.add_argument("--name", required=True)
    h3_paths.add_run_dir_arg(k)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    h3_paths.eval_dino()  # fail closed before any work
    final = select(args, run) if args.cmd == "select" else lock((run / args.package).resolve(), args.name, run)
    print(json.dumps({"final_package": str(final)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
