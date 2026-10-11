"""The directory RoboDojo's evaluation client runs from when PyRUA_Lean reads depth.

What an observation holds is the client's configuration: ``env_cfg/<env_cfg_type>.yml`` sends
three colour images and the robot's state, and a depth image and each camera's matrices where
``observation.vision`` switches them on (with the ``distance_to_image_plane`` capture of the
camera file).  RoboDojo ships with them off.  PyRUA_Lean locates objects from the depth image,
so its ``setup_eval_env_client.sh`` starts the client from a root of its own: RoboDojo's code
and configuration copied (a few megabytes), those switches on in the copy, ``Assets`` and
``eval_result`` linked back.  Nothing in the RoboDojo checkout is edited, and results land
where they always do.

    python client_root.py <RoboDojo root> <new root> [--env-cfg arx_x5] [--deploy deploy.yml]

prints the root to start the client from.  ``observation: rgb`` in ``deploy.yml`` leaves the
client where it is: the RoboDojo root is printed and nothing is built (the client's default
observation; the policy then works from the colour images alone).  Standard library and
PyYAML only: this runs with the simulator's interpreter.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

#: What the evaluation client imports and reads from RoboDojo's tree.
SOURCE_PARTS = ("env", "src", "task", "utils", "scripts", "env_cfg")
#: The cameras depth is rendered for, and the capture group they share.
DEPTH_CAPTURES = ("common", "cam_head", "cam_left_wrist", "cam_right_wrist")
POLICY_DIR = Path(__file__).resolve().parent
OBSERVATIONS = ("rgbd", "rgb")
_SKIP = shutil.ignore_patterns("__pycache__", "*.pyc")


def with_depth(env_cfg: Path, env_cfg_type: str) -> None:
    """Switch the depth image and the camera matrices on in a copy of ``env_cfg``."""

    import yaml

    top = env_cfg / f"{env_cfg_type}.yml"
    cfg = yaml.safe_load(top.read_text(encoding="utf-8"))
    vision = cfg.setdefault("observation", {}).setdefault("vision", {})
    vision.update(depth=True, intrinsic_matrix=True, extrinsic_matrix=True)
    top.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    camera_file = env_cfg / "camera" / f"{cfg['config']['camera']}.yml"
    camera = yaml.safe_load(camera_file.read_text(encoding="utf-8"))
    annotators = camera.setdefault("annotator", {})
    for name in DEPTH_CAPTURES:
        annotators.setdefault(name, {})["distance_to_image_plane_capture"] = {
            "type": "distance_to_image_plane",
            "device": "cpu",
        }
    camera_file.write_text(yaml.safe_dump(camera, sort_keys=False), encoding="utf-8")


def _assets(bench: Path) -> Path | None:
    """RoboDojo's ``Assets``: in the checkout, or where a packaged runtime is told to look."""

    for root in (bench, Path(os.environ.get("ROBODOJO_ASSETS_ROOT") or bench)):
        if (root / "Assets").is_dir():
            return (root / "Assets").resolve()
    return None


def lay_out(
    root: Path,
    bench: Path,
    *,
    depth: bool = True,
    env_cfg_type: str = "arx_x5",
    xpolicylab: Path | None = None,
    policy_dir: Path | None = None,
    results: Path | None = None,
) -> Path:
    """Build ``root`` from the RoboDojo checkout ``bench`` and return it.

    ``xpolicylab`` is the XPolicyLab checkout (default ``bench/XPolicyLab``): its own code is
    copied, and of its policies only this one (``policy_dir``, default this file's
    directory).  ``results`` is where ``eval_result`` points (default: a directory of
    ``root``'s own).  Copies, not links: RoboDojo and XPolicyLab find ``env_cfg`` and
    ``Assets`` by ``..`` from their own files, which from behind a link is the checkout.
    """

    root, bench = Path(root), Path(bench).resolve()
    xpolicylab = Path(xpolicylab).resolve() if xpolicylab else bench / "XPolicyLab"
    policy_dir = Path(policy_dir).resolve() if policy_dir else POLICY_DIR
    root.mkdir(parents=True, exist_ok=True)
    for part in SOURCE_PARTS:
        if not (root / part).exists():
            shutil.copytree(bench / part, root / part, ignore=_SKIP)
    if depth:
        with_depth(root / "env_cfg", env_cfg_type)
    assets = _assets(bench)
    if assets is not None and not (root / "Assets").exists():
        (root / "Assets").symlink_to(assets, target_is_directory=True)
    if results is not None and not (root / "eval_result").exists():
        Path(results).mkdir(parents=True, exist_ok=True)
        (root / "eval_result").symlink_to(Path(results).resolve(), target_is_directory=True)
    xpl = root / "XPolicyLab"
    xpl.mkdir(exist_ok=True)
    for entry in sorted(xpolicylab.iterdir()):
        if entry.name == "policy" or entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        copy = xpl / entry.name
        if copy.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, copy, ignore=_SKIP)
        else:
            shutil.copy2(entry, copy)
    policies = xpl / "policy"
    policies.mkdir(exist_ok=True)
    for entry in sorted((xpolicylab / "policy").iterdir()):
        if entry.is_file() and not (policies / entry.name).exists():  # __init__.py, lib.sh
            shutil.copy2(entry, policies / entry.name)
    if not (policies / policy_dir.name).exists():
        shutil.copytree(policy_dir, policies / policy_dir.name, ignore=_SKIP)
    return root


def observation(deploy_yml: Path | None) -> str:
    """``observation`` of the adapter's ``deploy.yml``: ``rgbd`` (the default) or ``rgb``."""

    if deploy_yml is None or not Path(deploy_yml).is_file():
        return "rgbd"
    import yaml

    cfg = yaml.safe_load(Path(deploy_yml).read_text(encoding="utf-8")) or {}
    value = str(cfg.get("observation") or "rgbd").lower()
    if value not in OBSERVATIONS:
        raise SystemExit(f"{deploy_yml}: observation must be one of {OBSERVATIONS}, got {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("bench", type=Path, help="the RoboDojo checkout (XPolicyLab inside)")
    parser.add_argument("root", type=Path, help="the directory to build")
    parser.add_argument("--env-cfg", default="arx_x5", help="env_cfg stem (default: arx_x5)")
    parser.add_argument(
        "--deploy", type=Path, default=POLICY_DIR / "deploy.yml", help="the adapter's deploy.yml"
    )
    args = parser.parse_args(argv)
    if observation(args.deploy) == "rgb":
        print(args.bench.resolve())
        return 0
    root = lay_out(
        args.root,
        args.bench,
        depth=True,
        env_cfg_type=args.env_cfg,
        results=args.bench / "eval_result",
    )
    print(root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
