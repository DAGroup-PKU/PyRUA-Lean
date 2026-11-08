"""One RoboCasa365 episode in its own process: ``python robocasa_episode.py REQUEST``.

Runs in the PyRUA-Lean Python (``paths.python``, which also loads SAM3) with the repository's
``src/`` (under ``paths.lean``) on the path (``run.py`` sets both); the kitchen runs in RoboCasa's
own interpreter (``paths.sim_python``).  The robot is ``robocasa_rgb`` (the colour primitives)
with the ``no-vla`` primitive set; the policy is :func:`pyrualean.play.play_episode`'s cell agent
with no skill library, as the PyRUA-Lean paper's RoboCasa365 runs, or, in an evaluation of
several libraries (``arms``), the library ``run.py init`` copied for the episode's task.
:func:`identity` hashes what must not change during an evaluation (the policy code without
``learn/`` and ``evolve/``, the Python sources of robocasa and robosuite in the kitchen's
interpreter, SAM3, the execution settings); an episode refuses to start when it differs from the
one ``run.py init`` recorded.

A task of the evaluation is a RoboCasa task name (``OpenDrawer``), or ``<arm>/<task>`` with
``arms``, and its layout ``n`` is the scene seed ``seeds_of(config, task)[n]``.  A config with
``replay_dir`` runs those saved cells instead of the model (a check of the whole chain at no
model cost).  ``result.json`` gains ``identity``, ``seed`` and ``arm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: The package directories the hash leaves out, as ``eval/robodojo/episode.py`` does.
NOT_POLICY = ("learn", "evolve", "__pycache__")
#: The kitchen's packages whose Python sources the identity covers.
SIM_PACKAGES = ("robocasa", "robosuite")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def tree_hash(root: Path, pattern: str = "*") -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"no directory {root}")
    return json_hash(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob(pattern))
            if path.is_file() and "__pycache__" not in path.parts
        }
    )


def policy_hash(package: Path) -> str:
    return json_hash(
        {
            str(path.relative_to(package)): tree_hash(path)
            if path.is_dir()
            else json_hash(path.read_text(errors="replace"))
            for path in sorted(package.iterdir())
            if path.name not in NOT_POLICY
        }
    )


def identity(config: dict[str, Any]) -> str:
    """The execution identity of ``config`` with the code and benchmark it names."""

    paths = config["paths"]
    return json_hash(
        {
            "execution": config["execution"],
            "policy": policy_hash(Path(paths["lean"]) / "src" / "pyrualean"),
            "kitchen": {
                name: tree_hash(Path(paths["sim_packages"]) / name, "*.py") for name in SIM_PACKAGES
            },
            "sam3": config["sam3_sha256"],
        }
    )


def split_task(task: str) -> tuple[str | None, str]:
    """An evaluation task as (arm, RoboCasa task): ``OpenDrawer`` is (None, ``OpenDrawer``),
    ``off10/OpenDrawer`` is (``off10``, ``OpenDrawer``)."""

    arm, _, name = task.rpartition("/")
    return arm or None, name


def library_sha256(view: Path) -> str:
    """The library hash ``pyrualean.play.copy_library`` records for the directory ``view``."""

    files = {
        f"{kind}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for kind, pattern in (("skills", "*.py"), ("lessons", "*.md"))
        for path in (sorted((view / kind).glob(pattern)) if (view / kind).is_dir() else [])
    }
    return hashlib.sha256(
        "".join(f"{name}:{sha}\n" for name, sha in sorted(files.items())).encode("utf-8")
    ).hexdigest()


def seeds_of(config: dict[str, Any], task: str) -> list[int]:
    """The scene seeds of ``task`` (a RoboCasa task, or ``<arm>/<task>``): ``seed_overrides[task]``
    or ``seeds``, only those in ``only_seeds[task]`` when that is given (a retest of some cells),
    in the same order."""

    task = split_task(task)[1]
    seeds = [int(s) for s in (config.get("seed_overrides") or {}).get(task, config["seeds"])]
    only = (config.get("only_seeds") or {}).get(task)
    return seeds if only is None else [s for s in seeds if s in set(only)]


def run(request: dict[str, Any]) -> dict[str, Any]:
    from pyrualean.hosts.robocasa_rgb import Cell
    from pyrualean.play import play_episode

    config = request["config"]
    if identity(config) != request["identity"]:
        raise RuntimeError("code, benchmark or execution settings changed since init")
    settings = config["execution"]
    task = request["task"]
    arm, name = split_task(task)
    if name not in config["task_names"] or (arm is None) != (not config.get("arms")):
        raise RuntimeError(f"{task}: not a task of this evaluation")
    seed = seeds_of(config, task)[int(request["layout"])]
    output = Path(request["output"]) / "episode"
    output.mkdir(parents=True, exist_ok=False)
    # a replay root (``replay_dir`` in its config) runs saved cells instead of the model
    replay = config.get("replay_dir")
    result = play_episode(
        # the physical GPU: the kitchen renders on it (EGL) and SAM3 pins it
        Cell(
            settings["split"],
            name,
            seed,
            int(settings["max_episode_steps"]),
            cuda_device=request["gpu"],
        ),
        output,
        arm="cells",
        max_attempts=1,
        model=settings["model"] if not replay else "replay",
        reasoning_effort=settings["extra_body"].get("reasoning_effort", "high"),
        budget_s=settings["policy_seconds"],
        max_turns=settings["max_cells"],
        runtime="replay" if replay else settings.get("runtime", "openai"),
        codex_bin=settings.get("codex_bin", "codex"),
        codex_home=settings.get("codex_home"),
        replay_dir=replay,
        robot=settings["robot"],
        primitives=settings["primitives"],
        images=settings["images"],
        feedback=settings["feedback"],
        library=request["view"] if arm is not None else None,
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        boot_options={
            "sim_python": config["paths"]["sim_python"],
            "robocasa_assets_path": config["paths"]["assets"],
        },
        openai_options={
            "max_requests": settings["max_model_requests"],
            "max_tokens": settings["max_tokens"],
            "nudges": settings["nudges"],
            "extra_body": settings["extra_body"],
            "retries": settings["provider_retries"],
        },
    )
    result.update(identity=request["identity"], seed=seed, arm=arm)
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False) + "\n")
    temporary.replace(output / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    run(json.loads(parser.parse_args().request.read_text()))


if __name__ == "__main__":
    main()
