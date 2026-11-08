"""One LIBERO-PRO episode in its own process: ``python libero_episode.py REQUEST``.

Runs in the RPent LIBERO environment (``paths.python``) with ``src/`` of the repository
``paths.lean`` on the path (``run.py`` sets both).  The robot is ``libero_rgb`` (the colour
primitives) or ``libero`` (RPent's own, as the PyRUA-Lean paper's LIBERO-PRO runs), with the
``no-vla`` primitive set and no Pi0.5 server; the policy is
:func:`pyrualean.play.play_episode`'s cell agent with the arm's library, as
``eval/robodojo/episode.py`` runs a RoboDojo one.  :func:`identity` hashes what must not
change during an evaluation (the policy code without ``learn/`` and ``evolve/``, RPent's LIBERO
code, the suite's BDDL and init states, SAM3, the execution settings); an episode refuses to
start when it differs from the one ``run.py init`` recorded.

A task of the evaluation is ``<arm>/<suite>-t<index>`` and its layout ``n`` is the LIBERO seed
``seeds[n]``.  A config with ``replay_dir`` runs those saved cells instead of the model (a
check of the whole chain at no model cost).  ``result.json`` gains ``identity``,
``library_name``, ``arm`` and ``seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: The package directories the hash leaves out, as ``eval/robodojo/episode.py`` does.
NOT_POLICY = ("learn", "evolve", "__pycache__")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"no directory {root}")
    return json_hash(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
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
    liberopro = Path(paths["liberopro"])
    return json_hash(
        {
            "execution": config["execution"],
            "policy": policy_hash(Path(paths["lean"]) / "src" / "pyrualean"),
            "rpent": {
                part: tree_hash(Path(paths["rpent_code"]) / part)
                for part in ("robots/libero", "rpent")
            },
            "benchmark": {
                part: tree_hash(liberopro / part / config["suite"])
                for part in ("bddl_files", "init_files")
            },
            "sam3": config["sam3_sha256"],
        }
    )


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


def parse_task(task: str) -> tuple[str, str, int]:
    """``<arm>/<suite>-t<index>`` as (arm, suite, index)."""

    arm, name = task.split("/")
    suite, index = name.rsplit("-t", 1)
    return arm, suite, int(index)


def run(request: dict[str, Any]) -> dict[str, Any]:
    from pyrualean.hosts.rpent_libero_rgb import Cell
    from pyrualean.play import play_episode

    config = request["config"]
    if identity(config) != request["identity"]:
        raise RuntimeError("code, benchmark or execution settings changed since init")
    settings = config["execution"]
    arm, suite, index = parse_task(request["task"])
    if suite != config["suite"]:
        raise RuntimeError(f"{request['task']}: not a task of {config['suite']}")
    seed = int(config["seeds"][int(request["layout"])])
    output = Path(request["output"]) / "episode"
    output.mkdir(parents=True, exist_ok=False)
    # a replay root (``replay_dir`` in its config) runs saved cells instead of the model
    replay = config.get("replay_dir")
    result = play_episode(
        # the physical GPU: RPent's servers pin it themselves (CUDA_VISIBLE_DEVICES, EGL device)
        Cell(suite, index, seed, int(settings["max_episode_steps"]), cuda_device=request["gpu"]),
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
        library=request["view"],
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        rpent_root=config["paths"]["rpent_code"],
        boot_options={"vla": settings["primitives"] != "no-vla"},
        openai_options={
            "max_requests": settings["max_model_requests"],
            "max_tokens": settings["max_tokens"],
            "nudges": settings["nudges"],
            "extra_body": settings["extra_body"],
            "retries": settings["provider_retries"],
        },
    )
    result.update(
        identity=request["identity"],
        library_name=request["library_name"],
        arm=arm,
        seed=seed,
    )
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
