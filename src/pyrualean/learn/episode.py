"""One RoboDojo (or RoboCasa365, ``"benchmark": "robocasa"``) episode in its own process:
``python -m pyrualean.learn.episode REQUEST``.

The request names the official layout (eval seed + index), the GPU, the library view and the
author of the cells: the model (``runtime: openai``, or ``execution.runtime: codex`` with
``execution.codex_bin`` and ``codex_home``) or saved cells (``runtime: replay``).
Every episode, of v0, of a learned revision or a replay, gets the same execution settings and
the same prompt contract (the library note is always there; v0's library is empty).

``result.json`` gains the request's identity and ``end``: why the episode stopped, a fact
the worker reads (a run cut short by the harness is not a policy lesson).

A task with expansions (``config["expansions"][task]["s<seed>l<layout>"]``: source, sha256,
text, frozen in the config) gets its layout's after the task card under
:data:`EXPANSION_HEADING`, as ``eval/robodojo/episode.py`` does; ``result.json`` gains
``expansion``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..evolve.splits import json_hash, tree_hash
from .config import robocasa, scene_seed


def execution_identity(config: dict[str, Any]) -> str:
    """Everything that must not change during a campaign, except the library."""

    repo = Path(__file__).resolve().parents[3]
    if robocasa(config):
        # as eval/robocasa: the kitchen's Python sources stand for the benchmark
        packages = Path(config["sim_packages"])
        return json_hash(
            {
                "execution": config["execution"],
                "policy": _policy_hash(repo / "src" / "pyrualean"),
                "kitchen": {name: _py_hash(packages / name) for name in ("robocasa", "robosuite")},
                "sam3": config.get("sam3_sha256"),
            }
        )
    source = Path(config["source_root"])
    return json_hash(
        {
            "execution": config["execution"],
            "policy": _policy_hash(repo / "src" / "pyrualean"),
            "benchmark": {part: tree_hash(source / part) for part in ("env_cfg", "task", "src")},
            "sam3": config.get("sam3_sha256"),
            **({"expansions": config["expansions"]} if config.get("expansions") else {}),
        }
    )


#: Heading of a layout's expansion, put after the prompt's task card (eval/robodojo/episode.py's).
EXPANSION_HEADING = "# Notes on this scene (written from the step-0 head image)"


def layout_expansion(config: dict[str, Any], task: str, eval_seed: int, layout: int) -> Any:
    """The expansion of one layout, or None for a task without expansions."""

    expansions = (config.get("expansions") or {}).get(task)
    if expansions is None:
        return None
    key = f"s{eval_seed}l{layout}"
    if key not in expansions:
        raise RuntimeError(f"{task}: no expansion for {key}")
    expansion = expansions[key]
    if hashlib.sha256(expansion["text"].encode("utf-8")).hexdigest() != expansion["sha256"]:
        raise RuntimeError(f"{task}: the expansion of {key} does not match its sha256")
    return expansion


def append_expansion(text: str) -> None:
    """Make the prompts :mod:`pyrualean.play` builds in this process end with ``text``."""

    from .. import play

    build = play.build_prompt

    def build_prompt(*args: Any, **kwargs: Any) -> str:
        return build(*args, **kwargs) + f"\n\n{EXPANSION_HEADING}\n\n{text}"

    play.build_prompt = build_prompt


def _py_hash(root: Path) -> str:
    """The ``*.py`` files under ``root`` (a kitchen package without its assets)."""

    return json_hash(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        }
    )


def _policy_hash(package: Path) -> str:
    """The code an episode runs: the package without the learning loop (learn/, evolve/)."""

    return json_hash(
        {
            str(path.relative_to(package)): tree_hash(path)
            if path.is_dir()
            else json_hash(path.read_text(errors="replace"))
            for path in sorted(package.iterdir())
            if path.name not in ("learn", "evolve", "__pycache__")
        }
    )


def end_reason(result: dict[str, Any]) -> str:
    """Why an episode stopped, from its record (``digest`` shows it to the worker)."""

    generation = result.get("generation") or {}
    verdict = result.get("verdict") or {}
    reasons = generation.get("finish_reasons") or []
    if result.get("environment_success"):
        return "success"
    if result.get("status") in ("boot_error", "error"):
        return f"infrastructure: {result.get('status')}"
    if generation.get("runtime") == "replay":
        return f"replay: {generation.get('stopped_by')}"
    if result.get("finish"):
        return "the policy called finish"
    if result.get("status") == "call_budget":
        return "cell budget exhausted"
    if generation.get("request_budget_exhausted"):
        return "model request budget exhausted"
    if result.get("status") == "timeout":
        return "policy wall clock exhausted"
    if result.get("status") == "planner_error":
        return "model provider failed mid-episode"
    limit, step = verdict.get("step_limit"), verdict.get("step")
    if limit and step is not None and step >= limit:
        return "native step limit reached"
    if reasons and reasons[-1] == "stop":
        return "the model stopped calling tools (nudges exhausted)"
    return "benchmark ended the episode"


def run(request: dict[str, Any]) -> dict[str, Any]:
    config = request["config"]
    if execution_identity(config) != request["identity"]:
        raise RuntimeError("code, benchmark or execution settings changed during the campaign")
    output = Path(request["output"]) / "episode"
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SAM3_CHECKPOINT_PATH"] = config["sam3_checkpoint"]
    replay = request.get("replay_dir")
    expansion = layout_expansion(
        config, request["task"], int(request["eval_seed"]), int(request["layout"])
    )
    if expansion is not None:
        append_expansion(expansion["text"])
    if robocasa(config):
        result = _robocasa_episode(request, output, replay)
    else:
        result = _robodojo_episode(request, output, replay)
    result.update(
        split=request["split"],
        eval_seed=int(request["eval_seed"]),
        revision=request["revision"],
        identity=request["identity"],
        end=end_reason(result),
    )
    if expansion is not None:
        result["expansion"] = {"source": expansion["source"], "sha256": expansion["sha256"]}
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False) + "\n")
    temporary.replace(output / "result.json")
    return result


def _openai_options(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_requests": settings["max_model_requests"],
        "max_tokens": settings["max_tokens"],
        "nudges": settings["nudges"],
        "extra_body": settings["extra_body"],
        "retries": settings["provider_retries"],
    }


def _robocasa_episode(request: dict[str, Any], output: Path, replay: Any) -> dict[str, Any]:
    """One RoboCasa365 cell: the layout's scene seed (as eval/robocasa/robocasa_episode.py, with
    the library view)."""

    from ..hosts.robocasa_rgb import Cell
    from ..play import play_episode

    config = request["config"]
    settings = config["execution"]
    seed = scene_seed(config, request["split"], request["task"], int(request["layout"]))
    result = play_episode(
        # the physical GPU: the kitchen renders on it (EGL) and SAM3 pins it
        Cell(
            settings["split"],
            request["task"],
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
        library=request["view"],
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        boot_options={
            "sim_python": config["sim_python"],
            "robocasa_assets_path": config["robocasa_assets"],
        },
        openai_options=_openai_options(settings),
    )
    result["seed"] = seed
    return result


def _robodojo_episode(request: dict[str, Any], output: Path, replay: Any) -> dict[str, Any]:
    from ..hosts.rpent_robodojo import Cell
    from ..play import play_episode

    config = request["config"]
    settings = config["execution"]
    return play_episode(
        Cell(
            settings["suite"], request["task"], int(request["layout"]), cuda_device=request["gpu"]
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
        robot="robodojo_xpl_rgb",
        primitives="no-vla",
        images=settings["images"],
        feedback=settings["feedback"],
        library=request["view"],
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        boot_options={
            "source_root": config["source_root"],
            "eval_seed": int(request["eval_seed"]),
            "sim_python": config["sim_python"],
        },
        openai_options=_openai_options(settings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    request = json.loads(parser.parse_args().request.read_text())
    run(request)


if __name__ == "__main__":
    main()
