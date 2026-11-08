"""One RoboDojo episode in its own process: ``python eval/robodojo/episode.py REQUEST``.

The call to :func:`pyrualean.play.play_episode` is the one ``pyrualean.learn.episode`` makes,
argument for argument, so that an episode here gets the prompt and execution of a learn
campaign's test episodes.  :func:`identity` hashes what must match them (the policy code
without ``learn/`` and ``evolve/``, the benchmark, the execution settings, SAM3) with the same
algorithm, so it equals the ``identity`` those episodes recorded; ``run.py`` refuses to start
otherwise.  Nothing from ``pyrualean.learn`` or ``pyrualean.evolve`` is imported.

``result.json`` gains ``eval_seed``, ``identity`` and ``library_name`` (the exported library).

``execution.runtime`` (default ``openai``) may be ``codex``: the episode then runs ``codex exec``
(``execution.codex_bin``) in the clean ``execution.codex_home``, against ``execution.base_url``
when it is set, otherwise the home's own login (see tools/codex-home/config.toml).

A layout with an expansion (``config["expansions"]``, see ``export.py``) gets it after the task
card under :data:`EXPANSION_HEADING`, and ``result.json`` gains ``expansion`` (source, sha256).
The plan's ``config["addendum"]`` (one text for every episode) ends the prompt under its own
heading, after any expansion; ``result.json`` then gains ``addendum`` (source, sha256).
``pyrualean.play.build_prompt`` is wrapped in this process, not edited: its files are part of
:func:`identity`, which is therefore the same with and without expansions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: The package directories the hash leaves out, as ``learn.episode`` does.
NOT_POLICY = ("learn", "evolve", "__pycache__")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def tree_hash(root: Path) -> str:
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
    """The execution identity of ``config`` with the installed ``pyrualean``."""

    import pyrualean

    source = Path(config["source_root"])
    return json_hash(
        {
            "execution": config["execution"],
            "policy": policy_hash(Path(pyrualean.__file__).resolve().parent),
            "benchmark": {part: tree_hash(source / part) for part in ("env_cfg", "task", "src")},
            "sam3": config.get("sam3_sha256"),
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


#: Heading of a layout's expansion, put after the prompt's task card.
EXPANSION_HEADING = "# Notes on this scene (written from the step-0 head image)"


def append_expansion(text: str, heading: str = EXPANSION_HEADING) -> None:
    """Make the prompts :mod:`pyrualean.play` builds in this process end with ``text``."""

    import pyrualean.play as play

    build = play.build_prompt

    def build_prompt(*args: Any, **kwargs: Any) -> str:
        return build(*args, **kwargs) + f"\n\n{heading}\n\n{text}"

    play.build_prompt = build_prompt


def run(request: dict[str, Any]) -> dict[str, Any]:
    from pyrualean.hosts.rpent_robodojo import Cell
    from pyrualean.play import play_episode

    config = request["config"]
    if identity(config) != request["identity"]:
        raise RuntimeError("code, benchmark or execution settings changed since init")
    settings = config["execution"]
    output = Path(request["output"]) / "episode"
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SAM3_CHECKPOINT_PATH"] = config["sam3_checkpoint"]
    expansion = config.get("expansions", {}).get(request["task"], {}).get(str(request["layout"]))
    if expansion is not None:
        if hashlib.sha256(expansion["text"].encode("utf-8")).hexdigest() != expansion["sha256"]:
            raise RuntimeError(f"the expansion of layout {request['layout']} changed since export")
        append_expansion(expansion["text"])
    addendum = config.get("addendum")
    if addendum is not None:
        if hashlib.sha256(addendum["text"].encode("utf-8")).hexdigest() != addendum["sha256"]:
            raise RuntimeError("the addendum changed since export")
        append_expansion(addendum["text"], addendum["heading"])
    result = play_episode(
        Cell(
            settings["suite"], request["task"], int(request["layout"]), cuda_device=request["gpu"]
        ),
        output,
        arm="cells",
        max_attempts=1,
        model=settings["model"],
        reasoning_effort=settings["extra_body"].get("reasoning_effort", "high"),
        budget_s=settings["policy_seconds"],
        max_turns=settings["max_cells"],
        runtime=settings.get("runtime", "openai"),
        codex_bin=settings.get("codex_bin", "codex"),
        codex_home=settings.get("codex_home"),
        replay_dir=None,
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
        openai_options={
            "max_requests": settings["max_model_requests"],
            "max_tokens": settings["max_tokens"],
            "nudges": settings["nudges"],
            "extra_body": settings["extra_body"],
            "retries": settings["provider_retries"],
        },
    )
    result.update(
        eval_seed=int(request["eval_seed"]),
        identity=request["identity"],
        library_name=request["library_name"],
    )
    if expansion is not None:
        result["expansion"] = {"source": expansion["source"], "sha256": expansion["sha256"]}
    if addendum is not None:
        result["addendum"] = {"source": addendum["source"], "sha256": addendum["sha256"]}
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
