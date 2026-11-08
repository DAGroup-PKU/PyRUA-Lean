"""Export the tested library of each task of finished learn campaigns, read-only.

``python eval/export.py ROOT --config CONFIG``

For each task of ``CONFIG`` (``{task: {campaign, layouts}}``) the ``final`` revision of its
``test.json``, which must still be the active one, is exported as the directory its test
episodes read (``library/<task>/views/<final>/<task>``) to ``ROOT/libraries/<task>``.  Its
hash must equal the one those episodes recorded, and so must their execution identity, which
this run takes over with the campaign's settings (execution, environment, simulator paths):
results here then compare with that test.  Nothing is written under a campaign.

With ``"source": "active"`` (campaigns run without test) the active revision is exported
instead, checked the same way against the dev episodes that ran it; a task's own ``source``
overrides the plan's.  A plan's ``environment`` or ``source_root`` (the RoboDojo tree, e.g. other
judging code) replaces the campaigns' (which must otherwise agree).  A plan's ``execution`` is a
dict of settings put over the campaigns' (e.g. ``runtime: codex``): the campaigns' own is kept
as ``learned_execution``, and every task must then be in ``allow_identity_mismatch``.  Tasks named in ``allow_identity_mismatch`` were learned with other code or prompts
than this run's: their library is still checked, their recorded identity is kept as
``learned_identity`` instead of having to equal this run's.

A task with ``"library": DIR`` runs that directory instead of a revision of its campaign (a
library made elsewhere, e.g. from other tasks).  The campaign then only supplies the settings,
checked against its round-0 (empty library) dev episodes, whose identity this run must keep.
With ``"settings_task": OTHER`` as well, those are the round-0 dev episodes of the campaign's
task ``OTHER``: a task the campaign never learned (e.g. one no campaign has) runs with them.

A task with ``"only": [N, ...]`` (as many as its ``layouts``, each below its ``pool``) runs those
layouts and no other: none is replaced by a later one (e.g. a retest of chosen layouts).

A task with ``"expansion": PATTERN`` (a path with ``{layout}``, e.g. notes written per layout, as
``expand/expanded.txt``) has the text of each layout of its pool frozen into
``config["expansions"][task][layout]``; ``episode.py`` puts it after the prompt's task card.

A plan's ``"addendum": {"path": FILE, "heading": HEADING}`` (one text for every task and layout,
e.g. how to write the code) is frozen into ``config["addendum"]`` (source, sha256, heading, text);
``episode.py`` puts it at the very end of the prompt, after any expansion.

``"allow_learned_seed": true`` lets a plan whose every task runs a given ``library`` (e.g. a
round-0 empty view) use the eval seed its campaign learned on: nothing was learned into it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode import identity, library_sha256  # noqa: E402

#: Taken from the source campaigns, which must agree on each.
INHERITED = (
    "execution",
    "environment",
    "source_root",
    "sim_python",
    "sam3_checkpoint",
    "sam3_sha256",
)


class ExportError(ValueError):
    """A configuration or a source campaign that cannot be evaluated as asked."""


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def layout_file(config: dict[str, Any], task: str, layout: int) -> Path:
    """The official layout an episode of ``task`` loads."""

    return (
        Path(config["source_root"])
        / "Assets"
        / "Eval_Layout"
        / "RoboDojo"
        / config["execution"]["suite"]
        / str(config["eval_seed"])
        / f"{task}_{layout}.json"
    )


def _tested(campaign: Path, task: str, final: str, split: str = "test") -> dict[str, Any]:
    """The identity and library hash shared by the ``split`` episodes of ``final``."""

    runs = sorted((campaign / "runs" / split / task).glob(f"{final}-l*"))
    identities = {_load(run / "request.json")["identity"] for run in runs}
    libraries = {
        result["library"]["sha256"]
        for run in runs
        if (result := _load(run / "episode" / "result.json")) is not None
    }
    if len(identities) != 1 or len(libraries) != 1:
        raise ExportError(
            f"{campaign.name}/{task}: {len(identities)} identities and {len(libraries)} "
            f"library hashes among the test runs of {final}"
        )
    return {"identity": identities.pop(), "library": libraries.pop()}


def check(plan: dict[str, Any]) -> None:
    if plan.get("eval_seed") not in (0, 1, 2):
        raise ExportError("eval_seed in 0..2")
    for task, spec in [("", plan), *plan.get("tasks", {}).items()]:
        if spec.get("source", "test") not in ("test", "active"):
            raise ExportError(f"{task or 'the plan'}: source is test or active")
    unknown = set(plan.get("allow_identity_mismatch", ())) - set(plan.get("tasks", {}))
    if unknown:
        raise ExportError(f"allow_identity_mismatch names no task of the plan: {sorted(unknown)}")
    if not plan.get("tasks"):
        raise ExportError("tasks: {task: {campaign, layouts}}")
    if "execution" in plan and not isinstance(plan["execution"], dict):
        raise ExportError("execution: a dict of settings over the campaigns'")
    for task, spec in plan["tasks"].items():
        if not spec.get("campaign") or not isinstance(spec.get("layouts"), int):
            raise ExportError(f"{task}: a campaign and a layout count")
        if spec["layouts"] < 1:
            raise ExportError(f"{task}: at least one layout")
        if spec.get("pool", spec["layouts"]) < spec["layouts"]:
            raise ExportError(f"{task}: a pool of at least its layouts")
        only = spec.get("only")
        if only is not None:
            if not isinstance(only, list) or len(set(only)) != len(only) or not all(
                isinstance(n, int) and n >= 0 for n in only
            ):
                raise ExportError(f"{task}: only is a list of distinct layout numbers")
            if len(only) != spec["layouts"]:
                raise ExportError(f"{task}: only names exactly its {spec['layouts']} layouts")
            if max(only) >= spec.get("pool", spec["layouts"]):
                raise ExportError(f"{task}: a pool above every layout of only")
        if "settings_task" in spec and not spec.get("library"):
            raise ExportError(f"{task}: settings_task goes with a given library")
        if "expansion" in spec and "{layout}" not in str(spec["expansion"]):
            raise ExportError(f"{task}: expansion is a path with {{layout}}")
        if plan.get("allow_learned_seed") and not spec.get("library"):
            raise ExportError(f"{task}: allow_learned_seed goes with a given library")
    addendum = plan.get("addendum")
    if addendum is not None and (
        not isinstance(addendum, dict)
        or set(addendum) != {"path", "heading"}
        or not str(addendum["heading"]).strip()
    ):
        raise ExportError("addendum: {path, heading}")
    for field in ("concurrency", "per_gpu", "need_gib"):
        if not isinstance(plan.get(field), (int, float)) or plan[field] <= 0:
            raise ExportError(f"{field} must be positive")
    if not plan.get("gpus"):
        raise ExportError("gpus is required")
    budgets = plan.get("budgets") or {}
    for field in ("max_launches", "max_hours", "max_infra_retries", "episode_watchdog_seconds"):
        if not isinstance(budgets.get(field), (int, float)) or budgets[field] < 0:
            raise ExportError(f"budgets.{field} is required")


def export(root: Path, config_path: Path) -> dict[str, Any]:
    plan = json.loads(Path(config_path).read_text())
    check(plan)
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty: an evaluation is never re-exported")
    config: dict[str, Any] = {**{k: v for k, v in plan.items() if k != "execution"}, "libraries": {}}
    if "addendum" in plan:
        path = Path(plan["addendum"]["path"])
        text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not text:
            raise ExportError(f"addendum: no text in {path}")
        config["addendum"] = {
            "source": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "heading": plan["addendum"]["heading"].strip(),
            "text": text,
        }
    seconds: dict[str, float] = {}
    tested: dict[str, str] = {}
    for task, spec in plan["tasks"].items():
        given = Path(spec["library"]).resolve() if spec.get("library") else None
        split = "dev" if given or spec.get("source", plan.get("source")) == "active" else "test"
        campaign = Path(spec["campaign"]).resolve()
        learned = _load(campaign / "config.json")
        source = spec.get("settings_task", task)  # the campaign's task whose episodes vouch
        if source not in learned["tasks"]:
            raise ExportError(f"{source} is not a task of {campaign}")
        if learned["dev"]["eval_seed"] == plan["eval_seed"] and not plan.get("allow_learned_seed"):
            raise ExportError(f"{task}: eval seed {plan['eval_seed']} is what it learned on")
        active = _load(campaign / "library" / source / "active.json")
        if split == "dev":
            rounds = (_load(campaign / "tasks" / source / "learn.json") or {}).get("rounds") or []
            if len(rounds) != learned["rounds"] or not all(r.get("done") for r in rounds):
                raise ExportError(f"{task}: its rounds in {campaign} are not all done")
            final = rounds[0]["revision"] if given else rounds[-1]["active"]
            if not given and (active or {}).get("revision") != final:
                raise ExportError(f"{task}: the active revision is not the last round's ({final})")
        else:
            test = _load(campaign / "tasks" / source / "test.json")
            if not test:
                raise ExportError(f"{task} has no test in {campaign}: its campaign is not done")
            final = test["final"]
            if (active or {}).get("revision") != final:
                raise ExportError(f"{task}: the active revision is not the one tested ({final})")
        found = _tested(campaign, source, final, split)
        # the settings the episodes ran with (their request carries the loaded config)
        ran = _load(next((campaign / "runs" / split / source).glob(f"{final}-l*")) / "request.json")
        for field in INHERITED:
            if field in ("environment", "source_root") and field in plan:
                continue
            if field in config and config[field] != ran["config"].get(field):
                raise ExportError(f"{field} differs between the source campaigns")
            config[field] = ran["config"].get(field)
        view = campaign / "library" / source / "views" / final / source
        if library_sha256(view) != found["library"]:
            raise ExportError(f"{task}: the view of {final} is not what its {split} episodes read")
        tested[task] = found["identity"]
        if given:
            if not given.is_dir():
                raise ExportError(f"{task}: no library directory {given}")
            config["libraries"][task] = {
                "campaign": str(campaign),
                "revision": "given",
                "library_dir": str(given),
                "sha256": library_sha256(given),
                "source_test": {"split": "given", "path": str(given)},
            }
            if source != task:
                config["libraries"][task]["settings_task"] = source
        elif split == "dev":
            # the last round that ran the active revision on dev (a rollback ran it earlier)
            ran_dev = [r for r in rounds if r["revision"] == final][-1]
            source = {
                "split": "dev",
                "eval_seed": learned["dev"]["eval_seed"],
                "round": ran_dev["round"],
                "episodes": ran_dev["dev"]["episodes"],
                "successes": ran_dev["dev"]["successes"],
            }
        else:
            finals = [p["final"] for p in test["pairs"] if p["final"] is not None]
            source = {
                "eval_seed": learned["test"]["eval_seed"],
                "episodes": len(finals),
                "successes": sum(finals),
            }
        if spec.get("expansion"):
            expansions = config.setdefault("expansions", {})[task] = {}
            for layout in range(spec.get("pool", spec["layouts"])):
                path = Path(spec["expansion"].format(layout=layout))
                text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
                if not text:
                    raise ExportError(f"{task}: no expansion {path}")
                expansions[str(layout)] = {
                    "source": str(path),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                }
        if not given:
            config["libraries"][task] = {
                "campaign": str(campaign),
                "revision": final,
                "sha256": found["library"],
                "source_test": source,
            }
        if task in plan.get("allow_identity_mismatch", ()):
            config["libraries"][task]["learned_identity"] = found["identity"]
        ledger = campaign / "episodes.jsonl"
        took = [
            entry["seconds"]
            for line in ledger.read_text().splitlines()
            if line
            and (entry := json.loads(line))["kind"] == "policy"
            and entry["split"] == split
            and entry["revision"] == final
        ]
        seconds[task] = sum(took) / len(took) if took else 0.0
    if "execution" in plan:
        config["learned_execution"] = config["execution"]
        config["execution"] = {**config["execution"], **plan["execution"]}
    if config["budgets"]["episode_watchdog_seconds"] <= config["execution"]["policy_seconds"]:
        raise ExportError("the watchdog must leave the episode its policy wall clock and boot")
    config["identity"] = identity(config)
    allowed = set(plan.get("allow_identity_mismatch", ()))
    differ = sorted(
        task for task, value in tested.items() if value != config["identity"] and task not in allowed
    )
    if differ:
        raise ExportError(f"execution identity differs from the test runs of {differ}")
    absent = [
        f"{task}_{layout}"
        for task, spec in plan["tasks"].items()
        for layout in range(spec.get("pool", spec["layouts"]))
        if not layout_file(config, task, layout).is_file()
    ]
    if absent:
        raise ExportError(f"no official layout for {absent[:5]} ({len(absent)} in all)")
    # longest episodes first, so that the last ones to finish are short
    config["order"] = sorted(plan["tasks"], key=lambda task: -seconds[task])
    root.mkdir(parents=True, exist_ok=True)
    for task, record in config["libraries"].items():
        view = (
            Path(record["library_dir"])
            if "library_dir" in record
            else Path(record["campaign"]) / "library" / task / "views" / record["revision"] / task
        )
        target = root / "libraries" / task
        shutil.copytree(view, target)
        if library_sha256(target) != record["sha256"]:
            raise ExportError(f"{task}: the exported copy differs from {view}")
    (root / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n")
    return planned(config)


def planned(config: dict[str, Any]) -> dict[str, Any]:
    tasks = {task: spec["layouts"] for task, spec in config["tasks"].items()}
    return {"episodes": sum(tasks.values()), "tasks": tasks, "concurrency": config["concurrency"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(export(args.root, args.config), indent=1) + "\n")


if __name__ == "__main__":
    main()
