"""Serial dev → reflection → paired online val → immutable publication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .campaign import BudgetExhausted, Campaign, CampaignError, Cell, summary
from .digest import digest
from .evidence import DevEvidence
from .gate import compare, task_metrics
from .launcher import execution_identity, launch_worker
from .library import Library, LibraryError
from .search import (
    create_candidate,
    evidence_bundle,
    parse_response,
    proposal_prompt,
    reflect,
    should_search,
    validate_lesson_update,
)
from .splits import TASKS, LayoutRegistry, json_hash


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def campaign_lock(root: Path):
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another coordinator owns this campaign") from exc
        yield


def initialize(config_path: Path, root: Path) -> None:
    config = json.loads(config_path.read_text())
    if config.get("launch_enabled"):
        raise CampaignError("initialize with launch_enabled=false, then explicitly enable")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    config["runs_root"] = str(root.resolve() / "runs")
    config["launcher"] = "internal-online-worker"
    save(root / "campaign.json", config)
    library = Library(config["library"])
    if library.active() is None:
        source = root / "empty-library"
        source.mkdir()
        library.set_active(library.freeze(source, parent=None, note="no learned memory baseline"))


class OnlineLoop:
    def __init__(self, root: Path, *, worker: Any = None, reflector: Any = None) -> None:
        self.root = root.resolve()
        self.campaign = Campaign(root)
        self.config = self.campaign.config
        self.worker = worker or launch_worker
        self.reflector = reflector or reflect
        self.library = self.campaign.library
        self.budget_path = root / "learning-budget.json"
        self.budget = (
            json.loads(self.budget_path.read_text())
            if self.budget_path.exists()
            else {
                "model_requests": 0,
                "reflection_requests": 0,
                "search_rounds": {},
            }
        )

    def preflight(self) -> None:
        config = self.config
        if not config.get("launch_enabled"):
            raise CampaignError("launch_enabled is false; no simulation or model calls")
        if not config["tasks"] or not set(config["tasks"]) <= set(TASKS):
            raise CampaignError("invalid task list")
        if config["budgets"].get("max_usd") is not None:
            raise CampaignError(
                "money is not a hard metered budget here; use request/token/time caps"
            )
        settings = config["execution"]
        for field in (
            "max_cells",
            "max_model_requests",
            "wall_seconds",
            "max_tokens",
            "local_attempts",
        ):
            if type(settings[field]) is not int or settings[field] < 1:
                raise CampaignError(f"execution.{field} must be a positive integer")
        for field in ("context_groups", "lesson_limit", "skill_limit"):
            value = settings.get(field)
            if value is not None and (type(value) is not int or value < 1):
                raise CampaignError(f"execution.{field} must be null or a positive integer")
        for field in (
            "search_rounds_per_task",
            "reflection_preview_cells",
            "reflection_preview_images",
        ):
            value = config["learning"].get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise CampaignError(f"learning.{field} must be null or a nonnegative integer")
        if not settings["model"] or not settings["base_url"]:
            raise CampaignError("provide an explicit model and endpoint")
        if not os.environ.get(settings["api_key_env"]):
            raise CampaignError("provide model credentials in the declared environment variable")
        checkpoint = Path(config["sam3_checkpoint"])
        hasher = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != config["sam3_sha256"]:
            raise CampaignError("SAM3 checkpoint identity mismatch")
        LayoutRegistry(config["layout_manifest"])
        identity = execution_identity(config)
        frozen_config = {key: value for key, value in config.items() if key != "launch_enabled"}
        frozen = {
            "execution_sha256": identity,
            "config_sha256": json_hash(frozen_config),
            "layouts_sha256": hashlib.sha256(
                Path(config["layout_manifest"]).read_bytes()
            ).hexdigest(),
        }
        manifest = self.root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != frozen:
            raise CampaignError(
                "campaign code, configuration or split changed; start a new campaign"
            )
        save(manifest, frozen)

    def _reserve(self, requests: int, *, reflection: bool = False) -> None:
        budgets = self.config["budgets"]
        if self.budget["model_requests"] + requests > budgets["max_model_requests"]:
            raise BudgetExhausted("global model request budget exhausted")
        if reflection and self.budget["reflection_requests"] >= budgets["max_reflection_requests"]:
            raise BudgetExhausted("reflection request budget exhausted")
        self.budget["model_requests"] += requests
        self.budget["reflection_requests"] += int(reflection)
        save(self.budget_path, self.budget)

    def _remaining_seconds(self) -> float:
        remaining = self.config["budgets"]["max_hours"] * 3600 - self.campaign.state["seconds"]
        if remaining <= 0:
            raise BudgetExhausted("campaign wall budget exhausted")
        return remaining

    def _validate_result(
        self,
        result: dict[str, Any],
        revision: str,
        split: str,
        cell: Cell,
    ) -> None:
        registry = LayoutRegistry(self.config["layout_manifest"])
        instance = registry.select(split, cell.task, cell.layout)
        identity = json.loads((self.root / "manifest.json").read_text())["execution_sha256"]
        view = self.library.view(revision, cell.task)
        files = {
            str(path.relative_to(view)): hashlib.sha256(path.read_bytes()).hexdigest()
            for kind in ("skills", "lessons")
            for path in (view / kind).glob("*")
            if path.is_file()
        }
        library_hash = hashlib.sha256(
            "".join(f"{name}:{sha}\n" for name, sha in sorted(files.items())).encode()
        ).hexdigest()
        if (
            result.get("instance") != registry.identity(instance)
            or result.get("execution_sha256") != identity
            or result.get("revision") != revision
            or result.get("task") != cell.task
            or result.get("seed") != cell.layout
            or result.get("split") != split
            or (result.get("host") or {}).get("runtime") != "openai"
            or (result.get("library") or {}).get("sha256") != library_hash
        ):
            raise CampaignError("worker result identity mismatch")

    def run_cells(
        self, revision: str, split: str, cells: list[Cell], *, heldout: bool = False
    ) -> list[dict[str, Any]]:
        registry = LayoutRegistry(self.config["layout_manifest"])
        for cell in cells:
            registry.select(split, cell.task, cell.layout)

        def launch(argv: list[str]) -> int:
            task, eval_seed, layout, _, tag, view = argv[1:7]
            identity = json.loads((self.root / "manifest.json").read_text())["execution_sha256"]
            if execution_identity(self.config) != identity:
                raise CampaignError("execution changed during campaign")
            maximum = self.config["execution"]["max_model_requests"]
            self._reserve(maximum)
            output = Path(self.config["runs_root"]) / tag
            request = {
                "task": task,
                "eval_seed": int(eval_seed),
                "layout": int(layout),
                "split": split,
                "revision": revision,
                "view": view,
                "output": str(output),
                "execution_sha256": identity,
            }
            timeout = min(self._remaining_seconds(), self.config["budgets"]["episode_wall_seconds"])
            code = self.worker(self.root / "campaign.json", request, timeout)
            result_file = output / "results" / "episode" / "result.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                self._validate_result(
                    result, revision, split, Cell(task, int(eval_seed), int(layout))
                )
                actual = (result.get("generation") or {}).get("requests")
                if type(actual) is int and 0 <= actual <= maximum:
                    self.budget["model_requests"] -= maximum - actual
                    save(self.budget_path, self.budget)
            return code

        return self.campaign.run(revision, split, cells, allow_heldout=heldout, launch=launch)

    def _history(self, task: str, round_index: int) -> list[dict[str, Any]]:
        history = []
        for directory in sorted((self.root / "iterations").glob(f"{task}-*")):
            previous_round = int(directory.name.rsplit("-", 1)[1])
            if previous_round >= round_index or not (directory / "complete.json").exists():
                continue
            decision = json.loads((directory / "decision.json").read_text())
            dev = json.loads((directory / "dev.json").read_text())
            history.append(
                {
                    "round": previous_round,
                    "proposal": json.loads((directory / "proposal.json").read_text()),
                    "accepted": decision["accepted"],
                    "skipped": decision.get("skipped", False),
                    "candidate_dev": decision.get("candidate_dev", {}),
                    "dev": dev,
                }
            )
        return history

    def _reflection(
        self,
        directory: Path,
        task: str,
        parent: str,
        bundle: dict[str, Any],
        count: int,
        history: list[dict[str, Any]],
        evidence: DevEvidence,
    ) -> dict[str, Any]:
        proposal_file = directory / "proposal.json"
        evidence_ids = evidence.episode_ids
        if proposal_file.exists():
            response = parse_response(
                proposal_file.read_text(), task=task, count=count, evidence_ids=evidence_ids
            )
            for proposal in response["proposals"]:
                validate_lesson_update(self.library, parent, proposal)
            return response
        prompt_history = [
            {
                "round": entry["round"],
                "proposal": entry["proposal"],
                "accepted": entry["accepted"],
                "skipped": entry["skipped"],
            }
            for entry in history
        ]
        prompt = proposal_prompt(
            bundle, self.library, parent, count, history=prompt_history, catalog=evidence.catalog
        )
        (directory / "reflection-prompt.txt").write_text(prompt)
        error = ""
        for attempt in range(1 + self.config["learning"]["format_retries"]):
            response_file = directory / f"reflection-{attempt}.json"
            reservation = directory / f"reflection-{attempt}.reserved.json"
            if response_file.exists():
                raw = json.loads(response_file.read_text())["text"]
            else:
                if reservation.exists():
                    raise CampaignError(
                        "interrupted reflection remains charged; "
                        "inspect before starting a new iteration"
                    )
                self._reserve(1, reflection=True)
                save(reservation, {"reserved": True})
                settings = self.config["execution"]
                started = time.monotonic()
                requests = 1

                def reserve_next(reservation_path: Path = reservation) -> None:
                    nonlocal requests
                    self._reserve(1, reflection=True)
                    requests += 1
                    save(reservation_path, {"reserved_requests": requests})

                def record(
                    event: dict[str, Any],
                    path: Path = directory / f"reflection-{attempt}.events.jsonl",
                ) -> None:
                    with path.open("a") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

                try:
                    raw, usage = self.reflector(
                        prompt + ("\nPrevious proposal rejected: " + error if error else ""),
                        model=settings["model"],
                        base_url=settings["base_url"],
                        api_key=os.environ.get(settings["api_key_env"]),
                        timeout_s=min(
                            self._remaining_seconds(),
                            self.config["learning"]["reflection_timeout_seconds"],
                        ),
                        max_tokens=settings["max_tokens"],
                        extra_body=settings.get("extra_body", {}),
                        images=[
                            path
                            for episode in bundle["episodes"]
                            for path in episode["model_visible_images"]
                        ][: self.config["learning"].get("reflection_preview_images", 4)],
                        evidence=evidence,
                        reserve_next=reserve_next,
                        record=record,
                    )
                    save(response_file, {"text": raw, "usage": usage})
                finally:
                    self.campaign.state["seconds"] += time.monotonic() - started
                    self.campaign._save()
            try:
                response = parse_response(raw, task=task, count=count, evidence_ids=evidence_ids)
                for index, proposal in enumerate(response["proposals"]):
                    validate_lesson_update(self.library, parent, proposal)
                    probe = directory / f"proposal-check-{attempt}-{index}"
                    probe.mkdir(exist_ok=True)
                    for relative, text in proposal["files"].items():
                        path = probe / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(text)
                    from .library import validate

                    errors = validate(probe)
                    if errors:
                        raise LibraryError("; ".join(errors))
                save(proposal_file, response)
                (directory / "reflection.md").write_text(
                    "\n\n".join(
                        f"## {key}\n{response[key]}"
                        for key in ("observed", "hypothesis", "validation")
                    )
                )
                return response
            except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                error = str(exc)
                save(directory / f"rejection-{attempt}.json", {"error": error})
        raise LibraryError("reflection format/validation retries exhausted: " + error)

    def _publish(self, directory: Path, decision: dict[str, Any]) -> None:
        if (directory / "complete.json").exists():
            return
        parent, winner = decision["parent"], decision["winner"]
        if self.library.active() not in {parent, winner}:
            raise CampaignError("active revision changed outside this serial iteration")
        iteration_id = str(directory.resolve())
        if not any(entry.get("iteration") == iteration_id for entry in self.library.promotions()):
            self.library.record({"iteration": iteration_id, **decision})
        if decision["accepted"]:
            self.library.set_active(winner)
        save(directory / "complete.json", {"active": self.library.active()})

    def iterate(self, task: str, round_index: int) -> None:
        directory = self.root / "iterations" / f"{task}-{round_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "decision.json").exists():
            self._publish(directory, json.loads((directory / "decision.json").read_text()))
            return
        start_file = directory / "start.json"
        if start_file.exists():
            parent = json.loads(start_file.read_text())["parent"]
        else:
            parent = self.library.active()
            save(start_file, {"parent": parent})
        if parent != self.library.active():
            raise CampaignError("cannot resume an iteration on a different parent")
        dev = [
            Cell(cell.task, cell.eval_seed, cell.layout, round_index)
            for cell in self.campaign.cells("dev", [task])
        ]
        val = self.campaign.cells("val", [task])
        if not dev or not val:
            raise CampaignError("need nonempty disjoint dev and val layouts")
        parent_dev = self.run_cells(parent, "dev", dev)
        dev_record = {
            "metrics": task_metrics(parent_dev),
            "runs": [facts["run_dir"] for facts in parent_dev],
        }
        save(directory / "dev.json", dev_record)
        history = self._history(task, round_index)
        run_dirs = [Path(path) for path in dev_record["runs"]]
        bundle = evidence_bundle(
            run_dirs,
            task,
            preview_cells=self.config["learning"].get("reflection_preview_cells", 12),
        )
        evidence = DevEvidence(
            run_dirs
            + [Path(path) for entry in history for path in entry["dev"]["runs"]]
            + [
                Path(path)
                for entry in history
                for runs in entry["candidate_dev"].values()
                for path in runs
            ],
            task,
        )
        search_file = directory / "search.json"
        if search_file.exists():
            count = json.loads(search_file.read_text())["count"]
        else:
            spent = sum(
                json.loads(path.read_text())["count"] == 2
                for path in (self.root / "iterations").glob(f"{task}-*/search.json")
            )
            limit = self.config["learning"].get("search_rounds_per_task")
            searching = should_search(
                [entry["dev"]["metrics"] for entry in history], dev_record["metrics"]
            ) and (limit is None or spent < limit)
            count = 2 if searching else 1
            save(search_file, {"count": count})
            if searching:
                self.budget["search_rounds"][task] = spent + 1
                save(self.budget_path, self.budget)
        response = self._reflection(directory, task, parent, bundle, count, history, evidence)
        revisions_file = directory / "candidates.json"
        if revisions_file.exists():
            revisions = json.loads(revisions_file.read_text())
        else:
            revisions = []
            for index, proposal in enumerate(response["proposals"]):
                campaign_id = hashlib.sha256(str(self.root).encode()).hexdigest()[:10]
                name = f"{campaign_id}-{directory.name}-{index}"
                existing = self.library.root / "candidates" / name
                if existing.exists():
                    revision = self.library.freeze(existing, parent=parent)
                    expected = dict(self.library.manifest(parent)["files"])
                    expected.update(
                        {
                            name: hashlib.sha256(text.encode()).hexdigest()
                            for name, text in proposal["files"].items()
                        }
                    )
                    if self.library.manifest(revision)["files"] != expected:
                        raise CampaignError(
                            "interrupted candidate is incomplete; do not evaluate it"
                        )
                else:
                    _, revision = create_candidate(self.library, parent, name, proposal)
                if revision == parent:
                    continue
                revisions.append(revision)
            save(revisions_file, revisions)
        if not revisions:
            decision = {
                "parent": parent,
                "winner": parent,
                "accepted": False,
                "skipped": True,
                "reason": response.get("no_change") or "proposal makes no content change",
                "evidence": dev_record["runs"],
                "candidate_dev": {},
                "validation": {},
            }
            save(directory / "decision.json", decision)
            self._publish(directory, decision)
            return
        winner = revisions[0]
        runs: dict[str, list[dict[str, Any]]] = {revision: [] for revision in revisions}
        if len(revisions) > 1:
            for cell_index, cell in enumerate(dev):
                order = revisions if cell_index % 2 == 0 else list(reversed(revisions))
                for revision in order:
                    runs[revision].extend(self.run_cells(revision, "dev", [cell]))
            for revision in revisions[1:]:
                if compare(runs[winner], runs[revision])["accepted"]:
                    winner = revision
        validation: dict[str, list[dict[str, Any]]] = {parent: [], winner: []}
        for index, cell in enumerate(val):
            order = [parent, winner] if index % 2 == 0 else [winner, parent]
            for revision in order:
                validation[revision].extend(self.run_cells(revision, "val", [cell]))
        gate = compare(validation[parent], validation[winner])
        decision = {
            "parent": parent,
            "winner": winner,
            **gate,
            "evidence": [facts["run_dir"] for facts in parent_dev],
            "candidate_dev": {
                revision: [facts["run_dir"] for facts in facts_list]
                for revision, facts_list in runs.items()
                if facts_list
            },
            "validation": {
                revision: [facts["run_dir"] for facts in facts_list]
                for revision, facts_list in validation.items()
            },
        }
        save(directory / "decision.json", decision)
        self._publish(directory, decision)

    def recover(self) -> None:
        pending = self.campaign.state.get("in_flight")
        if not pending:
            return
        output = Path(self.config["runs_root"]) / pending["tag"]
        worker_file = output / "worker.json"
        if worker_file.exists():
            pid = json.loads(worker_file.read_text())["pid"]
            if Path(f"/proc/{pid}").exists():
                raise CampaignError("worker PID still exists; inspect/stop it before recovery")
        run = output / "results" / "episode"
        facts = digest(run) if (run / "result.json").exists() else None
        recovery_error = None
        if facts and facts["kind"] == "policy":
            task, seed, layout, repeat = pending["cell"].split("/")
            cell = Cell(task, int(seed[1:]), int(layout[1:]), int(repeat[1:]))
            try:
                self._validate_result(
                    json.loads((run / "result.json").read_text()),
                    pending["revision"],
                    pending["split"],
                    cell,
                )
            except (ValueError, OSError) as exc:
                recovery_error = str(exc)
                facts = None
        if facts and facts["kind"] == "policy":
            self.campaign.state["done"][pending["key"]] = str(run)
            self.campaign.state["episodes"] += 1
        else:
            self.campaign.state["infra_runs"] += 1
        self.campaign.state["seconds"] += self.config["budgets"]["episode_wall_seconds"]
        self.campaign._append(
            {
                **pending,
                "kind": facts["kind"] if facts else "infra",
                "success": facts["success"] if facts else False,
                "run_dir": str(run),
                "recovered": True,
                "recovery_error": recovery_error,
            }
        )
        self.campaign.state["in_flight"] = None
        self.campaign._save()

    def report(self) -> dict[str, Any]:
        report = summary(self.campaign)
        report["learning_budget"] = self.budget
        groups: dict[str, list[float]] = {}
        for key, row in report["by_revision"].items():
            revision, split, _ = key.split()
            row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else None
            row["attempted"] = row["episodes"] + row["infra"]
            if row["success_rate"] is not None:
                groups.setdefault(f"{revision} {split}", []).append(row["success_rate"])
        report["macro_success_rate_completed_tasks"] = {
            key: sum(values) / len(values) for key, values in groups.items()
        }
        report["planned_cells_per_revision"] = {
            split: len(self.campaign.cells(split)) for split in self.config["splits"]
        }
        report["active"] = self.library.active()
        save(self.root / "report.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", type=Path, required=True)
    commands.add_parser("preflight")
    commands.add_parser("run")
    commands.add_parser("recover")
    commands.add_parser("report")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--allow-heldout", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.config, args.root)
        return
    with campaign_lock(args.root):
        loop = OnlineLoop(args.root)
        if args.command == "report":
            print(json.dumps(loop.report(), indent=2))
            return
        if args.command == "recover":
            loop.recover()
            return
        loop.preflight()
        if args.command == "run":
            if loop.campaign.state["heldout_revisions"]:
                raise CampaignError(
                    "this campaign has seen heldout results; do not resume learning"
                )
            for round_index in range(loop.config["learning"]["rounds_per_task"]):
                for task in loop.config["tasks"]:
                    loop.iterate(task, round_index)
        elif args.command == "evaluate":
            if not args.allow_heldout:
                raise CampaignError("explicit --allow-heldout is required")
            if not loop.campaign.cells("heldout"):
                raise CampaignError("no heldout instances declared")
            loop.run_cells(
                loop.library.active(), "heldout", loop.campaign.cells("heldout"), heldout=True
            )
        loop.report()


if __name__ == "__main__":
    main()
