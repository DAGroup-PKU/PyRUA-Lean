"""On-disk layout of one authoring round, and who may read or write each part.

::

    <round>/
      round.json                public  round identity, budgets, digests (no holdout data)
      brief.md                  public  the linted brief
      bin/xh                    public  the broker tool (read-only)
      handoff-in/  references/  public  previous-round bundle, verified reference experts
      workspace/                agent   the only place the agent writes files
      agent-home/               agent   the agent CLI's own HOME / config / session state
      tmp/                      agent   bound to /tmp inside the sandbox
      broker/requests/          agent   drop box for xh requests (harness validates)
      broker/responses/ results/ status.json   harness-written, agent-readable
      rollouts/<attempt>/       harness-written dev rollouts, agent-readable
      submissions/<id>/         harness-written expert snapshots, agent-readable
      ledger/attempts.jsonl     harness-owned attempt ledger (append-only), agent-readable
      ledger/events.jsonl       harness-owned events, agent-readable
      ledger/agent-calls.jsonl  datafarm.agent-call/1 records, agent-readable
      harness/                  PRIVATE: state, transcripts, tamper manifests, holdout seeds,
                                holdout runs and results. Masked in every sandbox.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RoundPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    # public (agent read-only)
    @property
    def round_json(self) -> Path:
        return self.root / "round.json"

    @property
    def brief(self) -> Path:
        return self.root / "brief.md"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def tool(self) -> Path:
        return self.bin_dir / "xh"

    @property
    def handoff_in(self) -> Path:
        return self.root / "handoff-in"

    @property
    def references(self) -> Path:
        return self.root / "references"

    @property
    def broker(self) -> Path:
        return self.root / "broker"

    @property
    def responses(self) -> Path:
        return self.broker / "responses"

    @property
    def results(self) -> Path:
        return self.broker / "results"

    @property
    def status(self) -> Path:
        return self.broker / "status.json"

    @property
    def rollouts(self) -> Path:
        return self.root / "rollouts"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger"

    @property
    def attempts(self) -> Path:
        return self.ledger / "attempts.jsonl"

    @property
    def events(self) -> Path:
        return self.ledger / "events.jsonl"

    @property
    def agent_calls(self) -> Path:
        return self.ledger / "agent-calls.jsonl"

    # agent writable
    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def agent_home(self) -> Path:
        return self.root / "agent-home"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def requests(self) -> Path:
        return self.broker / "requests"

    # private (masked in every sandbox)
    @property
    def harness(self) -> Path:
        return self.root / "harness"

    @property
    def state(self) -> Path:
        return self.harness / "state.json"

    @property
    def lock(self) -> Path:
        return self.harness / "run.lock"

    @property
    def sessions(self) -> Path:
        return self.harness / "sessions"

    @property
    def manifests(self) -> Path:
        return self.harness / "manifests"

    @property
    def holdout_seeds(self) -> Path:
        return self.harness / "holdout-seeds.json"

    @property
    def holdout_dir(self) -> Path:
        return self.harness / "holdout"

    @property
    def holdout_results(self) -> Path:
        return self.holdout_dir / "results.json"

    @property
    def processed(self) -> Path:
        return self.harness / "processed-requests"

    @property
    def private_events(self) -> Path:
        return self.harness / "events.jsonl"

    @property
    def round_result(self) -> Path:
        return self.harness / "round-result.json"

    def public_dirs(self) -> list[Path]:
        return [self.bin_dir, self.handoff_in, self.references, self.responses, self.results,
                self.rollouts, self.submissions, self.ledger]

    def agent_writable(self) -> list[Path]:
        return [self.workspace, self.agent_home, self.tmp, self.requests]

    def create(self) -> None:
        for path in (self.public_dirs() + self.agent_writable()
                     + [self.harness, self.sessions, self.manifests, self.holdout_dir, self.processed]):
            path.mkdir(parents=True, exist_ok=True)
        # Codex requires an existing CODEX_HOME even for a no-auth local provider.
        (self.agent_home / ".codex").mkdir(exist_ok=True)
        self.harness.chmod(0o700)
