"""Task-expert authoring harness and its eval suite.

An authoring agent (Claude Code or Codex, headless) writes a controller program -- a
"task expert" -- for a simulated robot task. This package provides: a linted brief, an OS-level sandbox, a broker
tool that is the agent's only way to run rollouts (and that keeps the attempt ledger),
bounded session loops with provider-reported cost, tamper manifests over the judge and
holdout seeds, harness-only holdout evaluation with a score-vs-cost curve, automatic
handoff bundles, a verified expert library and a multi-task eval suite.

Entry point: ``tools/expert_harness.py`` (``python -m datafarm.expert.cli``).
Design and operator guide: ``docs/reference/expert-harness.md``.
"""
