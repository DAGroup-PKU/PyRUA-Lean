"""Task-specific experience for an online code policy on RoboDojo: learn on dev, test frozen.

The protocol (``docs/REPRODUCE.md`` of the repository):

1. Each task has its own library lineage (``library/<task>/``), starting from the empty
   revision v0, the zero-shot baseline.
2. Learning, per task and round: the active revision runs on the task's dev layouts (official
   eval seed 1) -> a coding-agent worker (Claude Code, headless) reads those episodes, writes
   one change to the task's skills/lesson cards, and checks code fixes by model-free replay on
   dev layouts -> the change becomes the next active revision.  No per-round gate: a revision
   whose dev successes drop clearly below the best earlier revision's is rolled back.
3. Test: the final revision and v0 run on the same unseen layouts (another official eval
   seed), once each, and are compared pair by pair (here, or with ``eval/`` as in the paper).

Modules: :mod:`.config` (settings), :mod:`.slots` (GPU slots shared across processes),
:mod:`.episode` (one episode in its own process), :mod:`.runner` (jobs, retries, ledger,
budget), :mod:`.worker` (the coding-agent session), :mod:`.replay` (the worker's replay
command), :mod:`.campaign` (orchestration, report, CLI).
"""
