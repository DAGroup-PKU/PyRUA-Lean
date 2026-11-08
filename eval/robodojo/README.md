# eval: evaluate learned libraries on held-out RoboDojo layouts

The code here is apart from the learning loop: it imports nothing of `pyrualean.learn` or
`pyrualean.evolve`. An episode still calls `pyrualean.play.play_episode` with the arguments
`pyrualean.learn.episode` uses, so the actor gets the prompt and execution of the campaign's own
episodes.

## Consistency

- `episode.identity()` hashes, with the algorithm of `pyrualean.learn.episode`, the policy code
  (`src/pyrualean/` without `learn/` and `evolve/`), the benchmark (`env_cfg/`, `task/`, `src/`
  of the source root), the execution settings and the SAM 3 checkpoint.
- `export.py` requires it to equal the identity the source episodes recorded, and `run.py`
  checks it again at start and in every episode process: if the prompt or policy code changed
  since learning, the evaluation refuses to run. A plan lists in `allow_identity_mismatch` the
  tasks for which a difference is intended (e.g. a source root with other judging code).
- What is exported is the library directory the source episodes read; its hash must equal the
  one recorded in their `result.json` and is checked again after every episode.
- The report's `distinct_prompts` counts the prompt hashes of all episodes.

## Files

| File | Role |
|---|---|
| `export.py` | reads finished campaigns, copies each task's library to `<root>/libraries/<task>` and writes the frozen `<root>/config.json` |
| `episode.py` | one episode in its own process (`python eval/robodojo/episode.py request.json`) |
| `run.py` | the scheduler: concurrency limit, GPU admission, watchdog, infrastructure retries, pause on provider failures, ledger, resume, report |
| `run_eval_env.sh` | starts `run.py` detached (setsid); the actor's key from the variable `execution.api_key_env` names; log in `<root>/run.log` |
| `compare.py` | pairs two roots layout by layout (successes, layouts only one side solved, sign test), by either criteria of a dual-judged root |
| [`../empty_library/`](../empty_library) | the empty library (a round-0 view) |
| `tests/test_eval.py` | offline tests with fake episodes and fake GPUs |

The plans of the paper are in [`configs/eval/robodojo/`](../../configs/eval/robodojo) (templates,
see [docs/REPRODUCE.md](../../docs/REPRODUCE.md)).

## Plans

A plan names, per task, the campaign whose library runs (`campaign`), how many judged layouts
it needs (`layouts`) and below which layout number replacements may be taken (`pool`).

- `"source": "active"`: the campaign's active revision (campaigns run without their own test);
  otherwise the `final` revision of its `test.json`.
- `"library": DIR`: run that directory (a library made elsewhere, e.g. distilled from other tasks,
  or the empty one); the campaign only supplies the settings, checked against its round-0 dev
  episodes. With `"settings_task": OTHER` those are the episodes of the campaign's task `OTHER`,
  for a task no campaign learned.
- `"only": [N, ...]`: exactly these layouts, none replaced.
- A plan's `source_root` replaces the campaigns' RoboDojo tree, e.g. `faithful_criteria/build/v2-b9`
  for dual judging; a plan's `execution` is put over the campaigns' (e.g. another runtime).

## Scheduling

- `concurrency`: the most episodes at once (your endpoint's limit).
- GPU admission: a GPU of `gpus` takes an episode when it runs fewer than `per_gpu` of ours and
  its free memory (all users), less what our started episodes may still take, is at least
  `need_gib` (an episode peaks at about 45 GB).
- An episode without a verdict (boot error, crash, identity mismatch, watchdog) is
  infrastructure: retried at most `budgets.max_infra_retries` times, then missing, never a
  failure. A provider failure uses no retry; `pause_after_provider_failures` in a row write
  `PAUSED.json` and nothing new starts until `run` again.
- A layout the benchmark finds unstable is not retried and is replaced by the next layout below
  `pool`, as RoboDojo's own evaluation does.

## Commands

```bash
source tools/env.sh
python tools/render_config.py configs/eval/robodojo/test-orig.json runs/configs/test-orig.json   # CAMPAIGNS=...
python eval/robodojo/export.py runs/eval/test-orig --config runs/configs/test-orig.json
bash eval/robodojo/run_eval_env.sh runs/eval/test-orig            # again to resume
tail -f runs/eval/test-orig/run.log
python eval/robodojo/run.py runs/eval/test-orig report
python eval/robodojo/compare.py runs/eval/A runs/eval/B [--criteria orig]
```

Offline tests: `bash tools/test_offline.sh` (or
`PYTHONPATH=src python -m pytest -q --rootdir=eval eval/robodojo/tests`).
