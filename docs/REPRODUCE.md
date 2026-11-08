# Reproducing the paper

Every experiment is a learn campaign (`python -m pyrualean.learn.campaign`) followed by one or more
evaluations. All commands run from the repository root after `source tools/env.sh`, with the
actor's key in `ACTOR_API_KEY` ([MODELS.md](MODELS.md)). Run directories go under `runs/`
(git-ignored). Model calls cost money: try `configs/learn/robodojo/smoke.json` and
`configs/eval/robodojo/smoke.json` first.

| Paper | What | Configs |
|---|---|---|
| §4.1, Table 1, Fig. 3 | 25 RoboDojo tasks, w/o CE and w/ CE, tested under $C^{orig}$ against the empty library | `configs/learn/robodojo/group{1..5}.json`; `configs/eval/robodojo/test-orig.json`, `test-empty.json` |
| §4.2.1, Fig. 1b, 4 | the 16 criteria-limited tasks, every episode judged by $C^{orig}$ and $C^{faith}$ | `configs/eval/robodojo/test-dual16.json` |
| §4.2.2, Fig. 5 | the learner reads $C^{faith}$ (ten tasks) | `configs/learn/robodojo/faithful10.json`; `configs/eval/robodojo/test-faithful10.json` |
| §4.3, App. G–H | ten source libraries distilled into one, applied to ten other RoboDojo tasks | `transfer/robodojo/`; `configs/eval/robodojo/transfer-rest10*.json` |
| §4.3, App. I–J | RoboDojo libraries translated to LIBERO-PRO and RoboCasa365; RoboCasa365's own loop | `transfer/xbench/`, `transfer/robocasa/`; `configs/learn/robocasa/*.json`; `configs/eval/libero/`, `configs/eval/robocasa/` |

## Protocol

- Development: evaluation seed 1, layouts 0–4, every round; testing: seed 0, layouts 0–19, which
  no learner sees. Layouts the benchmark marks unstable are replaced by the next one in the
  official order (`"pool": 30` in the eval plans).
- Six rounds including round 0 (five rounds of improvement); rollback to the best complete
  earlier batch when a complete batch has two or more fewer successes.
- Each task has its own library, at most 40,000 characters; learner sessions of up to 2 attempts
  of 120 minutes, at most 8 verification rollouts, 3 per dev layout.
- Per actor episode: at most 100 cells, 200 model requests, 20 nudges, 90 minutes of policy time,
  and the task's own native-action limit.

The groups are fixed in the configs; the two settings use the same config and differ only in the
`--worker-scoring` switch, which is recorded at the first `run` and cannot change afterwards.

## 1. Learning (§4.1)

```bash
mkdir -p runs/configs
for g in 1 2 3 4 5; do
  python tools/render_config.py configs/learn/robodojo/group$g.json runs/configs/group$g.json
  for s in wo_ce w_ce; do
    python -m pyrualean.learn.campaign runs/learn/$s/group$g init --config runs/configs/group$g.json
  done
done
python -m pyrualean.learn.campaign runs/learn/wo_ce/group1 check-services   # both models answer
bash tools/run_learn.sh runs/learn/wo_ce/group1 --worker-scoring off        # w/o CE
bash tools/run_learn.sh runs/learn/w_ce/group1 --worker-scoring on          # w/ CE
python -m pyrualean.learn.campaign runs/learn/w_ce/group1 report            # dev curves
```

A campaign is resumable: running `tools/run_learn.sh` again continues it. When a model service
fails repeatedly it writes `PAUSED.json` and stops starting work; fix the service and run again.
`slots.gpus` / `slots.per_gpu` in the rendered config choose the GPUs (about 45 GB per episode).

## 2. Testing under the original criteria (§4.1)

```bash
for s in wo_ce w_ce; do
  CAMPAIGNS=$CE_ROOT/runs/learn/$s python tools/render_config.py \
      configs/eval/robodojo/test-orig.json runs/configs/test-orig-$s.json
  python eval/robodojo/export.py runs/eval/test-orig-$s --config runs/configs/test-orig-$s.json
  bash eval/robodojo/run_eval_env.sh runs/eval/test-orig-$s
done
CAMPAIGNS=$CE_ROOT/runs/learn/wo_ce python tools/render_config.py \
    configs/eval/robodojo/test-empty.json runs/configs/test-empty.json
python eval/robodojo/export.py runs/eval/test-empty --config runs/configs/test-empty.json
bash eval/robodojo/run_eval_env.sh runs/eval/test-empty

python eval/robodojo/run.py runs/eval/test-orig-w_ce report                    # per-task and macro rates
python eval/robodojo/compare.py runs/eval/test-orig-w_ce runs/eval/test-orig-wo_ce   # paired by layout
```

`eval/robodojo/export.py` copies each task's active library (as its episodes read it) and freezes the
campaign's settings; `eval/robodojo/run.py` refuses to start if the code, the benchmark or the settings
changed since. Running `eval/robodojo/run_eval_env.sh` again resumes. See [`eval/robodojo/README.md`](../eval/robodojo/README.md).

## 3. Re-judging with the instruction-faithful criteria (§4.2.1)

Build `faithful_criteria/build/v2-b9` first ([SETUP.md](SETUP.md), [`faithful_criteria/README.md`](../faithful_criteria/README.md)).

```bash
for s in wo_ce w_ce; do
  CAMPAIGNS=$CE_ROOT/runs/learn/$s python tools/render_config.py \
      configs/eval/robodojo/test-dual16.json runs/configs/test-dual16-$s.json
  python eval/robodojo/export.py runs/eval/test-dual16-$s --config runs/configs/test-dual16-$s.json
  bash eval/robodojo/run_eval_env.sh runs/eval/test-dual16-$s
done
python faithful_criteria/dual_table.py runs/eval/test-dual16-w_ce runs/eval/test-dual16-wo_ce
python eval/robodojo/compare.py runs/eval/test-dual16-w_ce runs/eval/test-dual16-wo_ce --criteria orig
python eval/robodojo/compare.py runs/eval/test-dual16-w_ce runs/eval/test-dual16-wo_ce --criteria faith
```

Every episode is judged by $C^{faith}$ (the ledger's `success`) and, alongside, by the original
criteria (the `v1 shadow final` line of its client log); an episode ends only when both have a
verdict. The libraries were learned against the original RoboDojo tree, so the plan lists every
task in `allow_identity_mismatch`: only the judging code differs.

## 4. Reading the instruction-faithful criteria (§4.2.2)

```bash
python tools/render_config.py configs/learn/robodojo/faithful10.json runs/configs/faithful10.json
python -m pyrualean.learn.campaign runs/learn/faithful10 init --config runs/configs/faithful10.json
bash tools/run_learn.sh runs/learn/faithful10 --worker-scoring on
FAITHFUL_CAMPAIGN=$CE_ROOT/runs/learn/faithful10 python tools/render_config.py \
    configs/eval/robodojo/test-faithful10.json runs/configs/test-faithful10.json
python eval/robodojo/export.py runs/eval/test-faithful10 --config runs/configs/test-faithful10.json
bash eval/robodojo/run_eval_env.sh runs/eval/test-faithful10
python faithful_criteria/dual_table.py runs/eval/test-faithful10 runs/eval/test-dual16-w_ce
```

The campaign's `source_root` is the faithful build, so its learner reads $C^{faith}$ and its
episodes are judged by both criteria.

## 5. Transfer to new RoboDojo tasks (§4.3)

The ten source tasks are groups 1 and 2; none of them is one of the ten target tasks.
Distill one library per setting from the finished campaigns as described in
[`transfer/README.md`](../transfer/README.md), then:

```bash
for s in wo_ce w_ce; do
  CAMPAIGNS=$CE_ROOT/runs/learn/$s TRANSFER_LIBRARY=/path/to/distilled-$s/library-v1 \
      python tools/render_config.py configs/eval/robodojo/transfer-rest10.json runs/configs/transfer-$s.json
  python eval/robodojo/export.py runs/eval/transfer-$s --config runs/configs/transfer-$s.json
  bash eval/robodojo/run_eval_env.sh runs/eval/transfer-$s
done
CAMPAIGNS=$CE_ROOT/runs/learn/wo_ce python tools/render_config.py \
    configs/eval/robodojo/transfer-rest10-empty.json runs/configs/transfer-empty.json
python eval/robodojo/export.py runs/eval/transfer-empty --config runs/configs/transfer-empty.json
bash eval/robodojo/run_eval_env.sh runs/eval/transfer-empty
python eval/robodojo/compare.py runs/eval/transfer-wo_ce runs/eval/transfer-empty --criteria orig
```

The target tasks take the execution settings of a finished campaign (`settings_task`), so the
actor runs exactly as it did while learning.

## 6. LIBERO-PRO and RoboCasa365 (§4.3)

- **Translation**: `transfer/xbench/` turns the two distilled RoboDojo libraries into
  `libero_rgb` and `robocasa_rgb` libraries.
- **LIBERO-PRO**: `configs/eval/libero/xbench-10swap-t569.json` and `xbench-goalswap-t159.json`
  run the empty library and both translated libraries on `libero_10_swap` tasks 5, 6, 9 and
  `libero_goal_swap` tasks 1, 5, 9, seeds 0–9, paired with the empty library. See
  [`eval/libero/README.md`](../eval/libero/README.md).
- **RoboCasa365, in-benchmark loop**: `configs/learn/robocasa/atomic5.json` and `composite5.json`
  (the ten source tasks), each in both settings as in section 1; `transfer/robocasa/` distills each
  setting's ten libraries into one.
- **RoboCasa365, held-out tasks**: `configs/eval/robocasa/heldout10-{atomic,composite}.json` run
  the translated and the in-benchmark libraries on ten target tasks, seeds 10–19, and
  `heldout10-*-v0.json` the empty library; `report --with` pairs them. `atomic-zero-shot.json` and
  `composite-zero-shot.json` are the zero-shot runs of the primitive alignment (App. J). See
  [`eval/robocasa/README.md`](../eval/robocasa/README.md).

## What to expect

Rates move by several points between reruns: verdicts on one layout change often between two
runs of the same library, so compare settings layout by layout on the test layouts
(`eval/robodojo/compare.py`) rather than by development curves. The actor and learner are hosted models;
their versions behind an endpoint may change over time.
