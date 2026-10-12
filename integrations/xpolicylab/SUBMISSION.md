# Submitting `PyRUA_Lean` to XPolicyLab / the RoboDojo leaderboard

`PyRUA_Lean/` in this directory is the adapter: the directory a submission adds to
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) as `policy/PyRUA_Lean/`. It was written and verified
against RoboDojo `main` `cb2a37d` (2026-10-08) and the XPolicyLab commit that pins, `53c5da5`.

The submission is one fixed system for every task: a language model writes Python against a small robot API
while the episode runs, and each motion call becomes joint targets. It reads **RoboDojo's default observation
only** (three colour images, the robot's state, the instruction): no depth image, no camera matrices. Nothing is
trained and there is no policy checkpoint; SAM3 (segmentation) is the only model the policy side loads.
Model types: **Agent, LLM**.

## For the evaluator: install and run from this repository

```bash
# 1. the adapter, beside the other policies of the XPolicyLab checkout inside RoboDojo
git clone <this repository> PyRUA-Lean
ln -s "$PWD/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean" <RoboDojo>/XPolicyLab/policy/PyRUA_Lean
#    (or copy the directory there)

# 2. the policy environment: any environment with a CUDA build of torch and the torchvision built for it
#    (install.sh installs neither and keeps both as they are)
bash <RoboDojo>/XPolicyLab/policy/PyRUA_Lean/install.sh                     # XPolicyLab's server, PyRUA-Lean (this checkout), sam3
bash <RoboDojo>/XPolicyLab/policy/PyRUA_Lean/download_assets.sh             # SAM3's checkpoint, 3.4 GB

# 3. the model that writes the programs: any OpenAI-compatible chat endpoint
export OPENAI_BASE_URL=https://YOUR_API_HOST/v1
export OPENAI_API_KEY=...
export PYRUALEAN_MODEL=gpt-6-astra       # the model's name at that endpoint

# 4. RoboDojo's own entry point, from the RoboDojo root
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_Lean --task stack_blocks \
  --ckpt no-checkpoint --policy-env <policy_env> --eval-env <eval_env> --env-cfg arx_x5 \
  --action-type joint --seed 0 --policy-gpu 0 --env-gpu 0 --eval-num 2
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/PyRUA_Lean --ckpt no-checkpoint \
  --policy-env <policy_env> --env-cfg arx_x5 --action-type joint --eval-num native --seed 0
```

The simulator's environment needs nothing beyond RoboDojo's own installation, and RoboDojo's configuration is
used as it ships. The policy server and the simulator client may also run on different machines
(`scripts/robodojo.sh server` / `client`): the policy server is the only side that calls the model, so it can
stay with the submitting team while the official client connects to it. The client's reachability
probe opens the server's port and closes it without a websocket handshake, which XPolicyLab's server logs as a
handshake error (`EOFError` / `InvalidMessage`) before the client connects: it is not a failure.

**Model calls.** One episode is at most 100 model replies that call a tool (24 on average), each request
carrying the conversation so far and the images the program asked for (about 0.8 M input tokens an episode).
With GPT-6 Astra one episode cost about 3 USD and 12 minutes, so one eval seed (2,100 episodes) is on the order
of 6,000 USD of model calls and about a day with 16-24 tasks in parallel (a task's episodes run one after
another). A rate-limited endpoint (HTTP 429) is waited out, not failed.

## What the policy reads, and what is built in

The adapter's README ("Observation" and "Notes") is the full statement. In short:

- Read at run time: the colour images, the joint and end-effector state, the instruction; from the environment
  object, `deploy.py` adds the step count, the step limit, the task's name, the eval seed and the layout index.
  No depth, no camera matrices, no object poses, no layout file. The verdict reaches the policy server through
  the protocol's `trial_end` after the episode is over and is only recorded: the model never sees a success flag.
- Built in, the same for every task and episode: the cameras' poses and focal length, the arms' bases and joint
  limits, the table top's height and extent, the gripper's opening-to-width relation (RoboDojo's public
  configuration, or measured once in the simulator). `policy/GPT_6_Astra_Direct_EEF` builds in the same table height
  and arm bases, and reads the step limit from the environment the same way.
- One prompt for every task, with no task recipes. It states general facts about how RoboDojo judges an episode
  (both arms back at their start pose with the grippers open; positions checked to about a centimetre) that were
  read from RoboDojo's task code.
- `deploy.py` ends an episode the program has finished by clearing the environment's success flag and calling
  `is_episode_end()` once more, so that RoboDojo's own check judges the scene as it stands.

## Checks to re-run in a fork (XPolicyLab `CONTRIBUTING.md`, "Testing")

```bash
# 1. static (repository root)
git diff --check
bash -n policy/PyRUA_Lean/*.sh
python -m py_compile policy/PyRUA_Lean/model.py policy/PyRUA_Lean/deploy.py
grep -rnE 'cv2\.imdecode|np\.frombuffer|Image\.open' policy/PyRUA_Lean/      # nothing
grep -rn 'cv2\.imencode' policy/PyRUA_Lean/                                  # nothing
grep -rnE 'COLOR_BGR2RGB|COLOR_RGB2BGR|\.\.\., ::-1' policy/PyRUA_Lean/      # nothing

# 2. debug closed loop (no simulator, no model call); both must end in "[MAIN] eval finished"
cd policy/PyRUA_Lean
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 <policy_env> <eval_env>
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 <policy_env> <eval_env>

# 3. simulator, from the RoboDojo root
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_Lean --task stack_blocks \
  --ckpt no-checkpoint --policy-env <policy_env> --eval-env <eval_env> --env-cfg arx_x5 \
  --action-type joint --seed 0 --policy-gpu 0 --env-gpu 0 --eval-num 2
```

## Before the pull request is opened

1. **PyRUA-Lean has to be installable from a source the evaluators can reach.** `install.sh` installs `pyrualean`
   from the checkout the adapter is linked from, or from its argument (a checkout or a pip requirement); for a
   copied adapter the default is `git+https://github.com/DAGroup-PKU/PyRUA-Lean.git`. The RoboDojo robot is not
   on that repository's default branch yet: until it is, install from a checkout of this repository as above,
   and pin the released ref in `install.sh` when it is published.
2. Decide who serves the model during the official evaluation (see "Model calls" above).

## Pull request

Title: `[policy] PyRUA_Lean: add eval-only RoboDojo adapter`

````markdown
## Policy
- Name / paper / upstream repo: PyRUA-Lean / [arXiv 2610.01939](https://arxiv.org/abs/2610.01939) / https://github.com/DAGroup-PKU/PyRUA-Lean
- Type: Agent, LLM. A language model writes Python against a robot API while the episode runs; each motion call becomes joint targets.
- Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`
- Training support: eval-only (nothing is trained; no policy checkpoint)

## Components
- [x] install.sh
- [x] model.py (+ `__init__.py`)
- [x] images: no decoding and no channel swaps in the adapter
- [x] deploy.yml (standard key set incl. `protocol: ws` / `host` / `port`; `policy_name` matches the directory)
- [x] deploy.py: divergence from demo_policy explained in the policy README (Notes)
- [x] eval.sh + setup_eval_policy_server.sh + setup_eval_env_client.sh
- [x] process_data.sh / train.sh omitted (eval-only, declared above)
- [x] policy README with install / eval commands

## Testing
- [x] `bash -n` + `py_compile` pass
- [x] decode/encode grep: no hit
- [x] `EVAL_ENV_TYPE=debug` closed loop passes, with plain image arrays and with `DEBUG_OBS_ENCODED=1`:

```text
[TestEnv] Episode finished
[MAIN] eval finished
```

## Checkpoint (required for leaderboard evaluation)
```bash
bash policy/PyRUA_Lean/download_assets.sh    # SAM3 (segmentation); there is no policy checkpoint
```

## Limitations / notes
- The model is called through an OpenAI-compatible endpoint: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PYRUALEAN_MODEL`.
- One conversation drives one environment (`eval_batch: false`).
````
