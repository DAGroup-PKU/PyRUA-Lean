# Submitting `PyRUA_Lean` to XPolicyLab / the RoboDojo leaderboard

`PyRUA_Lean/` in this directory is the adapter: copy it to `XPolicyLab/policy/PyRUA_Lean/` of a fork of
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) and open the pull request below. It was written and
verified against RoboDojo `main` `cb2a37d` (2026-10-08) and the XPolicyLab commit that pins, `53c5da5`.

Before the pull request is opened:

1. **PyRUA-Lean has to be installable from a public source.** `install.sh` installs `pyrualean` from
   `PYRUALEAN_SOURCE` (default `git+https://github.com/DAGroup-PKU/PyRUA-Lean.git`). The RoboDojo robot
   (`robodojo_xpl`, the `robodojo-rgb` branch) is not on that repository's default branch yet: publish it and
   pin the released ref in `install.sh`.
2. Fill in the README's header (`Paper`, `arXiv`) and the contributor line.
3. Decide who serves the model during the official evaluation: the adapter talks to any OpenAI-compatible chat
   endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PYRUALEAN_MODEL`), and one episode is up to 100 model calls
   with images.

## Checks to re-run in the fork (XPolicyLab `CONTRIBUTING.md`, "Testing")

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

## Pull request

Title: `[policy] PyRUA_Lean: add RoboDojo adapter (code-writing agent, evaluation-only)`

```markdown
## Policy
- Name / paper / upstream repo: PyRUA_Lean / <paper> / https://github.com/DAGroup-PKU/PyRUA-Lean
- Supported: bench_name=RoboDojo, env_cfg_type=arx_x5, action_type=joint
- Training support: eval-only. Nothing is trained: a language model writes Python against a small robot API
  while the episode runs, and each motion call becomes joint targets. There is no checkpoint to release; SAM3
  (segmentation) is the only model the policy side loads.

## Components
- [x] install.sh
- [x] model.py (+ __init__.py)
- [x] images: no decoding and no channel conversion in the adapter (the server's decoded arrays are used as they are)
- [x] deploy.yml (standard key set incl. protocol: ws / host / port, policy_name matches the directory)
- [x] deploy.py — diverges from demo_policy in three places, explained in the policy README ("Notes"):
      `get_action` is asked again while the policy answers "wait" (a model thinks for longer than one request
      may take), an empty answer ends the episode and RoboDojo judges the scene as it stands, and the verdict
      is reported through the protocol's `trial_end`
- [x] eval.sh + setup_eval_policy_server.sh + setup_eval_env_client.sh (eval.sh is demo_policy's, unchanged)
- [ ] process_data.sh / train.sh — omitted: eval-only (no dataset, no training)
- [x] policy README with install / eval commands

## Testing
- [x] bash -n + py_compile pass
- [x] decode/encode grep: no hit
- [x] EVAL_ENV_TYPE=debug closed loop passes, also with DEBUG_OBS_ENCODED=1 (log tail below)
- [x] Simulator eval through `scripts/robodojo.sh eval` on RoboDojo main (cb2a37d): <task=..., success=...>

## Checkpoint (required for leaderboard evaluation)
`bash policy/PyRUA_Lean/download_assets.sh` (SAM3 from Hugging Face `facebook/sam3`, or `... modelscope`).
There is no policy checkpoint.

## Limitations / notes
- **Observation.** Besides the colour images and the robot's state, the policy reads each camera's depth image
  and intrinsic / extrinsic matrix (fields of XPolicyLab's observation format that `env_cfg/arx_x5.yml` has
  switched off). With `observation: rgbd` (the default in deploy.yml) `setup_eval_env_client.sh` starts the
  simulator client from a copy of RoboDojo's code and configuration with those three switches on; the checkout
  is not edited and results are written to its `eval_result/`. With `observation: rgb` the client runs as it
  is and the policy works from the colour images alone (lower accuracy on tall or held objects).
- The agent's model is served by an OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `PYRUALEAN_MODEL`); the numbers above are with GPT-6 Astra.
- One conversation drives one environment (`eval_batch: false`).
```
