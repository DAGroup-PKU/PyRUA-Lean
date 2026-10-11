# PyRUA_Lean

**Contributor:** PyRUA-Lean contributors | **Paper:** TBD | **arXiv:** TBD | **Original code:** https://github.com/DAGroup-PKU/PyRUA-Lean

`PyRUA_Lean` runs a code-writing agent on RoboDojo: a language model writes Python against a small robot API (`robo`), one cell at a time, while the episode runs; each motion call becomes native joint actions. The same program-writing loop, the same six primitives and the same prompt serve every task: nothing is tuned, routed or stored per task. There is no VLA and no training entry; SAM3 (segmentation) is the only model the policy side loads. The adapter is evaluation-only and supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Inside the policy environment (any environment with a CUDA build of `torch`):

```bash
cd XPolicyLab/policy/PyRUA_Lean
bash install.sh              # XPolicyLab's server, PyRUA-Lean, the sam3 package
bash download_assets.sh      # SAM3's checkpoint (3.4 GB) into ./checkpoints/sam3.pt
```

`install.sh` takes a PyRUA-Lean checkout or pip requirement as its argument when the default is not the one to install. `download_assets.sh modelscope` fetches the checkpoint from the ModelScope mirror; `SAM3_CHECKPOINT_PATH` names a checkpoint kept elsewhere. The simulator's environment needs nothing beyond RoboDojo's own installation.

## Data Processing

Unsupported: there is no dataset. `process_data.sh` is omitted (evaluation-only adapter).

## Training

Unsupported: nothing is trained and there is no checkpoint directory. `train.sh` is omitted (evaluation-only adapter).

## Evaluation

The programs are written by a model behind an OpenAI-compatible chat endpoint:

```bash
export OPENAI_BASE_URL=https://YOUR_API_HOST/v1
export OPENAI_API_KEY=...
export PYRUALEAN_MODEL=gpt-6-astra          # the model's name at that endpoint

cd XPolicyLab/policy/PyRUA_Lean
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_blocks no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo

# Offline wiring check (no simulator, no model call)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo
```

`action_type` must be `joint`: the program's poses are solved to joint targets by PyRUA-Lean's own kinematics. `env_cfg_type` is `arx_x5`. `ckpt_name` is ignored; `no-checkpoint` is the conventional placeholder. Batch evaluation is not supported: one conversation drives one environment (`eval_batch: false`). The policy GPU holds SAM3 (about 8 GB). Each episode's programs, tool calls and transcript are kept under `result_dir` (`./results/<task>_l<layout>_s<seed>-<time>/`).

`EVAL_ENV_TYPE=debug` holds the robot where the debug client's observation has it: the loop, the action keys and their dimensions are exercised, and no episode is started.

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `observation` | `rgbd` (default): `setup_eval_env_client.sh` starts the simulator client with the depth image and the camera matrices switched on. `rgb`: RoboDojo's default observation |
| `robot` | `auto` (default): `robodojo_xpl` when the observation holds depth and the camera matrices, `robodojo_xpl_rgb` when it does not. Either name fixes the choice |
| `primitives` | `no-vla` (this adapter serves no VLA) |
| `runtime`, `model` | the agent that writes the programs: `openai` (a chat-completions endpoint) and the model's name there (or `PYRUALEAN_MODEL`) |
| `max_turns`, `budget_s` | an episode's decision budget (model replies that call a tool) and its wall-clock limit in seconds |
| `images`, `feedback` | `on-demand`: the model sees a camera when its program asks for it; `pure`: a tool call returns what the program printed and nothing else |
| `openai_options` | passed to the endpoint with every request (`max_tokens`, `extra_body.reasoning_effort`) and `nudges`, the text-only replies tolerated before the episode is ended |
| `poll_s` | the longest the policy server holds one `get_action` before answering "wait" |
| `wrap_up_s` | the longest the verdict's acknowledgement waits for the episode's record to be written |
| `result_dir` | where each episode's record is written, relative to this directory |

Environment variables: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PYRUALEAN_MODEL` (above); `SAM3_CHECKPOINT_PATH` (default `./checkpoints/sam3.pt`); `PYRUALEAN_RESULT_DIR` overrides `result_dir`.

### Observation

Besides the three colour images and the robot's state, the policy reads each camera's **depth image and intrinsic / extrinsic matrix**. They are fields of XPolicyLab's observation format (`depth`, `intrinsic_matrix`, `extrinsics_matrix`) that RoboDojo's simulator sends when `env_cfg` switches them on; `env_cfg/arx_x5.yml` ships with them off. With `observation: rgbd`, `setup_eval_env_client.sh` therefore starts the simulator client from a copy of RoboDojo's code and configuration (`client_root.py`, a few megabytes in `$TMPDIR`, removed afterwards) in which exactly these are changed:

```yaml
# env_cfg/arx_x5.yml
observation:
  vision:
    depth: true
    intrinsic_matrix: true
    extrinsic_matrix: true
# env_cfg/camera/camera_config.yml: the depth capture of each camera
annotator:
  cam_head:        {distance_to_image_plane_capture: {type: distance_to_image_plane, device: cpu}}
  cam_left_wrist:  {distance_to_image_plane_capture: {type: distance_to_image_plane, device: cpu}}
  cam_right_wrist: {distance_to_image_plane_capture: {type: distance_to_image_plane, device: cpu}}
```

The RoboDojo checkout itself is not edited, and `eval_result/` in the copy is the checkout's, so results, videos and resume manifests land where they always do. Tasks, layouts, physics, step limits and success checks are RoboDojo's own files, copied unchanged.

With `observation: rgb` the client runs from the checkout as it is, and the policy works from the colour images alone (`robodojo_xpl_rgb`): an object is measured from its masks in the cameras that see it. That is good to a few millimetres for things up to about 6 cm tall, worse for tall things only one camera sees, and it cannot measure what does not stand on the table (an object in the gripper, on another object, in a rack).

## Notes

- **What the policy reads.** The observation RoboDojo's client builds and nothing else: the colour images, the joint and end-effector state, the instruction, and the depth images and camera matrices described above. No object poses, no layout file, no success signal, no simulator handle: the policy server is a separate process that sees only what the protocol carries.
- **`deploy.py` differs from `demo_policy`'s in three places**, each because the policy is a program a model writes while the episode runs: `get_action` is asked again while it answers "wait" (a model thinks for minutes; no request is held longer than `poll_s`); an empty answer ends the episode, and RoboDojo judges the scene as it stands; every observation carries the episode's step count and limit (`additional_info["pyrualean"]`), which the program budgets its motions by, and the verdict is reported back through the protocol's `trial_end` once the episode is over. Nothing else is read from the environment.
- **Cost.** One episode is up to `max_turns` model calls with images; a rate-limited endpoint (HTTP 429) is waited out, not failed.
