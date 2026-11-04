<div align="center" markdown>
![PyRUA-Lean: Python for Lean Robot-Use Agents](../assets/logo.svg#only-light){ width="60%" }
![PyRUA-Lean: Python for Lean Robot-Use Agents](../assets/logo-dark.svg#only-dark){ width="60%" }
</div>

<p align="center"><b>Fewer Tokens, Better Action: GPT-6 Astra Robot Agents with 14% Higher Success Rate but 65% Fewer Tokens</b></p>

<p align="center">
  <a href="https://dagroup-pku.github.io/PyRUA-Lean/"><img src="https://img.shields.io/badge/Project-Page-8C0000.svg" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.01939"><img src="https://img.shields.io/badge/arXiv-2610.01939-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/DAGroup-PKU/PyRUA-Lean"><img src="https://img.shields.io/badge/GitHub-PyRUA--Lean-101A2B.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg" alt="Hugging Face model"></a>
</p>

This page covers installing PyRUA-Lean, running an episode with GPT-6 Astra or PyRUA-Lean Coder v0.1-4B, and the
results of the paper.

??? abstract "Abstract"

    Vision-language model (VLM) agents can control robots through visual feedback and action primitives, but
    repeated LLM calls and redundant observations incur substantial token overhead. We introduce PyRUA-Lean, an
    interactive code-execution framework in which the agent composes robot primitives, including
    vision-language-action (VLA) policies, into Python cells that check and retry locally and return only the images
    and state they request. Across 700 simulated task instances from LIBERO-PRO, RoboTwin 2.0, and RoboCasa365, it
    raises success from 63.1% to 71.7% over tool calling with the same GPT-6 Astra planner and primitives, using 49%
    fewer LLM calls and 65% fewer input tokens on jointly solved instances. On RoboDojo, without any VLA policy or
    privileged information, it reaches 51.6% average success rate (Score 58.2), 19.4 points above the top entry of
    the public leaderboard. Based on the PyRUA-Lean framework, we train PyRUA-Lean Coder v0.1-4B, an open 4B model,
    with GRPO on 45 LIBERO-PRO instances; it improves from 18.8% to 44.1% on unseen task instances, surpassing the
    Qwen3.5-27B base model.

![PyRUA-Lean at a glance](../assets/overview.png#only-light)
![PyRUA-Lean at a glance](../assets/overview-dark.png#only-dark)

______________________________________________________________________

## Requirements

| Benchmark | Robot | Runs on |
|---|---|---|
| [RoboDojo](../robots/robodojo.md) | `robodojo_xpl_rgb` | [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), under RoboDojo's own evaluation client |
| [LIBERO-PRO](../robots/libero.md) | `libero` | the robot stack of [RPent](https://github.com/RLinf/RPent) |
| [RoboTwin 2.0](../robots/robotwin.md) | `robotwin` | the robot stack of RPent |
| [RoboCasa365](../robots/robocasa.md) | `robocasa` | the robot stack of RPent |

!!! note
    Python 3.10 or later. Agents run through the [Codex CLI](https://github.com/openai/codex) 0.155.1, Claude Code,
    or any OpenAI-compatible endpoint.

______________________________________________________________________

## Installation

Install RPent with the robot you want first ([Quick Start](../getting-started/quickstart.md)); RoboDojo installs
through XPolicyLab instead ([RoboDojo](../robots/robodojo.md#installation-and-resources)).

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git && cd PyRUA-Lean
uv pip install -e .       # into the Python environment of your RPent robot stack (or pip)
pyrualean api             # the robot API the agent sees; no simulator needed
npm install -g @openai/codex@0.155.1 && codex login      # the agent runtime
```

______________________________________________________________________

## Running an Episode

### GPT-6 Astra on LIBERO-PRO

```bash
# one LIBERO-PRO episode: GPT-6 Astra writes code against robo, at most 40 code cells
export RPENT_ROOT=/path/to/rpent
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
pyrualean summarize runs/demo
```

### PyRUA-Lean Coder v0.1-4B

```bash
# serve the model
python -m sglang.launch_server --model-path DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B --port 30000 \
    --context-length 65536 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --constrained-json-disable-any-whitespace

# one LIBERO-PRO episode on a task the model never saw in training, with the settings of the paper
export RPENT_ROOT=/path/to/rpent OPENAI_API_KEY=none
pyrualean play --robot libero --arm cells --suite libero_goal_swap --task 3 --seed 0 \
    --runtime openai --base-url http://127.0.0.1:30000/v1 --api-key-env OPENAI_API_KEY \
    --model DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B \
    --openai-max-tokens 8192 --openai-tool-choice required --openai-temperature 1.0 \
    --openai-extra-body '{"top_p": 0.95, "top_k": 20}' \
    --images on-demand --feedback pure --guides --max-turns 40 --budget-s 5400 \
    --out runs/coder-demo --cuda-device 0
pyrualean summarize runs/coder-demo
```

### RoboDojo

```bash
# the model behind an OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://YOUR_API_HOST/v1
export OPENAI_API_KEY=...
export PYRUALEAN_MODEL=gpt-6-astra

# from the RoboDojo checkout, RoboDojo's own entry point
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_Lean --task stack_blocks \
    --ckpt no-checkpoint --policy-env <policy env> --eval-env <simulator env> \
    --env-cfg arx_x5 --action-type joint --seed 0 --eval-num native
```

______________________________________________________________________

## Results

### Code Cells Against Tool Calling

GPT-6 Astra, same primitives; calls, tokens and cost per solved episode on the task instances both agents solved.

| Benchmark | Success, tool calling | Success, PyRUA-Lean | LLM calls | Input tokens | Cost |
|---|---|---|---|---|---|
| LIBERO-PRO | 83.0 | **94.0** | 21.4 → 8.4 | 1.23M → 273k | $2.49 → $0.79 |
| RoboTwin 2.0 | 60.0 | **69.2** | 12.2 → 8.9 | 415k → 245k | $1.00 → $0.66 |
| RC365 atomic | 78.9 | **86.7** | 10.4 → 5.6 | 261k → 171k | $0.62 → $0.54 |
| RC365 composite | 34.4 | **39.4** | 28.4 → 14.9 | 1.22M → 603k | $2.01 → $1.25 |
| **All** | 63.1 | **71.7** | 17.0 → 8.7 | 788k → 276k | $1.63 → $0.74 |

### RoboDojo

Success rate %, the leaderboard's average over its five dimensions; PyRUA-Lean with GPT-6 Astra; top entries of a snapshot of the public
leaderboard taken on October 9, 2026; the official evaluation was run by the RoboDojo team
([full results](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/evaluation/robodojo_summary_official.md)).

| Entry | Type | Average | Gen. | Prec. | Long-Hor. | Memory | Open |
|---|---|---|---|---|---|---|---|
| **PyRUA-Lean**, official evaluation | Agent | **50.35** | 43.17 | **38.25** | 43.75 | **68.33** | 58.25 |
| **PyRUA-Lean**, our run | Agent | **51.62** | **45.83** | 37.25 | **48.00** | 63.00 | **64.00** |
| VPP2 | WAM | 32.26 | 29.50 | 35.17 | 37.00 | 55.56 | 4.08 |
| PhysicalRSI | Agent + VLA | 31.38 | 15.62 | 32.50 | 37.33 | 46.56 | 24.92 |
| Awomo-0.5 | WAM | 29.64 | 23.84 | 33.50 | 26.75 | 41.22 | 22.92 |
| Simate-beta | VLA | 27.96 | 27.95 | 26.92 | 43.42 | 33.00 | 8.50 |

### PyRUA-Lean Coder v0.1-4B

Success rate %, PyRUA-Lean framework.

| Planner | Trained on (45) | LIBERO-PRO other (155) | RoboTwin 2.0 (250) | RC365 atomic (90) | RC365 composite (160) | All unseen (655) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 37.8 | 13.5 | 14.0 | 52.2 | 12.5 | 18.8 |
| Qwen3.5-27B | 64.4 | **38.1** | 20.8 | 65.6 | 13.1 | 29.2 |
| **PyRUA-Lean Coder v0.1-4B** | **71.1** | 37.4 | **52.4** | **76.7** | **19.4** | **44.1** |

______________________________________________________________________

## Related

- [How It Works](../getting-started/how-it-works.md): what the agent writes, what `robo` returns, and what the
  model reads (Section 3 of the paper).
- [Agent Configuration](../guide/agents.md) and [Running Episodes](../guide/running.md): runtimes, models and the
  options of `pyrualean play`.
- [Robots](../robots/index.md): each robot with the API reference its agent reads.
- [RoboDojo](../robots/robodojo.md): the XPolicyLab adapter and its settings (Section 4).
- [PyRUA-Lean Coder v0.1-4B](../models/coder.md): the open 4B model and its evaluation (Section 5).

## Citation

```bibtex
@article{si2026fewer,
  title={Fewer tokens, better action: GPT-6 Astra robot agents with 14\% higher success rate but 65\% fewer tokens},
  author={Si, Ruiyang and Bi, Jianxin and Yang, Shunyu and Ni, Rui and Huang, Wenbo and Wang, Qiang and Jiang, Shulong and Zhang, Yintianrun and Ruan, Feixiang and Zhang, Kewei and Wang, Duomin and Li, Xiuyu and Feng, Haiwen and Dong, Zhen and Zhou, Daquan},
  journal={arXiv preprint arXiv:2610.01939},
  year={2026}
}
```
