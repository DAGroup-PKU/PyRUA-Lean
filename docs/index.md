<div align="center" markdown>
![PyRUA-Lean: Python for Lean Robot-Use Agents](assets/logo.svg#only-light){ width="72%" }
![PyRUA-Lean: Python for Lean Robot-Use Agents](assets/logo-dark.svg#only-dark){ width="72%" }
</div>

<p align="center">
  <a href="https://dagroup-pku.github.io/PyRUA-Lean/"><img src="https://img.shields.io/badge/Project-Page-8C0000.svg" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.01939"><img src="https://img.shields.io/badge/arXiv-2610.01939-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/DAGroup-PKU/PyRUA-Lean"><img src="https://img.shields.io/badge/GitHub-PyRUA--Lean-101A2B.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg" alt="Hugging Face model"></a>
  <a href="https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

______________________________________________________________________

## Introduction

**PyRUA-Lean** (*Py*thon for lean *R*obot-*U*se *A*gents, the robot counterpart of computer-use agents)
turns a robot into a Python object. A tool-calling agent spends a full LLM call on every small step of a task
and re-sends everything before it each time. A PyRUA-Lean agent writes code against `robo` instead, so one call
can find an object, move above it, grasp, check the gripper and try again.

![PyRUA-Lean at a glance](assets/overview.png)

PyRUA-Lean has three parts so far:

- **The framework.** The agent composes robot primitives, VLA policies among them, into Python cells that check
  and retry locally and return only the images and state it asks for. On 700 task instances from LIBERO-PRO,
  RoboTwin 2.0 and RoboCasa365, with the same GPT-6 Astra planner and the same primitives as tool calling, success
  goes from **63.1% to 71.7%**, with **49% fewer LLM calls** and **65% fewer input tokens** on the instances both
  agents solve.
- **RoboDojo.** With this library's own motion, gripper and perception primitives and the colour images of the
  benchmark's default observation, PyRUA-Lean reaches **51.62% average success rate (Score 58.20)**, first on all
  five dimensions of the public leaderboard (snapshot of October 9, 2026).
- **PyRUA-Lean Coder v0.1-4B.** Qwen3.5-4B trained with GRPO in the PyRUA-Lean framework, on 45 LIBERO-PRO
  instances, goes from **18.8% to 44.1%** on 655 task instances it never saw in training, above the Qwen3.5-27B base
  model (29.2%).

## Quick Start

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git && cd PyRUA-Lean
uv pip install -e .       # into the Python environment of your RPent robot stack (or pip)
pyrualean api             # the robot API the agent sees; no simulator needed

# one LIBERO-PRO episode: GPT-6 Astra writes code against robo, at most 40 code cells
export RPENT_ROOT=/path/to/rpent
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
pyrualean summarize runs/demo
```

LIBERO-PRO, RoboTwin 2.0 and RoboCasa365 run on [RPent](https://github.com/RLinf/RPent)'s robot stacks (the
[Quick Start](getting-started/quickstart.md) sets one up step by step); RoboDojo runs through XPolicyLab and needs no
RPent ([RoboDojo](robots/robodojo.md)).

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quick Start](getting-started/quickstart.md)**

    Install, run one LIBERO-PRO episode, and look at what the agent did.

-   :material-language-python: **[How It Works](getting-started/how-it-works.md)**

    What the agent writes, what `robo` returns, and what the model reads.

-   :material-robot-industrial: **[Robots](robots/index.md)**

    RoboDojo through XPolicyLab; LIBERO-PRO, RoboTwin 2.0 and RoboCasa365 on RPent. Each with the API reference its agent reads.

-   :material-brain: **[PyRUA-Lean Coder v0.1-4B](models/coder.md)**

    An open 4B model trained to write the cells, on Hugging Face.

</div>

## Citation

If you use PyRUA-Lean, please cite our paper:

```bibtex
@article{si2026fewer,
  title={Fewer tokens, better action: GPT-6 Astra robot agents with 14\% higher success rate but 65\% fewer tokens},
  author={Si, Ruiyang and Bi, Jianxin and Yang, Shunyu and Ni, Rui and Huang, Wenbo and Wang, Qiang and Jiang, Shulong and Zhang, Yintianrun and Ruan, Feixiang and Zhang, Kewei and Wang, Duomin and Li, Xiuyu and Feng, Haiwen and Dong, Zhen and Zhou, Daquan},
  journal={arXiv preprint arXiv:2610.01939},
  year={2026}
}
```
