<div align="center">

# PyRUA-Dex: Learning Programmatic Policies for Dexterous Bimanual Manipulation with Coding Agents

Feixiang&nbsp;Ruan<sup>1,4,\*</sup>, Guodong&nbsp;Lv<sup>1,\*</sup>, Kewei&nbsp;Zhang<sup>1,\*</sup>, Wenbo&nbsp;Huang<sup>1,\*</sup>, Yintianrun&nbsp;Zhang<sup>1</sup>, Qiang&nbsp;Wang<sup>1</sup>, Ruiyang&nbsp;Si<sup>1</sup>,
Shunyu&nbsp;Yang<sup>1</sup>, Shulong&nbsp;Jiang<sup>1</sup>, Weicheng&nbsp;Duan<sup>4</sup>, Rui&nbsp;Ni<sup>1</sup>, Jianxin&nbsp;Bi<sup>1,2</sup>, Lekai&nbsp;Wang<sup>1</sup>, Yuze&nbsp;Fan<sup>4</sup>,
Haobo&nbsp;Li<sup>4</sup>, Zeliang&nbsp;Fu<sup>4</sup>, Jiankun&nbsp;Zhang<sup>5</sup>, Duomin&nbsp;Wang<sup>3</sup>, Enze&nbsp;Xie<sup>3</sup>, Daquan&nbsp;Zhou<sup>1,†</sup>

<sup>1</sup>Peking&nbsp;University &nbsp; <sup>2</sup>National&nbsp;University&nbsp;of&nbsp;Singapore &nbsp; <sup>3</sup>NVIDIA<br>
<sup>4</sup>Tongji&nbsp;University &nbsp; <sup>5</sup>University&nbsp;of&nbsp;Chicago

<sup>\*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-8C0000.svg)](https://vme0522.github.io/PyRUA-Dex/)
![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey.svg)
![Code](https://img.shields.io/badge/Code-coming%20soon-lightgrey.svg)

[🌐 Project Page](https://vme0522.github.io/PyRUA-Dex/) · [🎬 Real-World Videos](https://vme0522.github.io/PyRUA-Dex/#real) · [📊 Results](https://vme0522.github.io/PyRUA-Dex/#results)

</div>

**PyRUA-Dex** is a framework that enables coding agents to develop dexterous manipulation through
programmatic policy, without the need of demonstrations. The programmatic policy is developed in two
stages: during exploration, multiple agents explore in parallel for task skills and ways to compose
them, and a distilling agent condenses the successful attempts into a task library and one program.
During refinement, this program is improved through randomized layouts for enhanced robustness and
execution efficiency.

<p align="center">
  <img src="assets/fig-teaser.png" width="100%" alt="PyRUA-Dex at a glance. Center: the real-world platform playing the piano, with tactile sensing at the fingertips. (a) Coding agents learn a task's program in simulation in two stages, exploration and refinement; the frozen program is deployed in the real world, and real-world feedback updates the simulation. (b) DexJoCo success rates. (c) Tasks learned for the real-world platform. (d) The Bench2Dex leaderboard. (e) The robot hands the programs drive. (f) The simulated and real-world tasks in five classes.">
</p>

## Code

The code is not open-sourced yet. We are organizing the PyRUA-Dex harness for release; it will be
published on this branch. **Coming soon.**

## Results

- **[Bench2Dex](https://github.com/Bench2Dex/Bench2Dex)**: PyRUA-Dex achieves SOTA with a score of
  **72.9**.
- **[DexJoCo](https://arxiv.org/abs/2605.16257)**: PyRUA-Dex reaches **81.0%** success under full
  randomization.
- On both benchmarks, PyRUA-Dex is ahead of every trained policy.
- **Real world**: on a bimanual platform with two 22-DoF hands, PyRUA-Dex learns to play piano,
  write calligraphy, and other tasks; learning a new piece from sheet music takes as little as
  30 minutes.

Task videos and the detailed results are on the
[project page](https://vme0522.github.io/PyRUA-Dex/).

## Citation

```bibtex
@article{ruan2026pyruadex,
  title   = {{PyRUA-Dex}: Learning Programmatic Policies for Dexterous Bimanual
             Manipulation with Coding Agents},
  author  = {Ruan, Feixiang and Lv, Guodong and Zhang, Kewei and Huang, Wenbo and
             Zhang, Yintianrun and Wang, Qiang and Si, Ruiyang and Yang, Shunyu and
             Jiang, Shulong and Duan, Weicheng and Ni, Rui and Bi, Jianxin and
             Wang, Lekai and Fan, Yuze and Li, Haobo and Fu, Zeliang and
             Zhang, Jiankun and Wang, Duomin and Xie, Enze and Zhou, Daquan},
  journal = {arXiv preprint},
  year    = {2026}
}
```
