# PyRUA-REAL

<p align="center"><b>PyRUA-REAL: Robot Exploration and Adaptive Learning through RSI in the Real World</b></p>

<p align="center">
  <a href="https://dagroup-pku.github.io/PyRUA-Lean/real/"><img src="https://img.shields.io/badge/Project-Page-8C0000.svg" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2611.06768"><img src="https://img.shields.io/badge/arXiv-2611.06768-b31b1b.svg" alt="Paper on arXiv"></a>
  <a href="https://github.com/DAGroup-PKU/PyRUA-Lean/tree/REAL"><img src="https://img.shields.io/badge/GitHub-REAL-101A2B.svg?logo=github" alt="Project website source"></a>
</p>

PyRUA-REAL turns trials on a physical robot into reusable Python skills. Across eight manipulation tasks, the robot averaged **91.1% success during development**. Seven tasks yielded programs that could run without a language-model call.

<details>
<summary>Abstract</summary>

How can a robot keep what a coding agent learns from a failed grasp or a successful recovery? PyRUA-REAL studies embodied recursive self-improvement (RSI). Agents run programs on a robot, inspect what happened, and carry both the revised code and a record of the trial into the next experiment. The study examines which feedback helps agents diagnose errors, whether skills transfer between tasks, and how much execution time a robot saves when it can run the resulting program on its own.

</details>

![Paper overview showing the agent workflow, physical manipulation results, and RoboDojo evaluation](https://dagroup-pku.github.io/PyRUA-Lean/real/assets/pyrua-real-teaser-final.png)

## How it works

A researcher agent chooses the next experiment. An engineer agent writes the Python program, runs it on the robot, and reports what happened. For example, in towel folding, depth and gripper logs revealed that the gripper was closing above the cloth. The engineer revised how the program estimated table height at the grasp point, then tested the change on the robot. Later experiments can reuse both the diagnosis and the working routine. At deployment, the program reads live sensor data and can adjust its actions through its own perception and recovery code. [Paper, Appendix A](https://arxiv.org/html/2611.06768v1#A1)

## Results

### Real-world manipulation

On a dual-arm Franka robot, the eight task success rates averaged **91.1%**. Each rate pools recorded attempts across program revisions during development. Ethernet insertion was the only task that needed online agent assistance for successful runs. [Task videos and results](https://dagroup-pku.github.io/PyRUA-Lean/real/#performance)

| Task | Development success rate |
| --- | ---: |
| Tic-tac-toe | 95.29% |
| Five-block stacking | 94.88% |
| Tower of Hanoi | 92.50% |
| Towel folding | 96.67% |
| Push T | 100.00% |
| Block sorting | 91.67% |
| Block storage | 98.89% |
| Ethernet insertion (online assistance) | 58.82% |

### Execution speed

On tic-tac-toe and five-block stacking, PyRUA-REAL's fixed programs achieved a **4.6–5.0× speedup** over PyRUA-Lean's online coding and a **9.9–11.5× speedup** over GPT-6-Astra's online action selection. For stacking, PyRUA-Lean spent 10.6 of its 13.7 minutes waiting for agent decisions. The fixed program completed the cycle in a median of 2.99 minutes. These timings include reset and exclude program development. [Paper, Figure 14](https://arxiv.org/html/2611.06768v1#S4.F14)

### RoboDojo

Across **42 simulated tasks**, fixed programs achieved **84.42% average success on development seed 0** and **80.73% on held-out seed 2**, with no language-model calls during evaluation. The benchmark gives equal weight to its five capability groups. [Results by capability group](https://dagroup-pku.github.io/PyRUA-Lean/real/#result-benchmark)

## Related

- [PyRUA-Lean](pyrua-lean.md) describes the online coding approach used in the execution-time comparison.
- The [REAL branch](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/REAL) contains the project website and its assets.

## Citation

```bibtex
@misc{wang2026pyruareal,
  title = {{PyRUA-REAL}: Robot Exploration and Adaptive Learning through {RSI} in the Real World},
  author = {Qiang Wang and Jianxin Bi and Shulong Jiang and Ruiyang Si and Wenbo Huang and Kaiqi Chen and Jiaxin Li and Liyun Li and Xuanyi Li and Daquan Zhou},
  year = {2026},
  eprint = {2611.06768},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2611.06768}
}
```
