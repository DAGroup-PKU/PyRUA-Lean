<div align="center">

# PyRUA-Lean

**Fewer Tokens, Better Action: GPT-6 Astra Robot Agents with 14% Higher Success Rate but 65% Fewer Tokens**

[![项目主页](https://img.shields.io/badge/项目-主页-8C0000.svg)](https://dagroup-pku.github.io/PyRUA-Lean/)
![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

**PyRUA-Lean**（*Py*thon for lean *R*obot-*U*se *A*gents，对应电脑操作领域的 computer-use agent）把机器人变成一个
Python 对象。工具调用式的 agent 在任务的每一个小步上都要花一次完整的 LLM 调用，而且每次都要把之前的全部内容重发一遍；
PyRUA-Lean 的 agent 则对着 `robo` 写代码，一次调用就能完成找到物体、移到上方、抓取、检查夹爪、失败重试这一整串动作。

<p align="center">
  <img src="docs/assets/results-overview.png" width="100%"
       alt="tool calling 与 PyRUA-Lean 的对比：九个子任务集上成功率都相同或更高（总体 63.1% 到 71.7%）；每个解出的任务输入 token 少 1.5 到 4.5 倍，按官方价格便宜 1.2 到 3.1 倍">
</p>

## agent 写的是什么

agent 只有一个工具 `python(code)`，在一个装着 `robo` 的持久命名空间里执行。下面是 GPT-6 Astra 在一个 LIBERO-PRO
episode 里实际写的一个 cell（论文 Figure 2），作用是把手里的碗放到盘子上。代码原样照录，三行注释是我们加的；`plate`
是前面的 cell 找到的：

```python
# 组合：根据场景几何算出放置目标
held_at_plate = robo.segment(point=(314, 795))
print('bowl above plate', held_at_plate)
assert held_at_plate.found
correction = np.array(plate.world_xyz[:2]) - np.array(held_at_plate.world_xyz[:2])
place_xy = np.array(robo.state().eef_pos[:2]) + correction
surface_map = robo.world_map()
bowl_patch = surface_map[290:365, 740:850]
bowl_heights = bowl_patch[:,:,2]
bowl_heights = bowl_heights[(bowl_heights>1.04) & (bowl_heights<1.17)]
bottom_z = float(np.quantile(bowl_heights, 0.02))
placement_z = plate.world_xyz[2] + robo.state().eef_pos[2] - bottom_z + 0.007
print('correction', correction, 'bottom', bottom_z, 'placement z', placement_z)
assert np.linalg.norm(correction)<0.07 and 0.95<placement_z<1.05
# 组合：在同一个 cell 里下放、检查、松开
lower = robo.move_to([*place_xy, placement_z], gripper=+1, step_clip=0.012, tol=0.005, max_steps=100)
print('lower', lower)
if lower.reached and not robo.done:
    print('release', robo.release())
print('done', robo.done)
# 选择：只有任务还没完成时才要图像
if not robo.done:
    robo.show('agentview')
```

这个 cell 打印了五行结果，没有要任何图像：碗已经放到盘子上，任务完成了。换成 tool calling，同样这一步要四次 LLM
调用，而且每次移动都返回三张相机图像。

## 快速上手

PyRUA-Lean 驱动的是 [RPent](https://github.com/RLinf/RPent) 的机器人环境：先按你要用的机器人装好 RPent
（版本、模型权重和环境变量见 [docs/setup.md](docs/setup.md)）。agent 通过
[Codex CLI](https://github.com/openai/codex) 0.155.1 或 Claude Code 运行。

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git && cd PyRUA-Lean
pip install -e .          # 装进你的 RPent 机器人环境所用的 Python
pyrualean api             # 打印 agent 看到的机器人接口；不需要仿真器

# 跑一局 LIBERO-PRO：GPT-6 Astra 对着 robo 写代码，最多 40 次 LLM 调用
export RPENT_ROOT=/path/to/rpent
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --out runs/demo --cuda-device 0
pyrualean summarize runs/demo
```

运行目录里保存了这一局的视频、agent 运行过的每个代码单元及其输出、原语调用记录、token 用量，以及基准自己的成功判定。

## 支持的机器人

| `--robot` | 基准 | 机器人与技能 |
|---|---|---|
| `libero`（默认） | LIBERO-PRO | Franka 单臂；脚本化伺服、Pi0.5 VLA、SAM3 感知 |
| `robotwin` | RoboTwin 2.0 | 双臂；LingBot-VLA |
| `robocasa` | RoboCasa365 | 移动操作机器人；RLDX-1 VLA、导航 |

新增机器人的约定见 [docs/multi-robot.md](docs/multi-robot.md)。

## 工作方式

- **同样的原语，同样的判分。** `robo` 的每个方法一一对应 RPent 的一个工具，并经由 RPent 执行，所以一局留下的记录和
  RPent 的一局相同，成功与否由基准自己的判定决定。
- **返回结果，而不是抛异常。** 运动调用会阻塞，并返回实际发生了什么（`Move.reached`、`Pick.success` 等）；
  误用会报错，任务完成后所有后续运动都会被拦下。
- **不会过时的 prompt。** agent 读到的接口说明直接从运行中的类生成。
- **三种运行方式。** `play --arm cells`（每次决策一个代码单元）、`play --arm program`（整段程序），以及
  `generate` + `run` 的单次生成。
- **沙箱执行。** agent 的代码只能导入固定的一组模块，文件访问限制在它的工作目录内，拿不到后端的句柄。

## 文档

- [docs/guide.zh-CN.md](docs/guide.zh-CN.md)：`robo` 接口、prompt、单次与交互式运行、预算、评测与配置。
- [docs/setup.md](docs/setup.md)：搭建三个机器人环境和 agent 运行时（英文）。

## 许可与致谢

Apache License 2.0（[LICENSE](LICENSE)）。PyRUA-Lean 构建在 [RPent](https://github.com/RLinf/RPent)（Apache-2.0）
之上：`robo` 背后的机器人环境、服务和原语实现都来自 RPent，工具调用的基线就是 RPent 本身，本仓库的部分代码也改编自它，
详见 `NOTICE`。如果你用到了 RPent，请引用它的论文
[Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448)。
