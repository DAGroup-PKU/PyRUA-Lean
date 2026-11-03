# Robots

| Robot | Benchmark | Runs on | VLA policy | Cameras the agent sees |
|---|---|---|---|---|
| `robodojo_xpl_rgb` | [RoboDojo](robodojo.md) | XPolicyLab, under RoboDojo's own evaluation client | none (SAM3 for segmentation) | three colour cameras |
| `libero` (default) | [LIBERO-PRO](libero.md) | RPent | Pi0.5 (and SAM3) | `agentview`, `wrist` |
| `robotwin` | [RoboTwin 2.0](robotwin.md) | RPent | LingBot-VLA | `head`, two wrists |
| `robocasa` | [RoboCasa365](robocasa.md) | RPent | RLDX-1 | `agentview`, `navview`, `wrist` |
| `robocoliseum` | [RoboColiseum](robocoliseum.md) | GenieSim, under its own benchmark (locally, or online through the platform's tunnel) | none (SAM3 for segmentation) | `head`, two wrists, with depth |

New robots are added through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
([Adding a Robot](adding-a-robot.md)): the benchmark's own client runs the simulator and judges the episode, and one
adapter directory can serve the benchmarks XPolicyLab integrates (today: RoboDojo). LIBERO-PRO, RoboTwin 2.0 and
RoboCasa365 run on [RPent](https://github.com/RLinf/RPent)'s stacks. RoboColiseum answers GenieSim's own inference
protocol, as the benchmark and the online platform call a policy.

Every robot page gives the API reference the agent reads, generated from the robot class when these docs are built.
