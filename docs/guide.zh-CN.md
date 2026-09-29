# PyRUA-Lean 使用指南

完整参考：支持的机器人、`robo` 接口、prompt、单次生成与交互式运行、预算、零样本隔离、评测与配置。简短介绍见 [README](../README.zh-CN.md)。

一个 policy 就是一个普通的 Python 文件：

```python
def run(robo):
    bowl = robo.segment("the black bowl on the stove")
    if not bowl.found:
        return
    x, y, _ = bowl.world_xyz
    robo.move_to([x, y + 0.045, 0.60])                    # 悬停在碗沿上方
    pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
    if robo.state().gripper_opening < 0.01:                # 夹空了
        pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
    robo.set_gripper(robo.CLOSE, steps=8)
    plate = robo.segment("the white plate")
    robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.60], gripper=robo.CLOSE)
    robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.47], gripper=robo.CLOSE, step_clip=0.012)
    robo.release()
```

宿主构造 `robo`，调用一次 `run(robo)`，成功与否由仿真器自己的任务谓词判定。程序执行期间没有任何语言模型参与。

`robo` 背后的原语就是 **RPent 的 LIBERO 原语，一个没改**：脚本化 OSC 伺服（`move_to`、`move_pose`、`rotate_*`、`set_gripper`、`release`）、冻结的 Pi0.5 VLA 技能（`pi0_pick`、`pi0_doubled`）以及 SAM3 / 反投影感知。每次调用仍然走 RPent 的 `Toolkit.execute_tool`，所以一回合 PyRUA-Lean 留下的 `states.json`、`episode.mp4`、逐步图片和 RPent agent 回合完全一样，判分也是同一个 `toolkit.solved()`。这正是这个项目的目的：**同一套原语、两种交互形态**，

| 臂 | 谁决定每一步 | 交互方式 | 在哪里 |
|---|---|---|---|
| 逐步工具调用（CUA 式） | 模型，每轮一个 MCP/tool call，每轮拿回观测 | RPent planner（`rpent --planner codex ...`） | [RPent](https://github.com/RLinf/RPent) |
| 代码（本包） | 模型，对着 `robo` 写 Python：每次决策一个 cell，或者一次写完整个程序 | `pyrualean play --arm cells`，或 `pyrualean generate` 再 `pyrualean run` | 本仓库 |

在相同的 `(suite, task, seed)` 格子上、用相同的成功谓词比较。

## 机器人

`--robot libero`（默认）、`--robot robocasa`、`--robot robotwin` 选择机器人：它面向 policy 的类（`pyrualean/<robot>.py`）、为它启动 RPent 服务栈的宿主（`pyrualean/hosts/rpent_<robot>.py`）和它的知识文件。三者共用 `pyrualean/_robot.py`（调用账本、回合守卫、图像、世界坐标图、`show`），通过 `pyrualean.robots` 注册；新增机器人的约定见 `docs/multi-robot.md`。下文以 LIBERO 为例，另外两个机器人结构相同、原语不同（RoboCasa365 是移动底盘加 RLDX-1 VLA，RoboTwin 是双臂加 LingBot-VLA）。

## 库

policy 唯一能看到的 API 是 `pyrualean.libero.LiberoRobot`。每个方法 1:1 对应一个 RPent 工具，参数名和默认值与 RPent 的 schema 一致，返回值是字段有文档的冻结 dataclass（这些文档就是 prompt，见下）。

| `robo.` | RPent 工具 | 返回 |
|---|---|---|
| `state(step=-1)` | `view_env_state` | `State`（末端位姿、夹爪开度、物体名、回合标志） |
| `image(camera, resolution)` / `world_map(...)` / `artifact(name)` | 记录的产物 | numpy 数组 / bytes |
| `camera_meta(camera)` | `view_camera_meta` | dict |
| `segment(prompt \| point=...)` | `segment` | `Segment`（`found`、`world_xyz`、`score`） |
| `back_project(row, col)` | `back_project` | `BackProjection` |
| `region_center(row_range, col_range)` | `back_project`（区域模式） | `RegionCenter` |
| `move_to(xyz, gripper=-1, ...)` | `move_to` | `Move`（`reached`、`final_dist_m` …） |
| `move_pose(xyz, target_pitch=..., ...)` | `move_pose` | `MovePose` |
| `rotate_wrist(target_yaw= / delta_yaw=)` | `rotate_wrist` | `Rotation` |
| `rotate_pitch(target_pitch= / delta_pitch=)` | `rotate_pitch` | `Rotation` |
| `set_gripper(gripper, steps=5)` | `set_gripper` | `Gripper` |
| `release(max_steps=20)` | `release` | `Release`（`terminated`） |
| `pi0_pick(prompt, max_chunks=24, ...)` | `pi0_pick` | `Pick`（`success`、`peak_lift_m` …） |
| `pi0_doubled(prompt, max_chunks=20)` | `pi0_doubled` | `Contact` |

policy 可以依赖的库规则：

- 动作调用阻塞到原语结束，返回真实发生的结果；`Move.reached` 即 `final_dist_m <= tol`，`Pick.success` 是 RPent 的抬升启发式。
- 用错了会抛异常：参数非法抛 `ValueError`（包括平面位移超过 0.30 m，RPent 禁止这样做因为会翻转 OSC 的 IK），后端拒绝或服务失败抛 `ToolError`。测量出来的结果（没找到 mask、伺服停滞）是返回值，不是异常。
- 任务谓词一旦触发，`robo.done` 为真，之后任何动作调用抛 `EpisodeFinished`；它是 `BaseException`，`except Exception` 吞不掉。宿主的超时看门狗走同一机制。
- 每次调用都记账（`calls.jsonl`：工具、参数、墙钟、精确仿真步数、结果摘要）。

`src/pyrualean/libero.py` 里的 docstring 和 `RESULT_TYPES` 是唯一真相；`pyrualean api` 打印生成的参考文档。

## Prompt

`pyrualean prompt` 输出交给模型的全部内容：

1. 契约（写一个定义 `run(robo)` 的 `policy.py`；怎么跑；怎么判分；输出格式）；
2. API 参考，用 `inspect` 从活的类生成，不可能和注入的对象脱节；
3. `knowledge/libero.md`：RPent system prompt 给它的 agent 的操作经验（夹爪符号、0.30 m 位移上限、按物体取 `step_clip`、VLA 用法、SAM3 措辞、agentview 定身份 / wrist 定几何、放置）；
4. 卷子（`card.json`）：suite/task/seed、任务语句、物体名、初始夹爪位姿、第 0 步的 `agentview`/`wrist` 图。

卷子上没有任何特权信息：这就是 RPent agent 第一次 `view_env_state` 看到的东西。

## 跑一个格子

需要：装了 `libero-pro` extra 的 RPent checkout、LIBERO-PRO 资产、Pi0.5 与 SAM3 权重及对应环境变量（`LIBERO_PRO_ASSET_PATH`、`PI05_CHECKPOINT_PATH`、`SAM3_CHECKPOINT_PATH`、`LIBERO_TYPE=pro`、`MUJOCO_GL=egl`），即 `rpent --robot libero` 需要的一切（三个机器人的完整安装见 `docs/setup.md`）。下面的命令用 RPent 的解释器执行，`PYTHONPATH` 指向 `src/`（或在那个环境里 `pip install -e .`）。RPent 没有以包形式安装时用 `RPENT_ROOT` 指向 checkout。

```bash
export RPENT_ROOT=/path/to/rpent
export PYTHONPATH=/path/to/pyrualean/src

# 1. 卷子（只起仿真器，约 1 分钟）
python -m pyrualean card --suite libero_object_swap --task 2 --seed 0 \
    --out runs/card-object_swap_t2_s0 --cuda-device 0

# 2. 用 Codex CLI 单次生成（只读沙箱、空工作目录、保留全部 Codex 事件以审计生成期间有没有用工具）
python -m pyrualean generate --card runs/card-object_swap_t2_s0/card.json \
    --out runs/gen-object_swap_t2_s0 --model gpt-6-astra --reasoning xhigh

# 3. 跑生成的文件：每回合一个子进程、硬超时，崩溃 / 超时 / 起不来都记为失败回合
python -m pyrualean run --policy runs/gen-object_swap_t2_s0/policy.py \
    --suite libero_object_swap --task 2 --seed 0 \
    --out runs/code-object_swap_t2_s0 --cuda-device 0 --timeout-s 1200

# 4. 列表 / 与 RPent 回合配对
python -m pyrualean summarize runs/code-object_swap_t2_s0 /path/to/rpent/logs/<run>
python -m pyrualean compare runs/code-object_swap_t2_s0 rpent-episodes.json --output paired.json
```

`generate --transfer` 把卷子措辞切成迁移模式：告诉模型这个程序会在同一任务的其他 seed 上跑、不能硬编码像素。不加就是逐格单次（附图就是本次运行的第 0 步）。

一个 run 目录里有：`policy.py`（副本，sha256 记在 `result.json`）、`prompt.txt`、`card.json` + `card/*.png`、`calls.jsonl`、`policy_output.txt`（程序的 stdout/stderr）、`result.json`，以及 RPent 自己的产物（`states.json`、`episode.mp4`、逐步图片、服务日志）。held-out 评测就是在没生成过的 seed 上重复第 3-4 步。

`examples/replay_object_swap_t2_s0.py` 是链路检查：从 RPent 在该 seed 上一次成功回合转写的命令序列，只证明库能把同一批原语驱动到同一结果，别的什么都不证明。

## 交互式臂：`pyrualean play`

`generate` + `run` 是严格的单次：一条回复、一个程序、零反馈。`play` 用同一个库、同一个判分跑交互式协议：代理循环由 agent 运行时负责（Codex CLI，或 `--runtime claude` 时经 Claude Agent SDK 的 Claude Code），PyRUA-Lean 通过一个进程内 MCP server（改编自 RPent 的实现，需要 RPent 环境里自带的 `mcp` 1.x、`uvicorn`、`httpx`）把臂的工具挂上去，并审计事件流。

| 臂 | 给 agent 的工具 | 一次调用执行什么 | 调用之间保留什么 |
|---|---|---|---|
| `cells` | `python(code)` | 持久命名空间里的一个 cell，预置 `robo`、`np`、`math`、`time`（OpenAI python tool 式） | 变量、函数、import、工作区文件 |
| `program` | `run_program(code \| path)` | 新命名空间里跑一个完整的 `run(robo)` 程序 | 工作区文件（可 import）、agent 自己的上下文 |

两臂都有 `finish(status, summary)`；`--max-programs N` 限制程序执行次数（`1` 即单次，执行后能看到结果）。每次工具结果都带 stdout/stderr、异常、原语调用账本、机器人状态。`--images` 决定何时附相机图：`on-motion`（默认）在每个动过机器人的轮次后附两张当前图，和工具调用式 agent 每个原语之后看到的一致；`on-demand` 只在代码调用了 `robo.show("agentview" | "wrist")` 时附图，即由程序决定作者什么时候需要看；`none` 从不附图。`--feedback pure` 只返回代码打印出来的内容（python tool 语义），`rich`（默认）再加上机器人状态和账本。

两种预算。默认只卡总墙钟（`--budget-s`，含思考）和仿真步数，轮数只记录不限制。加 `--max-turns N` 则改为**决策次数预算**：最多 N 次 `python` / `run_program` / `finish` 调用（只看不动的调用也算），也就是 N 次模型回复，因为 agent 每次回复只调用一次工具。此时 `--budget-s` 只是安全兜底，契约只告诉 agent"调用次数有限"而不给数字（RPent 的 prompt 也不给）。决策预算让结果不依赖提供方延迟，墙钟预算做不到这一点。

`--guides` 把 RPent 的三份 LIBERO 操作指南（它的工具调用 agent 每格开头都会读的文件）复制进工作区，并在契约里加一句"在那里、可选阅读"；`--guides-dir <dir>` 则改为复制另一个目录里的全部 `*.md` 文件，`result.json` 记下 agent 拿到的这些文件的哈希。`--vla-endpoint host:port` / `--sam3-endpoint host:port` 让回合挂到共享的 Pi0.5 / SAM3 服务器而不是各起一套；仿真器永远是回合私有的。

```bash
python -m pyrualean play --arm cells   --suite libero_object_swap --task 2 --seed 0 \
    --out runs/cells-object_swap_t2_s0 --cuda-device 0 --budget-s 1200 \
    --model gpt-6-astra --base-url https://gateway.example/v1 --api-key-env MY_KEY
python -m pyrualean play --arm program --max-programs 1 ...   # 单次
python -m pyrualean play --arm program --max-programs 5 ...   # 多轮
```

零样本运行的隔离：`--codex-home` 指向一个只含 `config.toml` 和凭据的 Codex home（`memories = false`、`approval_policy = "never"`；对比实验用的设置见 `docs/setup.md`），`--codex-bin` 指向真正的 Codex 二进制而不是会强制改写 `CODEX_HOME` 的包装脚本。`play` 只给 agent 最小环境，并关闭 Codex 的 shell snapshot 和 memories，沙箱 shell 因此找不到别的 Codex home 或历史会话。

**Codex 的 code mode。**
- Codex CLI 0.155.1 自带的模型目录把 `gpt-6-astra` 标成 `code_mode_only`：所有工具都被藏到一个 JavaScript `exec` 工具后面。对比实验用的网关 id（`openai/openai/gpt-6-astra`）Codex 不认识，所以跑在 Codex 的回退元数据上：工具是直接的 function call，prompt 是 Codex 的通用 prompt。
- 模型是 `gpt-6-astra` 时，`play` 和 `generate` 传入 `-c model_catalog_json=<路径>`，指向包里的 `codex-catalog-gpt-6-astra-fallback.json`：它就是 Codex 0.155.1 自带的模型目录，只把 `gpt-6-astra` 一项改成回退元数据的取值，所以 Codex 的系统 prompt、内置工具和各项设置与对比实验逐字节相同。（只含一个条目的目录也能得到直接工具调用，但只要设置了目录，Codex 就会保留 prompt 里的 Planning 各节。）用 `gpt-6-astra` 时，RPent 用的 Codex home 也要在 `config.toml` 里写同一行 `model_catalog_json = "<绝对路径>"`。
- 检查一次运行：会话 rollout（`$CODEX_HOME/sessions/**/rollout-*.jsonl`）里的第一个工具调用是 `function_call`，而不是名为 `exec` 的 `custom_tool_call`，base instructions 以 "You are a coding agent running in the Codex CLI" 开头。
- 固定使用 Codex 0.155.1（`npm install -g @openai/codex@0.155.1`）。

代码臂上的 policy 代码跑在沙箱里（`pyrualean/sandbox.py`）：只能 import 纯计算模块和工作区文件，`open` 限于工作区，去掉 `exec`/`eval`/`compile`，`robo` 对象不带任何后端句柄，提到 Python 内省途径（`__globals__`、`__func__`、`__subclasses__` …）的代码运行前就被拒绝，每个 cell 或程序都做静态审计；任何伸向宿主句柄、私有属性、内省或工作区外路径的代码都记在 `result.json`（`generation.sandbox_flags`）和 `tool_turns.json` 里。

run 目录多出 `workspace/`（agent 跑过的每个 cell / 程序和它自己写的文件）、`codex_events.jsonl`、`tool_turns.json`，`result.json` 多了 `arm`、`finish`、token 用量和 shell 命令审计。对照的工具调用臂就是 RPent 本身，原样运行（`rpent --robot libero --planner codex ...`），用 `parse_rpent_run` 解析。

## 评测

`pyrualean.evaluation` 提供回合记录（`EpisodeMetrics`）、RPent transcript 的只读解析（`parse_rpent_run`）和严格配对比较（`compare`，`(backend, task, seed)` 缺配或重复即判无效）。代码臂的 `environment_success` 永远是布尔值，崩溃、超时、起不来都是失败回合。`env_steps` 两边口径一致（代码臂是精确帧计数，RPent transcript 从 `states.json` 重建）。

## 搭建机器人环境

[`docs/setup.md`](setup.md) 写明三套 RPent 服务栈（版本、权重、环境变量）和 agent 一侧的准备（Codex CLI、Codex home、端点、Claude 运行时）。

## 配置

库从环境变量读取的设置都叫 `PYRUALEAN_<NAME>`。

| 变量 | 谁读 | 含义（默认值） |
|---|---|---|
| `PYRUALEAN_CODEX_BIN`、`PYRUALEAN_CODEX_HOME` | `play` | Codex 可执行文件（`codex`）和干净的 Codex home（无） |
| `PYRUALEAN_RUNTIME`、`PYRUALEAN_CLAUDE_CLI` | `play` | agent 运行时 `codex` / `claude`（`codex`），`claude` 可执行文件（`claude`） |
| `PYRUALEAN_PRIMITIVES` | `play` | 原语集 `full` / `no-vla`（`full`） |
| `PYRUALEAN_CODEX_RETRIES` | `play`、`generate` | `--base-url` 提供方的请求 / 流重试次数（`10`） |

`RPENT_ROOT` / `RPENT_REPO_ROOT` 指向 RPent checkout；各机器人自己的环境变量见 `docs/setup.md`。

## 开发

```bash
pip install -e '.[dev]'
pytest          # 离线用例，跑在仿 RPent 形状的脚本后端上
ruff check .
```
