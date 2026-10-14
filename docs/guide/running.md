# Running Episodes

`pyrualean play` runs one episode: it boots the robot, gives the agent the task and serves its code until the
episode is over.

```bash
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
```

`--robot` picks the robot (`libero`, the default, `robotwin` or `robocasa`) and `--suite`, `--task`, `--seed` the
task instance; each [robot page](../robots/index.md) says what they mean there. Always pass `--seed`: `play` sends
`0` when it is left out, whatever the robot's own default. RoboDojo runs through XPolicyLab
([RoboDojo](../robots/robodojo.md)).

## Cells or programs

| `--arm` | What one LLM call runs | What carries over |
|---|---|---|
| `cells` | a cell in a persistent namespace with `robo`, like a notebook | variables, functions, imports, files |
| `program` | a complete program, `def run(robo): ...`, in a fresh namespace | files in the workspace, and the conversation |

`--max-programs N` caps the number of programs on the `program` arm; `1` is a single shot whose result the agent
still sees.

## What the agent sees after a call

| Option | Values |
|---|---|
| `--images` | `on-motion` (default): the current camera images after every call that moved the robot. `on-demand`: an image only when the code asked for one with `robo.show(camera)`. `none`. |
| `--feedback` | `rich` (default): what the code printed, plus the robot's state and the primitives it called. `pure`: only what the code printed. |

## Budgets

| Option | Meaning |
|---|---|
| `--max-turns N` | A decision budget: at most N code cells (`python` or `run_program` calls; `finish` is not counted). The agent knows the cells are limited, not how many. |
| `--budget-s S` | A wall-clock budget in seconds, thinking included (default `1200`). With `--max-turns` it is a safety ceiling: set it well above what the cells take, so that outcomes do not depend on how fast the provider is (the paper's runs used `7200`). |

## VLA policies, guides and shared servers

- `--primitives no-vla` removes the robot's VLA primitives: they disappear from the API reference and the
  knowledge text, and calling one is refused.
- `--guides` gives the agent RPent's three LIBERO operating guides in its workspace (optional reading);
  `--guides-dir <dir>` gives it the `*.md` files of another directory.
- `--vla-endpoint host:port` (and `--sam3-endpoint` on LIBERO-PRO) attach the episode to running model servers
  instead of booting its own; the simulator is always the episode's own.

## A single shot

The strictest setting: one reply, one program, no feedback.

```bash
pyrualean card --suite libero_object_swap --task 2 --seed 0 --out runs/card --cuda-device 0
pyrualean generate --card runs/card/card.json --out runs/gen --model gpt-6-astra
pyrualean run --policy runs/gen/policy.py --suite libero_object_swap --task 2 --seed 0 \
    --out runs/code --cuda-device 0
```

`card` boots only the simulator and saves the task card; `generate` asks the model for `policy.py`
(`--transfer` tells it the program will run on other seeds, so it must not rely on pixels); `run` runs the file
once, with a hard timeout, and a crash or timeout counts as a failed episode.
