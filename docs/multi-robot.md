# Adding a robot to PyRUA-Lean

PyRUA-Lean drives any RPent robot through three pieces, all selected by the
robot's name (`--robot libero|robocasa|robotwin`):

| piece | file | what it is |
|---|---|---|
| robot class | `src/pyrualean/<robot>.py` | the policy-facing API: one method per RPent tool, typed results, docstrings that *are* the prompt |
| host | `src/pyrualean/hosts/rpent_<robot>.py` | boots RPent's servers for one cell, builds the toolkit, dumps the task card |
| knowledge | `src/pyrualean/knowledge/<robot>.md` | the operating knowledge RPent's tool-calling agent gets, rewritten for a program author, with no memory / recipe / audit content |

`src/pyrualean/libero.py` and `src/pyrualean/hosts/rpent_libero.py` are the
reference implementation; `src/pyrualean/_robot.py` (`RobotBase`) and
`src/pyrualean/hosts/common.py` hold what every robot shares.  The
tool-calling arm of the comparison is RPent itself, run unchanged.

## 1. The robot class

```python
from pyrualean._robot import RobotBase
from pyrualean.robots import RobotAdapter

class RobocasaRobot(RobotBase):
    """<model-facing description of the robot, frame, units, gripper convention>"""

    NAME = "robocasa"
    CAMERAS = ("agentview", "navview", "wrist")            # names robo.show()/image() accept
    IMAGE_ARTIFACTS = {("agentview", "high"): "...", ...}  # (camera, resolution) -> artifact
    WORLD_MAP_ARTIFACTS = {("agentview", "high"): "...npz", ...}
    ON_MOTION_IMAGES = ("...png", "...png")                # attached after a moving turn
    STATEFUL_TOOLS = frozenset({"move_to", "navigate_to", ...})   # tools that step the sim
    RESULT_TYPES = (State, Move, ...)                      # rendered into the API reference
```

Rules (they keep the LIBERO prompt byte-identical and make the reference
generator work):

- Every public method or property is rendered into the API reference from
  its signature and docstring; everything host-facing starts with `_`.
  The base class already provides `task`, `done`, `state`, `image`,
  `world_map`, `show`, `artifact`; override them only to give them
  robot-specific docstrings (call `super()`), as `libero.py` does.
- Implement `_state_from_envelope(envelope) -> State` (RPent's
  `view_env_state` payload -> your frozen `State` dataclass; it must expose
  `.task`) and `_summary_dict()` (a small dict shown in the tool feedback:
  step, pose, gripper, success flags).
- Read-only tools go through `self._readonly(tool, kwargs)`, stateful
  tools through `self._stateful(tool, kwargs, build)` where `build` maps
  the primitive's payload (`log.result` of the envelope) to the result
  dataclass.  Check in the RPent toolkit how the envelope of *this* robot
  looks: `_backend.unwrap()` treats a payload as an envelope when it has
  both `log` and `state`; if the robot's envelope differs, adapt the
  payload in `build`.
- Measured outcomes are results (`reached`, `success`, `found`), never
  exceptions; `ToolError` for backend errors, `ValueError` for bad
  arguments, `EpisodeFinished` once the episode is over (the base raises
  it from `_guard`).  Give your `State` `terminated` / `truncated`
  attributes and set them in `_state_from_envelope` from RPent's own flags
  (`success`, `eval_success`, a step budget such as
  `take_action_cnt >= step_lim`): the base `_absorb` reads them, so no
  override is needed.
- Keep argument names and defaults identical to RPent's tool schema
  (`TOOLS_SPEC`); the docstring documents every argument and result field
  the way `libero.py` does.  Constants the policy may need (`OPEN`,
  `CLOSE`, arm names) are uppercase class attributes; `RESULT_TYPES` (the
  frozen result dataclasses of the reference) and `ClassVar`-annotated
  configuration are not rendered as constants.
- `world_map` reads `.npz` archives and bare `.npy` arrays; override it only
  for a robot-specific docstring.
- At the end of the module:

```python
ADAPTER = RobotAdapter(
    name="robocasa",
    robot_cls=RobocasaRobot,
    host="pyrualean.hosts.rpent_robocasa",
    knowledge="robocasa",
    blurb="a mobile-base PandaOmron robot in the RoboCasa365 kitchen benchmark",
    guides_subdir="",                 # RPent guides dir relative to its checkout, "" if none
    guide_names=(),
    show_examples="`robo.show('agentview')`, `robo.show('navview')` or `robo.show('wrist')`",
    services="a frozen RLDX-1 VLA",   # phrase after "bound to the live simulator, "
)
```

## 2. The host

Module-level functions `pyrualean.play` calls (see `hosts/rpent_libero.py`):

```python
@dataclass(frozen=True)
class Cell:            # must have .suite, .task, .seed, .max_episode_steps, .cuda_device, .tag
    ...

def add_cell_args(parser): ...   # --task (name), --suite (split / task config), --seed, ...
def add_boot_args(parser): ...   # shared servers, model paths (env vars as defaults)
def cell_from_args(args) -> Cell: ...
def boot_options(args) -> dict: ...          # keyword arguments of boot()
def boot(cell, out_dir, *, rpent_root=None, env_only=False, **options) -> (toolkit, daemons)
def dump_card(toolkit, cell, out_dir) -> TaskCard   # card.json + card/*.png
def make_backend(toolkit) -> RpentBackend           # optional: RPent's toolkit lacks solved() /
                                                    # primitives, or counts steps in another unit
def main(argv=None) -> int                          # `card` and `run` subcommands
```

`boot` mirrors `rpent_libero.boot`: `resolve_rpent_root`, `init_output_dir`,
build the `argparse.Namespace` the robot's `_init_runtime` / `_parse_config`
expect (read `robots/<robot>/robot_spec.py`), `spec.init_runtime(args,
out_dir, NullDashboardEventSink(), components)`, then the robot's
`get_toolkit(...)`.  Memory: pass an empty corpus directory (create
`runs/_empty-memory-<robot>/MEMORY.md`) so no HF sync happens and nothing is
read.  `dump_card` fills `TaskCard(robot=<name>, suite=<variant>,
task=<task name>, seed, task_language, object_names=[...], facts={...},
images={camera: path})` (the single-arm `eef_pos` / `eef_quat` /
`gripper_opening` fields stay `None`); `facts` are rendered as bullets
(initial base / arm poses, gripper opening, table height, whatever the
tool-calling agent also sees in its first `view_env_state`).  Nothing
privileged (no object poses).

## 3. Knowledge, tests, smoke

- `knowledge/<robot>.md`: distil `robots/<robot>/prompts/system.py` and
  the robot's guide the way `knowledge/libero.md` distils LIBERO's prompt:
  frames and units, gripper conventions, motion limits, the VLA rules,
  perception tools, placing / recovery.  No memory, recipe, audit, launch
  or server text; no per-task recipes.
- Tests: a scripted fake backend in the RPent envelope shape (see
  `tests/conftest.py`) and tests in the style of `tests/test_libero.py`
  covering every method, `_summary_dict`, the card and the API reference
  (`api_reference(RobocasaRobot)` lists every public method).
- Smoke: `python -m pyrualean card --robot <robot> ...` boots only the
  simulator; a replay script like `examples/replay_object_swap_t2_s0.py`
  drives the primitives on the real simulator through `pyrualean run`;
  then one `pyrualean play --robot <robot> --arm cells` episode.


## 4. The no-VLA primitive set

`--primitives {full,no-vla}` (`play`, `api`, `prompt`; environment default
`PYRUALEAN_PRIMITIVES` for `play`) picks the primitive set; the default leaves
every prompt byte-for-byte as it is without the switch.  `no-vla` removes the
robot's learned VLA primitives (`VLA_PRIMITIVES` on the robot class:
`pi0_pick`/`pi0_doubled`, `lingbot_act`, `rldx_skill`/`rldx_arm`) from the
code arm's API reference, knowledge (`knowledge/<robot>-novla.md`) and
contract, and refuses them at call time with
`"<name> is not available in this run (no-VLA primitive set)"`
(`result.json` records `host.primitive_set`).  The servers still boot as
usual.

## 5. Agent runtimes

`pyrualean play --runtime codex|claude` (env `PYRUALEAN_RUNTIME`) picks who
runs the agent loop; everything else (boot, prompt, the HTTP MCP server over
`ArmToolkit`, budgets, `result.json`) is shared.  `codex` launches `codex exec`
in a scrubbed subprocess.  `claude` (`src/pyrualean/claude_runtime.py`) drives
Claude Code through `claude_agent_sdk.query` with `--claude-cli` (env
`PYRUALEAN_CLAUDE_CLI`; Opus 5.5 needs CLI >= 2.1.280, newer than the SDK's
bundled one): the prompt is the first user message with the card images as
base64 image blocks, Claude Code keeps its default system prompt (as RPent's
Claude planner does), tools are the read-only built-ins `Read Glob Grep` plus
`mcp__pyrualean__python|run_program` and `mcp__pyrualean__finish`, all
pre-approved through `allowed_tools` (no permission mode); `--reasoning none`
disables thinking, any other level is passed as `effort`; the SDK's
`max_turns` is `--max-turns + 5` (one turn per model request, `Read` calls
included) or 10000 without a decision budget; `--budget-s` cancels the query.
The agent process inherits the launcher's environment plus what Claude Code
needs to reach the model (`ANTHROPIC_*`, or a relay's `CLAUDE_RELAY_*`, its
proxy and `NODE_EXTRA_CA_CERTS`), so export those last in the launcher.  Outputs:
`claude_events.jsonl` (every SDK message, images reduced to sizes),
`claude_stderr.txt`, `last_message.md`, and `generation` with
`input_tokens` (= input + cache read + cache creation, Codex's convention),
`cached_input_tokens`, `output_tokens`, `reasoning_output_tokens` (run totals
from the result message; its `iterations` list is *not* per request),
`requests` and `request_profile` (cumulative prompt tokens per model request,
one per assistant message id), `tool_requests` (the
request that issued each MCP call), `request_latency_s` and `model_usage`
(per-model breakdown; Claude Code's own Haiku side calls appear there).
`claude_runtime.replay_events(path)` re-audits an events file offline.
