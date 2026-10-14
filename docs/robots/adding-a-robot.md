# Adding a Robot

Robots are added through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab). The benchmark's own evaluation
client keeps the simulator, the observation and the verdict; PyRUA-Lean answers it as a policy, through the
adapter that every benchmark XPolicyLab integrates already talks to. [RoboDojo](robodojo.md) is the reference
implementation.

## How an episode runs

1. The client sends an observation (`update_obs`). On the first one of a trial, the adapter starts a PyRUA-Lean
   episode: the agent loop, the sandboxed cells and the robot `robo`.
2. A cell calls a primitive; the primitive turns what it was asked into native actions, computed from the
   observation.
3. The client asks for an action (`get_action`) and gets the next one. While the model is still thinking, the
   answer is "wait" and the adapter asks again, so the client's request never times out.
4. The client steps the simulator and sends the observation that followed, which the primitive reads to report
   what happened.
5. When the episode is over (the task solved, the action budget spent, or the program finished), the client
   reports the verdict (`trial_end`).

`pyrualean.hosts.xpolicylab_bridge` turns XPolicyLab's two calls into an environment a primitive can step; it is
the same for every robot.

## What a robot needs

| Piece | File | RoboDojo |
|---|---|---|
| Robot class: the API the agent writes against, one method per primitive with typed results; its docstrings are the API reference in the prompt | `src/pyrualean/<robot>.py` | `robodojo.py` (`RobodojoRobot`), specialised by `robodojo_xpl_rgb.py` |
| Primitives: motion, gripper and perception of this library's own, computed from the observation | `src/pyrualean/hosts/<robot>_tools.py` and helpers | `robodojo_tools.py`, `robodojo_kinematics.py`, `robodojo_sight.py` |
| Toolkit: the primitives over an environment that has only an observation and an action, with each step's images and state recorded | `src/pyrualean/hosts/<robot>_native.py` | `robodojo_native.py` |
| Host: builds the toolkit over the bridge and writes the task card; for `pyrualean play` on one machine it also starts the benchmark's client | `src/pyrualean/hosts/xpolicylab_<benchmark>.py` | `xpolicylab_robodojo.py`, `xpolicylab_robodojo_rgb.py` |
| Knowledge: one text for every task | `src/pyrualean/knowledge/<robot>.md`, `<robot>-novla.md` | `robodojo_xpl_rgb.md`, `robodojo_xpl_rgb-novla.md` |
| Adapter: XPolicyLab's policy directory | `integrations/xpolicylab/PyRUA_Lean/` | the same directory |

Shared, and not changed by a new robot: the bridge, the agent runtimes ([Agent Configuration](../guide/agents.md)),
the cells and their sandbox, and the prompt with its API reference generated from the robot class.

## The robot class

- Every public method and property is rendered into the API reference from its signature and docstring, so write
  them for the model; everything host-facing starts with `_`.
- Results, not exceptions: what a primitive measured (`reached`, `found`, how wide the gripper is) is a field of a
  frozen result dataclass. `ValueError` for bad arguments, `ToolError` when a tool fails, `EpisodeFinished` once the
  episode is over.
- A member the benchmark cannot support goes into `ABSENT_MEMBERS` instead of answering with a constant: RoboDojo
  judges only the final scene, so its robot has no `done`.
- Only what a program cannot compose from other primitives is a primitive; the knowledge text says how to compose
  the rest (on RoboDojo a twist is a `move_to` with another `roll`, a button press a `move_to` past the surface).
- Nothing privileged: the primitives read the observation the client sends, and constants of the embodiment and
  the default scene taken from the benchmark's public configuration.
- The class subclasses `RobotBase` (`src/pyrualean/_robot.py`): it declares `CAMERAS`, `IMAGE_ARTIFACTS`,
  `ON_MOTION_IMAGES`, `STATEFUL_TOOLS`, `RESULT_TYPES` and `VLA_PRIMITIVES`, implements `_state_from_envelope` and
  `_summary_dict`, and sends every primitive through `_readonly` or `_stateful`.
- The module ends with its registration, `ADAPTER = RobotAdapter(name=..., robot_cls=..., host=..., knowledge=...,
  blurb=...)`, with the XPolicyLab host as `host`, and the robot's name goes into `ROBOTS` in
  `src/pyrualean/robots.py`: `get_adapter` refuses any other name.

## The adapter

`integrations/xpolicylab/PyRUA_Lean/` is one directory for every benchmark, as XPolicyLab keeps one per policy.

| File | Role |
|---|---|
| `deploy.yml` | The robot, the primitive set, the runtime, the model, the budgets, when the model sees images and what a tool call returns. |
| `model.py` | XPolicyLab's model: starts one PyRUA-Lean episode per trial on the first observation and serves it through the bridge. It knows RoboDojo today; a new benchmark adds its robot and host here, chosen by `bench_name`. |
| `deploy.py` | XPolicyLab's evaluation loop with three changes for a policy that thinks between actions: ask again on "wait", end the episode on an empty answer, and pass the step count and limit with every observation. It reports the verdict back through `trial_end`. |
| `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh` | XPolicyLab's entry scripts. |
| `install.sh`, `download_assets.sh` | The policy environment and SAM3's checkpoint. |
| `README.md` | XPolicyLab's policy README: installation, evaluation, configuration and notes. |

## The knowledge text

`knowledge/<robot>.md` (and `<robot>-novla.md` without the VLA primitives) is the same for every task: frames and
units, how each primitive behaves, how to judge progress from the images, and the benchmark's success rules that
an instruction does not spell out (RoboDojo, for most tasks: both arms back at their start pose, the grippers open). No task
recipes.

## Checking a new robot

1. Unit tests against a scripted environment, for every primitive and the API reference (`tests/test_robodojo*.py`).
2. XPolicyLab's debug client, which exercises the loop, the action keys and their dimensions without a simulator
   or a model: `EVAL_ENV_TYPE=debug bash eval.sh <bench_name> <task> ...`.
3. One episode with `pyrualean play --robot <robot>`, which starts the benchmark's client itself (for RoboDojo it needs
   `ROBODOJO_SOURCE_ROOT` and one environment with both sides).
4. A few episodes with a model through the benchmark's official entry (RoboDojo: `scripts/robodojo.sh eval`), then
   the full task set.
