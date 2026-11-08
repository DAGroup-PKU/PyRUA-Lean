# Carry a skill library over to another robot benchmark

You are writing one skill library that a robot policy will be given on a **benchmark and robot it
was not learned on**. Your material is a library distilled from what improvers learned on ten tasks
of another benchmark: RoboDojo, an ARX-X5 two-arm robot, RGB cameras only, robot API
`robodojo_xpl_rgb`, policy GPT-6 Astra writing Python cells (library `{SOURCE}`). {SIDE_SENTENCE}
{TARGET_SENTENCE} Whether
experience carries over across robots and benchmarks is the question of the experiment, so write
only what you believe holds on the target robot, and state it in the target's API.

## Inputs (read-only): `inputs/`

- `source/library/lessons/*.md`, `source/library/skills/*.py`: the source library, exactly as the
  source policy was given it.
- `source/NOTES.md`: where each source card and skill came from (source tasks, evidence) and what
  was left out; `source/SOURCES.json`: each source task's successes per round on 5 dev layouts
  (first = empty library). Use them to judge how well supported a source rule is.
- `source/system_prompt.md`: everything the source policy read besides its library and its task
  card: the contract, how an episode is scored, the API reference of its `robo` object and the
  operating knowledge. Read it to understand what the source cards' API words mean.
- `source/api/*.py`: the source robot API code (`robodojo_xpl_rgb.py`, its base `robodojo.py`,
  `_robot.py`).
- `target/system_prompt.md`: everything the target policy will read besides your library and its
  task card, verbatim, rendered from the target code the way its evaluation builds it: contract
  (including the note that introduces the library), scoring, API reference, operating knowledge.
  Your lessons are appended right after its operating knowledge.
- `target/api/*.py`: the target robot API code (`{TARGET_MODULE}.py`, `_robot.py`); the API reference
  in `target/system_prompt.md` is generated from it.

Nothing tells you which target tasks will be run. Do not try to find out; the library must not be
tailored to them.

## What carries over

- Compare the two API references before writing anything. The APIs differ in names, arguments,
  cameras, frames, gripper conventions, how motion ends, and how and when success is checked.
- The target policy already reads `target/system_prompt.md`. Do not repeat what its operating
  knowledge already says; add only what the source experience adds to it or sharpens. Where a
  source rule contradicts the target's prompt or API, the target is right: adapt the rule or drop it.
- Statements about RoboDojo itself (what its judge checks, its cameras, its table, its two arms, its
  step and scoring rules) do not carry over as such. Behaviour that follows from physics,
  perception, control and budgets may, if the target's prompt and API give no reason to doubt it.

## Output: `out/`

The target policy receives this as its library: every `out/lessons/*.md` is appended **verbatim**
to its prompt (no filtering), and `out/skills/*.py` is importable as `library.<module>`. So:

1. **`out/lessons/*.md`: at most 12 000 characters in total.** Cards in the format the source uses
   (`## id`, `stage:`, `symptom:`, `applies:`, `do:`, `evidence:`). Each card:
   - is a general rule about perception, grasping, placing, transport, recovery, verification or
     using the step/call budget on the target robot; not about one task's objects or goal;
   - in `do:` uses only functions, arguments, result fields, constants and camera names that exist
     in `target/api/` (check them there), and names no task and no object of either benchmark
     (`applies:` says *when* it applies in general terms);
   - has `evidence:` naming the source card(s) it comes from and the source tasks behind them, and
     why it should hold on the target (e.g. what the target's API or prompt says that makes it apply).
   - Prefer few, sharp cards over many vague ones. Drop anything you cannot state precisely for the
     target.
2. **`out/skills/*.py`: optional, at most 2 modules, 15 000 characters in total.** Only task-agnostic
   helpers written for the target API: every `robo.<name>(...)` call must exist in
   `target/api/{TARGET_MODULE}.py` (or its base) with exactly those arguments; no source-API calls. Every
   function's docstring says when to use it. No task names, no hard-coded object sizes or positions.
   If you are not confident a helper is correct on the target, leave it out: a wrong skill is worse
   than none. Make sure it at least parses:
   `python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" FILE`.
3. **`out/NOTES.md`** (not given to the policy):
   - the source-to-target API mapping you relied on (what each source API word used in the source
     cards corresponds to on the target, or that it has no counterpart);
   - for every source card and skill: carried over as which target card or skill, adapted how, or
     dropped, with one line why;
   - for every target card and skill: which source cards and which target API or prompt facts it
     rests on.

There must be at least one file in `out/lessons/` or `out/skills/`.

## Rules

- Work only inside this directory (`{DIR}`); write only under `out/`. No subagents, no web.
- Check the character limits yourself before you finish (`wc -c out/lessons/*.md out/skills/*.py`).
