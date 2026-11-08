"""The worker: one headless Claude Code session turns a task's dev batch into one library change.

A session directory holds everything the worker may use::

    session.json      task, round, parent revision, campaign and config paths
    BRIEF.md          the instructions (also the prompt)
    episodes.md       this batch's episodes: digest, why each ended, run dir
    history.md        earlier rounds of this task: what changed, dev results, rollbacks
    candidate/        an editable copy of the parent revision; the worker edits tasks/<task>/
    FINDINGS.md       written by the worker
    replays/          replays the worker ran (``python -m pyrualean.learn.replay``)

With ``scoring`` off the worker is not shown how RoboDojo checks and scores the task: no
benchmark source, and only success from each episode.  Its evidence is then a filtered copy
inside the session (``evidence/``: run dirs without the benchmark tree or the verdict's
stages; ``sources/``: the package without the learning loops ``learn/`` and ``evolve/``), and
it may read nothing outside it.

After the session, :func:`accept` checks the candidate (static rules, task-local paths, card
IDs kept, size, ``dev_verified`` only with a judged replay) and freezes it.  The worker never
writes the library, runs the model policy, or sees a test layout: the campaign does the first,
the replay command refuses the others.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..evolve.library import LESSON_RE, SKILL_RE, Library, LibraryError, parse_lessons, validate
from .config import load, robocasa
from .episode import EXPANSION_HEADING

BRIEF = """# Worker brief: {task}, round {round}

You improve the skill library of ONE RoboDojo task, `{task}`, for an online code policy
(GPT-6-astra writing short Python cells against a frozen `robo` API, RGB perception only).
The library is the only thing that changes between rounds: skills (Python) the policy may
import and lesson cards appended to its prompt. Primitives, perception, the base knowledge,
the prompt contract and the benchmark are frozen.

Working directory: this session directory. Write only under `candidate/tasks/{task}/` and
`FINDINGS.md`, `notes/` and `replays/` here. Everything else is read-only evidence.

Tools: read files with Read, list and search with Glob and Grep (Read also shows the PNG
frames). Bash runs exactly three commands, from this directory: `ls`, `python -m
pyrualean.learn.replay ...` and `python -m pyrualean.learn.worker check`. Anything else in
Bash is refused: `cat`, `grep`, `jq`, `cd ... &&`, pipes into readers, `python -c`, other
python interpreters or scripts you write. Do not retry a refused command in another form; use
Read/Grep instead (Grep searches JSON files such as `calls.jsonl` and `messages.json`).

## Goal
Your change is judged by the next dev batch: the policy runs your library on the same
{n_dev} dev layouts. Aim at RoboDojo's own measures, in this order:
1. Success rate: more layouts where the benchmark reports success. Do not trade a layout that
   succeeds now for anything else: read the successful episodes too and keep what made them
   work.
2. Score, where no failing layout can be made to succeed this round: RoboDojo's per-episode
   `score` (1 on success; on failure the partial credit of the task's `get_score` tiers, e.g.
   0.4) on the failing layouts. Aim at the tier most failing layouts miss.
`stages passed` (how many of `run_reward`'s ordered checks passed) is a diagnostic for where
an episode got stuck, not a measure RoboDojo reports. The campaign keeps or rolls back a
change on successes alone: {rollback} or more fewer than the best earlier batch rolls it back.

## Inputs
- `episodes.md`: every dev episode of the parent revision `{parent}` this round (digest, why
  it ended, its run dir). Run dirs hold `workspace/cell_*.py` (the code), `messages.json`
  (the model's turns), `calls.jsonl` (each primitive call and its result), `steps/NNNN/*.png`
  (camera frames), `prompt.txt` (exactly what the policy was told, including the library).
- `history.md`: earlier rounds of this task: what each revision changed and how it did on dev,
  and the path of each earlier round's `FINDINGS.md` (readable: what that worker concluded).
- `candidate/`: a copy of the parent revision; `candidate/tasks/{task}/skills/*.py` and
  `candidate/tasks/{task}/lessons/*.md`.
- API sources (read-only): {api_sources}
- How RoboDojo judges this task (read-only): {scoring}. `run_reward` lists the checks in
  order (a stage passes when its check holds); `reward_manager.py` / `func_parser.py` define
  what each check measures and its thresholds. The policy never sees these; you may use them
  to understand why an episode failed and to write procedures and cards that meet them. Do not
  read other tasks' code, the assets (layouts, meshes) or any other task's runs.

## How RoboDojo ends and judges an episode (read from its evaluation loop)
- The checks are evaluated after EVERY native action, not on a final scene. `check` stages
  pass in order: the first pending stage is tested each action and dropped once all of its
  conditions hold in that same action. A `trigger_check` is tested only at the moment its
  trigger holds (e.g. a bottle back within 30 degrees of upright: the liquid is judged then).
  A `query` counts actions in which its conditions hold; more than the target fails the
  episode at once, fewer than the target at the end fails it too.
- The episode ends the moment everything has passed (success), or when the native-action
  budget is spent or a query overran (failure). So conditions that hold only at one instant
  count (a container still lifted while an object lies in it; a fold at the moment both arms
  arrive home), and a final scene that looks right does not count if that instant never came.
- `truncated` in the policy's state / `calls.jsonl` means only "the benchmark ended the
  episode": in a run that succeeded it appears with budget left. Use `success` and
  `stages passed` in the digest, not `truncated`, to tell success from failure.
- The policy reads a frozen base knowledge file (in `prompt.txt`) that is general: it gives
  no task's thresholds and tells the policy that a lesson stating the task's checks overrides
  it. Cards may state the task's real conditions (the exact thresholds, the order of checks,
  where the checks are more specific than the instruction); quote the check you rely on in
  `evidence`.

An episode whose "end" is infrastructure or the harness (e.g. budget, the model stopping) is
not evidence about manipulation; say so instead of writing a lesson on it.

## Time
You have {minutes} minutes in total, and a replay takes several minutes. Write a candidate that
passes `check` and a first FINDINGS.md by minute {soft}; refine after that. Work saved in
`candidate/`, `replays/` and `FINDINGS.md` survives if the session is cut off.{resume}

## Method
1. For each episode find the earliest wrong step (cell, frames, the model's reasoning before
   it) and classify it: information / judgement / capability / uncertain.  An episode the
   digest marks OVERCLAIM (the policy finished with success, the benchmark judged a
   failure) has a second wrong step: the moment the policy took the task for done.  Name
   what it believed and what in the frames or results contradicted it (judgement); report
   it even when you fix the manipulation, since it stops the policy from fixing it itself.
2. Group episodes by cause. Fix the cause that keeps the most layouts from succeeding; between
   equals, the one that blocks the earliest stage or the lowest score tier. A cause seen
   once is reported, not fixed, unless it is cheap and certain.
3. Make ONE coherent change (one skill module and/or a few cards). Prefer a skill when the fix
   is procedure the policy keeps getting wrong; a card when it is judgement or ordering.
4. Check code fixes without a model: write cells that reproduce the failing part using the
   candidate skill, under `replays/cells-<name>/cell_001.py ...` (cells may `from library.<module>
   import <fn>`; the candidate is installed as `library/`). Run
   `python -m pyrualean.learn.replay replays/cells-<name> --layout <dev layout>`; it prints the
   verdict, each cell's output and the run dir with frames. Budget: {max_replays} replays this
   session, {per_layout} per layout; the command refuses beyond. A replay starts from the
   layout's initial state, not from the failure's state, and takes several minutes (it boots
   the simulator); run one at a time and wait for it. The evidence that matters is a replay
   that gets further than the parent's episode on that layout (success, a higher score or
   more stages passed); a replay that only runs without error shows the code runs, not that
   it helps.
5. Validate: `python -m pyrualean.learn.worker check` must print `ok`.

## Rules for the library
- Skills: functions taking `robo` first, public `robo` methods and pure computation only,
  module top level only definitions/constants, imports limited to math/numpy/statistics and
  `library.*`. Each checks its precondition, re-locates what it manipulates, checks its effect,
  and returns a dict `{{"ok": bool, "why": str, ...measurements}}`; ok means this step, never
  "task done". Return control often: the policy keeps looking at images between calls.
- Cards: `## {task}-<stage>-<nnn>` then one line each: stage, symptom, applies, do, status,
  evidence. stage in observe/locate/grasp/transport/manipulate/place/verify/recover.
  status: `hypothesis`, or `dev_verified` only when a replay or a dev episode in your inputs
  shows the fix working; cite it in evidence as `replays/NN` (a judged replay of this
  session), `dev round N` (an earlier round's batch) or `parent dev`, or the card is refused.
- Never delete a card: narrow it, add counter-evidence, or set `status: retired`.
- No layout ids, absolute object positions, or anything that branches on which scene it is.
  Relative offsets, robot constants and procedures are fine.
- Keep the task's library under {max_chars} characters; the policy reads all of it.
- If nothing is worth changing, change nothing and say why in FINDINGS.md.

## FINDINGS.md
Sections: Prior hypothesis, Episodes (per layout: outcome, score, end, earliest wrong step,
class; overclaims marked), Root causes, Hypothesis, The change (files and why), Evidence
(replays: what they show and do not), Expected effect (per dev layout: success and score
you expect next batch, and why; which layouts that succeed now the change could break), Not
fixed.
- Hypothesis, written before you edit: the cause you will fix, stated so this batch could
  prove it wrong; what you would see in the next batch if it is wrong (the falsifier); and
  at least one alternative explanation of the same episodes, with why you rank it lower.
- Prior hypothesis: the previous round's Hypothesis (its FINDINGS.md, named in
  `history.md`), judged on this batch: supported / refuted / inconclusive, with the
  episodes that decide it. A refuted one is counter-evidence for the cards it produced:
  narrow or retire them rather than adding a new card on top. Round 1: "none".
"""

#: The brief of a worker that is not shown how RoboDojo checks and scores the task.
BLIND_BRIEF = """# Worker brief: {task}, round {round}

You improve the skill library of ONE RoboDojo task, `{task}`, for an online code policy
(GPT-6-astra writing short Python cells against a frozen `robo` API, RGB perception only).
The library is the only thing that changes between rounds: skills (Python) the policy may
import and lesson cards appended to its prompt. Primitives, perception, the base knowledge,
the prompt contract and the benchmark are frozen.

Working directory: this session directory. Write only under `candidate/tasks/{task}/` and
`FINDINGS.md`, `notes/` and `replays/` here. Everything else is read-only evidence, and
nothing outside this directory (except earlier rounds' FINDINGS.md) is readable.

Tools: read files with Read, list and search with Glob and Grep (Read also shows the PNG
frames). Bash runs exactly three commands, from this directory: `ls`, `python -m
pyrualean.learn.replay ...` and `python -m pyrualean.learn.worker check`. Anything else in
Bash is refused: `cat`, `grep`, `jq`, `cd ... &&`, pipes into readers, `python -c`, other
python interpreters or scripts you write. Do not retry a refused command in another form; use
Read/Grep instead (Grep searches JSON files such as `calls.jsonl` and `messages.json`).

## Goal
Your change is judged by the next dev batch: the policy runs your library on the same
{n_dev} dev layouts. Aim at more layouts where the benchmark reports success. Do not trade a
layout that succeeds now for anything else: read the successful episodes too and keep what
made them work. The campaign keeps or rolls back a change on successes alone: {rollback} or
more fewer than the best earlier batch rolls it back.

You are not shown how the benchmark checks or scores this task: no benchmark source, and of
each episode only whether it succeeded. Work out what the task asks from its instruction
(in `episodes.md` and `prompt.txt`), the frames of the successful and failed episodes, and the
outcome. Do not look for the benchmark's code.

## Inputs
- `episodes.md`: every dev episode of the parent revision `{parent}` this round (instruction,
  digest, why it ended, its run dir). Run dirs (under `evidence/`) hold `workspace/cell_*.py`
  (the code), `messages.json` (the model's turns), `calls.jsonl` (each primitive call and its
  result), `steps/NNNN/*.png` (camera frames), `prompt.txt` (exactly what the policy was told,
  including the library). Absolute paths inside these records point at the originals, which
  are not readable: open the same file under the run dir in `evidence/`.
- `evidence/`: the dev run dirs of this task so far (all rounds), copied as above.
- `history.md`: earlier rounds of this task: what each revision changed and how it did on dev,
  and the path of each earlier round's `FINDINGS.md` (readable: what that worker concluded).
- `candidate/`: a copy of the parent revision; `candidate/tasks/{task}/skills/*.py` and
  `candidate/tasks/{task}/lessons/*.md`.
- API sources (read-only): {api_sources}

`truncated` in the policy's state / `calls.jsonl` means only "the benchmark ended the episode";
use `success` in the digest, not `truncated`, to tell success from failure.

An episode whose "end" is infrastructure or the harness (e.g. budget, the model stopping) is
not evidence about manipulation; say so instead of writing a lesson on it.

## Time
You have {minutes} minutes in total, and a replay takes several minutes. Write a candidate that
passes `check` and a first FINDINGS.md by minute {soft}; refine after that. Work saved in
`candidate/`, `replays/` and `FINDINGS.md` survives if the session is cut off.{resume}

## Method
1. For each episode find the earliest wrong step (cell, frames, the model's reasoning before
   it) and classify it: information / judgement / capability / uncertain.  An episode the
   digest marks OVERCLAIM (the policy finished with success, the benchmark judged a
   failure) has a second wrong step: the moment the policy took the task for done.  Name
   what it believed and what in the frames or results contradicted it (judgement); report
   it even when you fix the manipulation, since it stops the policy from fixing it itself.
2. Group episodes by cause. Fix the cause that keeps the most layouts from succeeding; between
   equals, the one that goes wrong earliest in the episode. A cause seen once is reported,
   not fixed, unless it is cheap and certain.
3. Make ONE coherent change (one skill module and/or a few cards). Prefer a skill when the fix
   is procedure the policy keeps getting wrong; a card when it is judgement or ordering.
4. Check code fixes without a model: write cells that reproduce the failing part using the
   candidate skill, under `replays/cells-<name>/cell_001.py ...` (cells may `from library.<module>
   import <fn>`; the candidate is installed as `library/`). Run
   `python -m pyrualean.learn.replay replays/cells-<name> --layout <dev layout>`; it prints the
   verdict, each cell's output and the run dir with frames. Budget: {max_replays} replays this
   session, {per_layout} per layout; the command refuses beyond. A replay starts from the
   layout's initial state, not from the failure's state, and takes several minutes (it boots
   the simulator); run one at a time and wait for it. The evidence that matters is a replay
   that succeeds where the parent's episode on that layout failed, or that visibly gets
   further in the frames; a replay that only runs without error shows the code runs, not that
   it helps.
5. Validate: `python -m pyrualean.learn.worker check` must print `ok`.

## Rules for the library
- Skills: functions taking `robo` first, public `robo` methods and pure computation only,
  module top level only definitions/constants, imports limited to math/numpy/statistics and
  `library.*`. Each checks its precondition, re-locates what it manipulates, checks its effect,
  and returns a dict `{{"ok": bool, "why": str, ...measurements}}`; ok means this step, never
  "task done". Return control often: the policy keeps looking at images between calls.
- Cards: `## {task}-<stage>-<nnn>` then one line each: stage, symptom, applies, do, status,
  evidence. stage in observe/locate/grasp/transport/manipulate/place/verify/recover.
  status: `hypothesis`, or `dev_verified` only when a replay or a dev episode in your inputs
  shows the fix working; cite it in evidence as `replays/NN` (a judged replay of this
  session), `dev round N` (an earlier round's batch) or `parent dev`, or the card is refused.
- Never delete a card: narrow it, add counter-evidence, or set `status: retired`.
- No layout ids, absolute object positions, or anything that branches on which scene it is.
  Relative offsets, robot constants and procedures are fine.
- Keep the task's library under {max_chars} characters; the policy reads all of it.
- If nothing is worth changing, change nothing and say why in FINDINGS.md.

## FINDINGS.md
Sections: Prior hypothesis, Episodes (per layout: outcome, end, earliest wrong step, class;
overclaims marked), Root causes, Hypothesis, The change (files and why), Evidence (replays:
what they show and do not), Expected effect (per dev layout: whether you expect success next
batch, and why; which layouts that succeed now the change could break), Not fixed.
- Hypothesis, written before you edit: the cause you will fix, stated so this batch could
  prove it wrong; what you would see in the next batch if it is wrong (the falsifier); and
  at least one alternative explanation of the same episodes, with why you rank it lower.
- Prior hypothesis: the previous round's Hypothesis (its FINDINGS.md, named in
  `history.md`), judged on this batch: supported / refuted / inconclusive, with the
  episodes that decide it. A refuted one is counter-evidence for the cards it produced:
  narrow or retire them rather than adding a new card on top. Round 1: "none".
"""

#: The brief of a RoboCasa365 campaign's worker (scoring on: it reads the task's success check).
ROBOCASA_BRIEF = """# Worker brief: {task}, round {round}

You improve the skill library of ONE RoboCasa365 kitchen task, `{task}`, for an online code
policy (a model writing short Python cells against a frozen `robo` API, RGB perception only:
the `robocasa_rgb` robot, a mobile base with one arm in a kitchen). The library is the only
thing that changes between rounds: skills (Python) the policy may import and lesson cards
appended to its prompt. Primitives, perception, the base knowledge, the prompt contract and
the benchmark are frozen.

Working directory: this session directory. Write only under `candidate/tasks/{task}/` and
`FINDINGS.md`, `notes/` and `replays/` here. Everything else is read-only evidence.

Tools: read files with Read, list and search with Glob and Grep (Read also shows the PNG
frames). Bash runs exactly three commands, from this directory: `ls`, `python -m
pyrualean.learn.replay ...` and `python -m pyrualean.learn.worker check`. Anything else in
Bash is refused: `cat`, `grep`, `jq`, `cd ... &&`, pipes into readers, `python -c`, other
python interpreters or scripts you write. Do not retry a refused command in another form; use
Read/Grep instead (Grep searches JSON files such as `calls.jsonl` and {turns}).

## Goal
Your change is judged by the next dev batch: the policy runs your library on the same
{n_dev} dev layouts (each a fixed RoboCasa scene seed: kitchen layout, style and object
placement). A digest's "layout N" is the scene seed; the replay command's `--layout` takes the
dev layout index, the `-l<index>-` in the run dir's name.
Aim at more layouts where the benchmark reports success. Do not trade a layout that succeeds
now for anything else: read the successful episodes too and keep what made them work.
RoboCasa reports success only: no score, no partial credit, no stages (the digest's score and
stages fields are empty). The campaign keeps or rolls back a change on successes alone:
{rollback} or more fewer than the best earlier batch rolls it back.

## Inputs
- `episodes.md`: every dev episode of the parent revision `{parent}` this round (digest, why
  it ended, its run dir). Run dirs hold `workspace/cell_*.py` (the code), {turns}
  (the model's turns and reasoning), `calls.jsonl` (each primitive call and its result),
  `steps/NNNN/*.png` (camera frames), `prompt.txt` (exactly what the policy was told,
  including the library), `card.json` (the task card) and `result.json` (the verdict).
- `history.md`: earlier rounds of this task: what each revision changed and how it did on dev,
  and the path of each earlier round's `FINDINGS.md` (readable: what that worker concluded).
- `candidate/`: a copy of the parent revision; `candidate/tasks/{task}/skills/*.py` and
  `candidate/tasks/{task}/lessons/*.md`.
- API sources (read-only): {api_sources}
- How RoboCasa judges this task (read-only): {scoring}. The task's class defines
  `_check_success` (often on a parent class in the same file); the fixture and object code it
  calls is not readable. The policy never sees it; you may use it to understand why an
  episode failed and to write procedures and cards that meet it. Do not read other tasks'
  classes or any other task's runs.

## How RoboCasa ends and judges an episode
- `_check_success` is evaluated by the host after EVERY native action, not on a final scene;
  the episode ends the moment it holds (success). A state that holds only at one instant
  counts.
- Otherwise the episode ends when the task's native-action budget (its official horizon,
  `verdict.step_limit` in `result.json`) is spent, or when the policy's cell budget or wall
  clock runs out (failure).
- The policy is shown no success signal at all (no `done`, no task progress, no check): it
  judges completion from images and primitive results, and may call finish believing it
  succeeded. Cards may state the task's real condition (the exact fixture state or threshold
  `_check_success` tests); quote the check you rely on in `evidence`.

An episode whose "end" is infrastructure or the harness (e.g. budget, the model stopping) is
not evidence about manipulation; say so instead of writing a lesson on it.

## Time
You have {minutes} minutes in total, and a replay takes several minutes. Write a candidate that
passes `check` and a first FINDINGS.md by minute {soft}; refine after that. Work saved in
`candidate/`, `replays/` and `FINDINGS.md` survives if the session is cut off.{resume}

## Method
1. For each episode find the earliest wrong step (cell, frames, the model's reasoning before
   it) and classify it: information / judgement / capability / uncertain.  An episode the
   digest marks OVERCLAIM (the policy finished with success, the benchmark judged a
   failure) has a second wrong step: the moment the policy took the task for done.  Name
   what it believed and what in the frames or results contradicted it (judgement); report
   it even when you fix the manipulation, since it stops the policy from fixing it itself.
2. Group episodes by cause. Fix the cause that keeps the most layouts from succeeding. A
   cause seen once is reported, not fixed, unless it is cheap and certain.
3. Make ONE coherent change (one skill module and/or a few cards). Prefer a skill when the fix
   is procedure the policy keeps getting wrong; a card when it is judgement or ordering.
4. Check code fixes without a model: write cells that reproduce the failing part using the
   candidate skill, under `replays/cells-<name>/cell_001.py ...` (cells may `from library.<module>
   import <fn>`; the candidate is installed as `library/`). Run
   `python -m pyrualean.learn.replay replays/cells-<name> --layout <dev layout>`; it prints
   the verdict, each cell's output and the run dir with frames. Budget: {max_replays} replays
   this session, {per_layout} per layout; the command refuses beyond. A replay starts from the
   layout's initial state, not from the failure's state, and takes several minutes (it boots
   the kitchen); run one at a time and wait for it. The evidence that matters is a replay that
   succeeds where the parent's episode on that layout failed; a replay that only runs without
   error shows the code runs, not that it helps.
5. Validate: `python -m pyrualean.learn.worker check` must print `ok`.

## Rules for the library
- Skills: functions taking `robo` first, public `robo` methods and pure computation only,
  module top level only definitions/constants, imports limited to math/numpy/statistics and
  `library.*`. Each checks its precondition, re-locates what it manipulates, checks its effect,
  and returns a dict `{{"ok": bool, "why": str, ...measurements}}`; ok means this step, never
  "task done". Return control often: the policy keeps looking at images between calls.
- Cards: `## {task}-<stage>-<nnn>` then one line each: stage, symptom, applies, do, status,
  evidence. stage in observe/locate/grasp/transport/manipulate/place/verify/recover.
  status: `hypothesis`, or `dev_verified` only when a replay or a dev episode in your inputs
  shows the fix working; cite it in evidence as `replays/NN` (a judged replay of this
  session), `dev round N` (an earlier round's batch) or `parent dev`, or the card is refused.
- Never delete a card: narrow it, add counter-evidence, or set `status: retired`.
- No scene seeds, absolute object positions, or anything that branches on which scene it is
  (kitchens differ in layout and style across seeds). Relative offsets, robot constants and
  procedures are fine.
- Keep the task's library under {max_chars} characters; the policy reads all of it.
- If nothing is worth changing, change nothing and say why in FINDINGS.md.

## FINDINGS.md
Sections: Prior hypothesis, Episodes (per layout: outcome, end, earliest wrong step, class;
overclaims marked), Root causes, Hypothesis, The change (files and why), Evidence (replays:
what they show and do not), Expected effect (per dev layout: success you expect next batch,
and why; which layouts that succeed now the change could break), Not fixed.
- Hypothesis, written before you edit: the cause you will fix, stated so this batch could
  prove it wrong; what you would see in the next batch if it is wrong (the falsifier); and
  at least one alternative explanation of the same episodes, with why you rank it lower.
- Prior hypothesis: the previous round's Hypothesis (its FINDINGS.md, named in
  `history.md`), judged on this batch: supported / refuted / inconclusive, with the
  episodes that decide it. A refuted one is counter-evidence for the cards it produced:
  narrow or retire them rather than adding a new card on top. Round 1: "none".
"""


def _replaced(text: str, *pairs: tuple[str, str]) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"brief text to replace not found once: {old[:60]!r}")
        text = text.replace(old, new)
    return text


#: The brief of a RoboCasa365 worker with scoring off: BLIND_BRIEF with the robot, the run
#: records (the Codex CLI's events instead of messages.json) and RoboCasa's ending of an episode.
ROBOCASA_BLIND_BRIEF = _replaced(
    BLIND_BRIEF,
    (
        "ONE RoboDojo task, `{task}`, for an online code policy\n(GPT-6-astra writing short "
        "Python cells against a frozen `robo` API, RGB perception only).",
        "ONE RoboCasa365 kitchen task, `{task}`, for an online code\npolicy (a model writing "
        "short Python cells against a frozen `robo` API, RGB perception only:\nthe "
        "`robocasa_rgb` robot, a mobile base with one arm in a kitchen).",
    ),
    ("`calls.jsonl` and `messages.json`).", "`calls.jsonl` and {turns})."),
    (
        "{n_dev} dev layouts. Aim at",
        "{n_dev} dev layouts (each a fixed RoboCasa scene seed: kitchen layout, style and\n"
        "object placement; a digest's \"layout N\" is the scene seed, the replay command's\n"
        "`--layout` takes the dev layout index, the `-l<index>-` in the run dir's name). Aim at",
    ),
    (
        "`messages.json` (the model's turns)",
        "{turns} (the model's turns and reasoning)",
    ),
    (
        "`truncated` in the policy's state / `calls.jsonl` means only \"the benchmark ended the "
        'episode";\nuse `success` in the digest, not `truncated`, to tell success from failure.',
        "An episode ends the moment the benchmark judges the task done (success), or when the "
        "task's\nnative-action budget (`verdict.step_limit` in `result.json`), the policy's cell "
        "budget or its\nwall clock runs out (failure). The policy is shown no success signal "
        "either: it judges\ncompletion from the images and the primitives' results.",
    ),
)

SCORING_MODES = ("on", "off")

#: What an episode run dir may show a worker with scoring off: what the policy saw and did.
#: Not ``bench/`` (RoboDojo's tree, its task and reward code included) nor its logs.
EVIDENCE_FILES = (
    "workspace", "steps", "images", "card", "card.json", "calls.jsonl", "messages.json",
    "tool_turns.json", "prompt.txt", "last_message.md", "openai_events.jsonl",
    "openai_stderr.txt", "replay_results.json", "replay_events.jsonl", "replay_stderr.txt",
)  # fmt: skip
#: A RoboCasa run dir's: also the model's turns of a Codex policy (its events);
#: not ``kitchen.log`` and ``host.json`` (the kitchen's interpreter and its packages).
ROBOCASA_EVIDENCE_FILES = (*EVIDENCE_FILES, "codex_events.jsonl", "codex_stderr.txt")
#: Verdict fields withheld with scoring off: they name the task's checks and its progress.
VERDICT_HIDDEN = ("stages_passed", "stages_total", "blocking_stage", "score")


def evidence_files(config: dict[str, Any]) -> tuple[str, ...]:
    return ROBOCASA_EVIDENCE_FILES if robocasa(config) else EVIDENCE_FILES


def mirror_run(run_dir: Path, target: Path, names: tuple[str, ...] = EVIDENCE_FILES) -> None:
    """A copy of the episode run dir ``run_dir`` with only ``names`` (hard links; by default
    :data:`EVIDENCE_FILES`) and a ``result.json`` without :data:`VERDICT_HIDDEN`."""

    target.mkdir(parents=True)
    for name in names:
        source = run_dir / name
        if source.is_dir():
            shutil.copytree(source, target / name, copy_function=os.link)
        elif source.is_file():
            os.link(source, target / name)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if isinstance(result.get("verdict"), dict):
            result["verdict"] = {
                k: v for k, v in result["verdict"].items() if k not in VERDICT_HIDDEN
            }
        _write(target / "result.json", json.dumps(result, indent=1) + "\n")


def evidence_path(session: Path, run_dir: str | Path) -> Path:
    """Where a dev run dir (``.../runs/dev/<task>/<run>/episode``) is in a session's
    ``evidence/``."""

    return session / "evidence" / Path(run_dir).parent.name / "episode"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def prepare(
    session: Path,
    *,
    task: str,
    round_index: int,
    parent: str,
    library: Library,
    campaign: Path,
    config_path: Path,
    config: dict[str, Any],
    episodes: str,
    history: str,
    scoring: str = "on",
) -> None:
    if scoring not in SCORING_MODES:
        raise ValueError(f"scoring must be one of {SCORING_MODES}, not {scoring!r}")
    session.mkdir(parents=True, exist_ok=True)
    candidate = session / "candidate"
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir()
    for name in library.manifest(parent)["files"]:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(library.revisions / parent / name, target)
        target.chmod(0o644)
    (candidate / "tasks" / task / "skills").mkdir(parents=True, exist_ok=True)
    (candidate / "tasks" / task / "lessons").mkdir(parents=True, exist_ok=True)
    pristine = session / ".parent"
    if pristine.exists():
        shutil.rmtree(pristine)
    shutil.copytree(candidate, pristine)
    spec = {
        "task": task,
        "round": round_index,
        "parent": parent,
        "campaign": str(campaign),
        "config": str(config_path),
        "scoring": scoring,
    }
    package = repo = Path(__file__).resolve().parents[1]
    if scoring == "off":
        # the package the policy runs, without the learning loops (this brief's sighted version
        # in learn/ states RoboDojo's checks; evolve/ reads its stages)
        repo = session / "sources" / "pyrualean"
        if repo.exists():
            shutil.rmtree(repo)
        shutil.copytree(
            package,
            repo,
            ignore=lambda d, names: [
                n
                for n in names
                if n == "__pycache__" or (n in ("learn", "evolve") and Path(d) == package)
            ],
        )
        evidence = session / "evidence"
        if evidence.exists():
            shutil.rmtree(evidence)
        evidence.mkdir()
        runs = campaign.resolve() / "runs" / "dev" / task
        for run_dir in sorted(runs.glob("*/episode")) if runs.is_dir() else []:
            mirror_run(run_dir, evidence_path(session, run_dir), evidence_files(config))
    _write(session / "session.json", json.dumps(spec, indent=1) + "\n")
    api = ("robodojo.py", "robodojo_xpl_rgb.py", "_robot.py")
    brief = BRIEF if scoring == "on" else BLIND_BRIEF
    if robocasa(config):
        api = ("robocasa_rgb.py", "_robot.py")
        brief = ROBOCASA_BRIEF if scoring == "on" else ROBOCASA_BLIND_BRIEF
    # where a run dir holds the model's turns: the Codex CLI's events (runtime codex) or
    # messages.json
    turns = (
        "`codex_events.jsonl`"
        if config["execution"].get("runtime") == "codex"
        else "`messages.json`"
    )
    sources = ", ".join(str(repo / name) for name in api)
    worker = config["worker"]
    minutes = int(worker["seconds"] // 60)
    _write(
        session / "BRIEF.md",
        brief.format(
            task=task,
            turns=turns,
            round=round_index,
            parent=parent,
            api_sources=sources,
            scoring=", ".join(str(p) for p in scoring_sources(config, task)),
            minutes=minutes,
            soft=int(minutes * 0.75),
            resume="{resume}",
            max_replays=worker["max_replays"],
            per_layout=worker["max_replays_per_layout"],
            max_chars=worker["max_library_chars"],
            n_dev=len(config["dev"]["layouts"]),
            rollback=config["learning"]["rollback_if_worse_by"],
        )
        + (
            SCENE_NOTES.format(
                heading=EXPANSION_HEADING,
                paths="\n".join(f"- `{path}`" for path in expansion_sources(config, task)),
            )
            if expansion_sources(config, task)
            else ""
        ),
    )
    _write(session / "episodes.md", episodes)
    _write(session / "history.md", history)


SCENE_NOTES = """
## Scene notes
Each dev layout's prompt ends with notes on its scene (`{heading}` in `prompt.txt`), written
once, before the campaign, from the layout's step-0 head image by a separate model that knows
nothing about the robot API or the benchmark. They are frozen: the policy gets the same notes
every round, and the library cannot change them. The notes, one file per dev layout (readable):
{paths}
"""


def expansion_sources(config: dict[str, Any], task: str) -> list[Path]:
    """The files of ``task``'s expansions (``config["expansions"]``; see ``episode.py``)."""

    expansions = (config.get("expansions") or {}).get(task) or {}
    return [Path(expansions[key]["source"]) for key in sorted(expansions)]


def scoring_sources(config: dict[str, Any], task: str) -> list[Path]:
    """RoboDojo's success definition of ``task``: its checks, and what each check measures.
    RoboCasa's: the file of the kitchen package that defines the task's class."""

    if robocasa(config):
        kitchen = Path(config["sim_packages"]).resolve() / "robocasa" / "environments" / "kitchen"
        pattern = re.compile(rf"^class {re.escape(task)}\b", re.MULTILINE)
        found = [p for p in sorted(kitchen.rglob("*.py")) if pattern.search(p.read_text())]
        if len(found) != 1:
            raise ValueError(f"{task}: {len(found)} files of {kitchen} define its class")
        return found
    root = Path(config["source_root"]).resolve()
    return [
        root / "task" / "RoboDojo" / "tasks" / f"{task}.py",
        root / "task" / "RoboDojo" / "config" / f"{task}.yml",
        root / "env" / "reward_manager" / "reward_manager.py",
        root / "env" / "reward_manager" / "func_parser.py",
    ]


def other_judges(config: dict[str, Any], task: str, evidence: Path) -> list[str]:
    """Deny rules for every other task's checks and config: ``--add-dir`` opens the directories
    of :func:`scoring_sources`, and each episode's ``episode/bench/`` under ``evidence`` is a
    whole RoboDojo copy; a worker reads only its own task's success definition (from the
    source root)."""

    own = set(scoring_sources(config, task))
    # RoboCasa: the folder of the task's file holds the other tasks' classes
    folders = sorted(
        {path.parent for path in own if path.parent.name in ("tasks", "config") or robocasa(config)}
    )
    return [
        *(
            f"Read(/{path})"
            for folder in folders
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path not in own
        ),
        *(f"{tool}(/{folder}{suffix})" for tool in ("Glob", "Grep") for folder in folders
          for suffix in ("", "/**")),
        *(f"{tool}(/{evidence}/**/episode/bench/**)" for tool in ("Read", "Glob", "Grep")),
    ]  # fmt: skip


#: Bash commands that read files: denied, so every read goes through the path-scoped Read,
#: Glob and Grep tools (Claude Code otherwise auto-allows read-only shell commands anywhere).
#: Not ``python``: a deny rule outranks every allow rule, and ``Bash(python:*)`` would refuse
#: the two allowed ``python -m pyrualean.learn...`` commands; under ``dontAsk`` any other
#: ``python ...`` is refused anyway, being on no allow list.
SHELL_READERS = (
    "cat", "head", "tail", "less", "more", "grep", "rg", "find", "sed", "awk", "jq", "cd",
    "wc", "od", "xxd", "strings", "cp", "bash", "sh", "for",
)  # fmt: skip


def earlier_findings(session: Path) -> list[Path]:
    """The FINDINGS.md of this task's earlier rounds (``history.md`` names them)."""

    spec = json.loads((session / "session.json").read_text())
    tasks = Path(spec["campaign"]).resolve() / "tasks" / spec["task"]
    return [tasks / f"worker-r{number}" / "FINDINGS.md" for number in range(1, int(spec["round"]))]


def transcripts(session: Path) -> Path:
    """Where a session's transcripts go: outside anything the worker may read (a model that
    reads its own earlier outputs is refused by the worker's usage policy)."""

    spec = json.loads((session / "session.json").read_text())
    return Path(spec["campaign"]) / ".transcripts" / spec["task"] / session.name


def worker_environment(config: dict[str, Any], env: dict[str, str]) -> dict[str, str]:
    """The worker CLI's environment: the campaign's, minus the policy's key variable (the
    worker's own token, ``worker.token_env``, is the campaign's: see ``use_key``)."""

    return {
        k: v
        for k, v in env.items()
        if k != config["execution"]["api_key_env"] and not k.startswith("PYRUALEAN_LEARN")
    }


def isolation(config: dict[str, Any]) -> list[str]:
    """CLI flags so that the user's own Claude settings (whose ``env`` would set another
    token, overriding this campaign's) do not apply; the gateway settings come from the
    campaign's ``worker.settings``."""

    return [
        "--setting-sources",
        "project,local",
        "--settings",
        json.dumps(config["worker"].get("settings") or {}),
    ]


def command(config: dict[str, Any], session: Path) -> list[str]:
    worker = config["worker"]
    here = "/" + str(session.resolve())  # "//abs/path": an absolute path rule
    writable = [f"{here}/candidate/**", f"{here}/replays/**", f"{here}/notes/**"]
    spec = json.loads((session / "session.json").read_text())
    evidence = Path(spec["campaign"]).resolve() / "runs" / "dev" / spec["task"]
    sources = Path(__file__).resolve().parents[1]
    readable = [str(evidence), str(sources)]
    scoring = scoring_sources(config, spec["task"])
    blind = spec.get("scoring", "on") == "off"
    hidden: list[str] = []
    if blind:
        # everything is in the session (evidence/, sources/): nothing outside it is readable,
        # and what holds RoboDojo's code or the unfiltered runs is denied outright
        hidden = [
            # the benchmark: RoboDojo's tree, or the kitchen's packages (robocasa, robosuite)
            str(Path(config["sim_packages" if robocasa(config) else "source_root"]).resolve()),
            str(evidence.parent.parent),
            str(Path(spec["campaign"]).resolve() / ".replays"),
            str(sources / "learn"),
            str(sources / "evolve"),
        ]
        readable, scoring = [], []
    allowed = [
        *(
            f"{tool}(/{path}/**)"
            for tool in ("Read", "Glob", "Grep")
            for path in (session.resolve(), *readable)
        ),
        *(f"Read(/{path})" for path in scoring),
        *(f"Read(/{path})" for path in earlier_findings(session)),
        *(f"Read(/{path})" for path in expansion_sources(config, spec["task"])),
        *(f"{tool}({path})" for tool in ("Write", "Edit") for path in writable),
        f"Write({here}/FINDINGS.md)",
        f"Edit({here}/FINDINGS.md)",
        "Bash(python -m pyrualean.learn.replay:*)",
        "Bash(python -m pyrualean.learn.worker check)",
        "Bash(ls:*)",
    ]
    brief = (session / "BRIEF.md").read_text().replace("{resume}", resume_note(session))
    argv = [
        worker["command"],
        "-p",
        brief + "\nStart now. End with a one-paragraph summary of the change.",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        "Read,Glob,Grep,Write,Edit,Bash",  # no subagents, no web: one inspectable session
        "--permission-mode",
        "dontAsk",  # anything not allowed below is refused, edits included
        "--allowedTools",
        *allowed,
        "--disallowedTools",
        *(f"{tool}(/{path})" for tool in ("Write", "Edit") for path in scoring),
        *(f"{tool}(/{path}/**)" for tool in ("Read", "Glob", "Grep") for path in hidden),
        *(other_judges(config, spec["task"], evidence) if not blind else []),
        *(f"Bash({name}:*)" for name in SHELL_READERS),
        "--no-session-persistence",
        *isolation(config),
        *(
            ["--add-dir", *readable, *sorted({str(path.parent) for path in scoring})]
            if not blind
            else []
        ),
        "--effort",
        worker["effort"],
    ]
    if worker.get("model"):
        argv += ["--model", worker["model"]]
    return argv


def run(config: dict[str, Any], session: Path, *, launch: Any = None) -> dict[str, Any]:
    """Run the session, retried once when it fails; returns its record.

    ``launch`` replaces the CLI in tests.  A failed attempt is kept as ``worker-a<N>.json``.
    The retry continues from what that attempt left (candidate, replays, findings), told so by
    :func:`resume_note`; only a stuck attempt's work is discarded.  Transcripts are kept outside
    what the worker may read.
    """

    record_path = session / "worker.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    from .runner import ServiceDown, environment, pause, paused

    campaign = Path(json.loads((session / "session.json").read_text())["campaign"])
    if (why := paused(campaign)) is not None:
        raise ServiceDown(f"campaign paused: {why['reason']}")
    env = worker_environment(config, environment(config))
    policy_bin = str(Path(sys.executable).parent)
    repo_src = str(Path(__file__).resolve().parents[2])
    env.update(
        PATH=policy_bin + os.pathsep + env.get("PATH", ""),
        PYTHONPATH=os.pathsep.join(p for p in (repo_src, env.get("PYTHONPATH", "")) if p),
        PYRUALEAN_LEARN_SESSION=str(session.resolve()),
        # a replay is a whole simulator episode: minutes, not the Bash tool's default 2
        BASH_DEFAULT_TIMEOUT_MS=str(int(config["worker"]["replay_watchdog_seconds"] * 1000)),
        BASH_MAX_TIMEOUT_MS=str(int(config["worker"]["replay_watchdog_seconds"] * 1000)),
    )
    started = time.time()
    attempts = int(config["worker"]["attempts"])
    _hide_transcripts(session)
    for attempt in range(attempts):
        # an earlier attempt's candidate, replays and findings are kept: the next one goes on
        # from them (resume_note says where it stopped)
        if launch is not None:
            outcome = launch(session, env)
        else:
            outcome = _claude(command(config, session), session, env, config["worker"])
        outcome["attempt"] = attempt
        if service_error(outcome):
            # the worker's model service failed: not this round's failure; pause and keep the
            # session to be run again, from the parent, when the campaign resumes
            failed = len(list(session.glob("worker-service-*.json")))
            (session / f"worker-service-{failed}.json").write_text(json.dumps(outcome, indent=1))
            restore(session)
            pause(campaign, f"worker model service failed: {str(outcome.get('result'))[:200]}")
            raise ServiceDown(f"worker model service failed: {outcome.get('result')}")
        if succeeded(outcome) or attempt == attempts - 1:
            break
        (session / f"worker-a{attempt}.json").write_text(json.dumps(outcome, indent=1) + "\n")
        if outcome.get("stalled") and not str(outcome["stalled"]).startswith("wall"):
            restore(session)  # stuck, not slow: its half-written work is not trusted
    outcome["seconds"] = round(time.time() - started, 1)
    record_path.write_text(json.dumps(outcome, indent=1) + "\n")
    return outcome


#: What the Claude CLI reports when its own service, not the session, failed.
SERVICE_ERRORS = re.compile(
    r"Unable to connect|ConnectionRefused|ECONNRESET|ECONNREFUSED|ETIMEDOUT|fetch failed|"
    r"overloaded|Internal server error|socket hang up|API Error: 5\d\d|Request timed out",
    re.IGNORECASE,
)
#: A refusal by the worker's model is the session's outcome, not an outage: retried, not paused.
REFUSAL = re.compile(r"Usage Policy|violate", re.IGNORECASE)


def service_error(outcome: dict[str, Any]) -> bool:
    text = str(outcome.get("result") or "")
    return (
        bool(outcome.get("is_error"))
        and bool(SERVICE_ERRORS.search(text))
        and not REFUSAL.search(text)
    )


def _hide_transcripts(session: Path) -> None:
    """Move transcripts out of the session (older layouts kept them there), and record an
    attempt that a stopped campaign interrupted, so the next one resumes from its work."""

    target = transcripts(session)
    target.mkdir(parents=True, exist_ok=True)
    for source in (session, session / ".transcripts"):
        for name in ("worker.stream.jsonl", "worker.stderr"):
            if (source / name).exists():
                with (target / name).open("a") as out:
                    out.write((source / name).read_text())
                (source / name).unlink()
    if (session / ".transcripts").is_dir():
        (session / ".transcripts").rmdir()
    stream = target / "worker.stream.jsonl"
    started = stream.exists() and stream.stat().st_size > 0
    recorded = len(list(session.glob("worker-a*.json")))
    attempts = stream.read_text().count('"subtype":"init"') if started else 0
    if attempts > recorded:  # an attempt ran but left no record: the campaign was stopped
        (session / f"worker-a{recorded}.json").write_text(
            json.dumps({"exit": None, "stalled": "interrupted: campaign stopped"}) + "\n"
        )


def resume_note(session: Path) -> str:
    """What an earlier, cut-off attempt left, for the attempt that goes on from it."""

    earlier = sorted(session.glob("worker-a*.json"))
    if not earlier:
        return ""
    last = json.loads(earlier[-1].read_text())
    why = last.get("stalled") or str(last.get("result") or f"exit {last.get('exit')}")[:120]
    replays = (
        json.loads((session / "replays.json").read_text())
        if (session / "replays.json").exists()
        else []
    )
    blind = json.loads((session / "session.json").read_text()).get("scoring", "on") == "off"
    lines = [
        "",
        "",
        f"## Resuming: an earlier session on this round was cut off ({why})",
        "Its work is still here: the edited `candidate/`, `replays/` (cells and run dirs) and,",
        "if written, `FINDINGS.md`/`notes/`. Do not redo it; read it, finish the change, run",
        "`check`, and write FINDINGS.md. Replays already used count against your budget:",
    ]
    for number, used in enumerate(replays):
        entry = used.get("entry") or {}
        lines.append(
            f"- replay {number:02d}: layout {used['layout']}, cells {Path(used['cells']).name},"
            f" success {entry.get('success')}"
            + ("" if blind else f", stages {entry.get('stages')}")
        )
    if not replays:
        lines.append("- none")
    return "\n".join(lines)


def restore(session: Path) -> None:
    """The candidate back to the parent revision; an unfinished attempt's findings removed."""

    if (session / ".parent").is_dir():
        shutil.rmtree(session / "candidate")
        shutil.copytree(session / ".parent", session / "candidate")
    (session / "FINDINGS.md").unlink(missing_ok=True)


def succeeded(outcome: dict[str, Any]) -> bool:
    return outcome.get("exit") == 0 and not outcome.get("is_error") and not outcome.get("stalled")


def _replay_running(session: Path) -> bool:
    started = (
        len(list((session / "replays").glob("[0-9][0-9]"))) if (session / "replays").is_dir() else 0
    )
    used = session / "replays.json"
    finished = len(json.loads(used.read_text())) if used.exists() else 0
    return started > finished


def _claude(argv: list[str], session: Path, env: dict[str, str], worker: dict) -> dict:
    """The CLI with a wall clock and an idle watchdog (no transcript output for too long)."""

    stream = transcripts(session) / "worker.stream.jsonl"
    stream.parent.mkdir(parents=True, exist_ok=True)
    offset = stream.stat().st_size if stream.exists() else 0  # earlier attempts' lines
    stopped = None
    with stream.open("a") as out, (stream.parent / "worker.stderr").open("a") as err:
        process = subprocess.Popen(
            argv, cwd=session, env=env, stdout=out, stderr=err, start_new_session=True
        )
        started = time.time()
        while process.poll() is None:
            time.sleep(5)
            idle = time.time() - max(stream.stat().st_mtime, started)
            limit = worker["idle_seconds"] + (
                worker["replay_watchdog_seconds"] if _replay_running(session) else 0
            )
            if time.time() - started > worker["seconds"]:
                stopped = "wall clock"
            elif idle > limit:
                stopped = f"no output for {int(idle)} s"
            if stopped:
                from ..evolve.launcher import stop_tree

                stop_tree(process)
                break
    outcome: dict[str, Any] = {"exit": 124 if stopped else process.returncode}
    if stopped:
        outcome["stalled"] = stopped
    final = None
    with stream.open() as handle:
        handle.seek(offset)
        lines = handle.read().splitlines()
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "result":
            final = event
    if final:
        outcome.update(
            {k: final.get(k) for k in ("result", "total_cost_usd", "num_turns", "is_error")}
        )
        outcome["usage"] = final.get("usage")
    return outcome


# -- acceptance ------------------------------------------------------------------


def problems(session: Path, library: Library) -> list[str]:
    """Every reason the session's candidate cannot become a revision (empty: it can)."""

    spec = json.loads((session / "session.json").read_text())
    task, parent = spec["task"], spec["parent"]
    limit = load(spec["config"])["worker"]["max_library_chars"]
    candidate = session / "candidate"
    found = validate(candidate, allow_empty=True)
    before = library.manifest(parent)["files"]
    after = {
        p.relative_to(candidate).as_posix(): p for p in sorted(candidate.rglob("*")) if p.is_file()
    }
    for name in sorted(set(before) | set(after)):
        changed = (
            name not in before
            or name not in after
            or ((library.revisions / parent / name).read_bytes() != after[name].read_bytes())
        )
        if changed and not name.startswith(f"tasks/{task}/"):
            found.append(f"{name}: only tasks/{task}/ may change")
        if name in before and name not in after and LESSON_RE.fullmatch(name):
            found.append(f"{name}: a lessons file was deleted (retire cards instead)")
    replays = session / "replays.json"
    # replays/NN of this session that the benchmark judged (NN from the run dir: an
    # interrupted replay leaves a folder but no entry, so it is not the entry's position)
    judged = {
        match.group(1)
        for r in (json.loads(replays.read_text()) if replays.exists() else [])
        if (r.get("entry") or {}).get("kind") == "policy"
        and (match := re.search(r"/replays/(\d{2})/", (r.get("entry") or {}).get("run_dir", "")))
    }
    size = 0
    for name, path in after.items():
        if not name.startswith(f"tasks/{task}/"):
            continue
        text = path.read_text()
        size += len(text)
        if not LESSON_RE.fullmatch(name):
            if not SKILL_RE.fullmatch(name):
                found.append(f"{name}: not a skill or lessons file")
            continue
        old = (
            {c["id"]: c for c in parse_lessons((library.revisions / parent / name).read_text())}
            if name in before
            else {}
        )
        new = {c["id"]: c for c in parse_lessons(text)}
        for missing in sorted(old.keys() - new.keys()):
            found.append(f"{name}#{missing}: card removed (retire it instead)")
        for identifier, card in new.items():
            promoted = card.get("status") == "dev_verified" and (
                old.get(identifier, {}).get("status") != "dev_verified"
                or card != old.get(identifier)
            )
            if promoted and not _cites_evidence(card.get("evidence", ""), judged, spec["round"]):
                found.append(
                    f"{name}#{identifier}: dev_verified without replay or dev evidence (cite "
                    "`replays/NN` of a judged replay of this session, `dev round N` of an "
                    "earlier round, or `parent dev`)"
                )
    if size > limit:
        found.append(f"tasks/{task}: {size} characters, over the {limit} the policy reads")
    return found


def _cites_evidence(evidence: str, judged: set[str], round_index: int) -> bool:
    """``evidence`` names a judged replay of this session or a dev batch the worker was given:
    ``replays/NN`` (``replay NN``), ``dev round N`` / ``round N`` with N before this round, or
    ``parent dev`` (the batch of the parent revision)."""

    if any(n in judged for n in re.findall(r"\breplays?[/ ](\d{2})\b", evidence)):
        return True
    if re.search(r"\bparent(?:'s)? dev\b", evidence):
        return True
    return any(int(n) < round_index for n in re.findall(r"\bround (\d+)\b", evidence))


def accept(session: Path, library: Library) -> dict[str, Any]:
    """Freeze the candidate; ``revision`` equals the parent when nothing changed."""

    spec = json.loads((session / "session.json").read_text())
    parent = spec["parent"]
    record = (
        json.loads((session / "worker.json").read_text())
        if (session / "worker.json").exists()
        else {}
    )
    found = problems(session, library)
    if found:
        return {"accepted": False, "revision": parent, "problems": found}
    candidate = session / "candidate"
    current = {
        p.relative_to(candidate).as_posix(): p.read_bytes()
        for p in candidate.rglob("*")
        if p.is_file()
    }
    unchanged = current == {
        name: (library.revisions / parent / name).read_bytes()
        for name in library.manifest(parent)["files"]
    }
    if unchanged and record and not succeeded(record):
        # nothing to run: the round is not spent on the parent (the campaign holds the task)
        why = record.get("stalled") or str(record.get("result") or f"exit {record.get('exit')}")
        return {"accepted": False, "revision": parent, "problems": [f"worker failed: {why[:200]}"]}
    if unchanged:
        return {"accepted": True, "revision": parent, "changed": False}
    note = f"round {spec['round']} worker; see {session / 'FINDINGS.md'}"
    try:
        revision = library.freeze(session / "candidate", parent=spec["parent"], note=note)
    except LibraryError as exc:
        return {"accepted": False, "revision": spec["parent"], "problems": [str(exc)]}
    return {"accepted": True, "revision": revision, "changed": revision != spec["parent"]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m pyrualean.learn.worker")
    parser.add_argument("action", choices=["check"])
    parser.parse_args(argv)
    session = os.environ.get("PYRUALEAN_LEARN_SESSION")
    if not session:
        sys.stderr.write("PYRUALEAN_LEARN_SESSION is not set\n")
        return 2
    spec = json.loads((Path(session) / "session.json").read_text())
    library = Library(Path(spec["campaign"]) / "library" / spec["task"])
    found = problems(Path(session), library)
    sys.stdout.write("ok\n" if not found else "\n".join(found) + "\n")
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
