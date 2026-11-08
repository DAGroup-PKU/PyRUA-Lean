"""The campaign configuration: loaded, checked, and frozen at init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASKS = (
    "make_toast_random",
    "hang_mugs_random",
    "sweep_blocks_random",
    "plug_in_charger",
    "swap_blocks",
    "make_kong",
    "pour_by_language",
    "organize_table",
    "pick_from_conveyor_by_image",
    "play_Xylophone",
    "store_laptop_and_headphones",
    "insert_key",
    "press_by_number",
    "store_tools_in_toolbox",
    "build_tower",
    "fill_egg_holder",
    "pour_liquid_into_cup",
    "fill_pen_holder",
    "pack_objects_into_box",
    "deposit_coin",
    "align_blocks",
    "play_stacking_toy",
    "match_and_pick_from_conveyor",
    "stack_bowls_random",
    "insert_tubes",
    "fasten_screws",
    "play_tic_tac_toe",
    "pour_balls_into_vase",
    "solve_equation",
    "fold_clothes_random",
)

#: Execution fixed for every episode of a campaign, v0 and learned revisions alike.  The
#: defaults are the submission's (``src/xpolicylab_adapter/deploy.yml``).
EXECUTION_DEFAULTS: dict[str, Any] = {
    "suite": "arx_x5",
    "images": "on-demand",
    "feedback": "pure",
    "max_cells": 100,
    "max_model_requests": 200,
    "max_tokens": 8192,
    "nudges": 20,
    "policy_seconds": 5400,
    "provider_retries": 2,
    "extra_body": {"reasoning_effort": "high"},
}

WORKER_DEFAULTS: dict[str, Any] = {
    "command": "claude",
    "model": None,
    # where the worker CLI gets the key of ``api_key_env`` (when given): ANTHROPIC_API_KEY, or
    # ANTHROPIC_AUTH_TOKEN for a gateway
    "token_env": "ANTHROPIC_API_KEY",
    "effort": "high",
    "seconds": 7200,
    "max_replays": 8,
    "max_replays_per_layout": 3,
    "max_library_chars": 40000,
    "concurrency": 3,
    "replay_watchdog_seconds": 1500,
    "idle_seconds": 900,
    "attempts": 2,
}


class ConfigError(ValueError):
    """A configuration that cannot run."""


def load(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    config["execution"] = {**EXECUTION_DEFAULTS, **config.get("execution", {})}
    config["worker"] = {**WORKER_DEFAULTS, **config.get("worker", {})}
    config.setdefault("learning", {}).setdefault("rollback_if_worse_by", 2)
    config["learning"].setdefault("pause_after_provider_failures", 3)
    config.setdefault("environment", {})
    check(config)
    return config


def robocasa(config: dict[str, Any]) -> bool:
    """A RoboCasa365 campaign (``"benchmark": "robocasa"``): tasks are RoboCasa task names, a
    split's layout ``n`` is the scene seed ``scene_seed(config, split, task, n)`` of the
    ``execution.split`` (``eval_seed`` only labels the split); the robot is ``execution.robot``
    (``robocasa_rgb``)."""

    return config.get("benchmark", "robodojo") == "robocasa"


def scene_seeds(config: dict[str, Any], split: str, task: str) -> list[int]:
    """The scene seeds of a robocasa split, by layout: ``seed_overrides[split][task]`` or
    ``scene_seeds[split]`` (as eval/robocasa's ``seed_overrides``: the paper's CoffeeSetupMug)."""

    name = "test" if split == "test" else "dev"
    override = ((config.get("seed_overrides") or {}).get(name) or {}).get(task)
    return [int(s) for s in (override or config["scene_seeds"][name])]


def scene_seed(config: dict[str, Any], split: str, task: str, layout: int) -> int:
    name = "test" if split == "test" else "dev"
    return scene_seeds(config, name, task)[config[name]["layouts"].index(layout)]


def check(config: dict[str, Any]) -> None:
    tasks = config.get("tasks") or []
    if config.get("benchmark", "robodojo") not in ("robodojo", "robocasa"):
        raise ConfigError("benchmark is robodojo (the default) or robocasa")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ConfigError("tasks must be distinct names")
    if not robocasa(config) and set(tasks) - set(TASKS):
        raise ConfigError(f"tasks must be distinct names from {TASKS}")
    dev, test = config.get("dev") or {}, config.get("test") or {}
    for name, split in (("dev", dev), ("test", test)):
        if split.get("eval_seed") not in (0, 1, 2) or not split.get("layouts"):
            raise ConfigError(f"{name}: eval_seed in 0..2 and a nonempty layout list")
    if dev["eval_seed"] == test["eval_seed"]:
        raise ConfigError("dev and test must use different eval seeds (disjoint layout sets)")
    if robocasa(config):
        seeds = config.get("scene_seeds") or {}
        for name in ("dev", "test"):
            if not isinstance(seeds.get(name), list):
                raise ConfigError(f"robocasa: scene_seeds.{name} is required (a seed per layout)")
        for split, by_task in (config.get("seed_overrides") or {}).items():
            if split not in ("dev", "test") or set(by_task) - set(tasks):
                raise ConfigError("robocasa: seed_overrides is {dev|test: {task: [seed, ...]}}")
        for task in tasks:
            got = {name: scene_seeds(config, name, task) for name in ("dev", "test")}
            for name, split in (("dev", dev), ("test", test)):
                if len(got[name]) != len(split["layouts"]) or len(set(got[name])) != len(got[name]):
                    raise ConfigError(
                        f"robocasa: {task}: one distinct {name} scene seed per layout"
                    )
            if set(got["dev"]) & set(got["test"]):
                raise ConfigError(f"robocasa: {task}: dev and test scene seeds must be disjoint")
    if not isinstance(config.get("rounds"), int) or config["rounds"] < 1:
        raise ConfigError("rounds must be a positive integer (dev batches per task)")
    execution = config["execution"]
    if execution.get("runtime", "openai") not in ("openai", "codex"):
        raise ConfigError("execution.runtime is openai (the default) or codex")
    # codex: execution.codex_bin (e.g. tools/codex-mcp-only) logs in on execution.codex_home by
    # itself, so the campaign holds no endpoint or key of the policy's
    codex = execution.get("runtime") == "codex"
    needed = ("codex_bin", "codex_home") if codex else ("base_url", "api_key_env")
    for field in ("model", *needed):
        if not execution.get(field):
            raise ConfigError(f"execution.{field} is required")
    if config["worker"]["token_env"] not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        raise ConfigError("worker.token_env is ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
    if {"messages", "model", "tools", "tool_choice"} & set(execution["extra_body"]):
        raise ConfigError("extra_body cannot replace the model or the tool protocol")
    if "api_key_env" in config["worker"] and not config["worker"]["api_key_env"]:
        raise ConfigError("worker.api_key_env, when given, names an environment variable")
    slots = config.get("slots") or {}
    if not slots.get("gpus") or int(slots.get("per_gpu", 0)) < 1:
        raise ConfigError("slots.gpus and slots.per_gpu are required")
    budgets = config.get("budgets") or {}
    for field in ("max_launches", "max_hours", "max_infra_retries", "episode_watchdog_seconds"):
        if not isinstance(budgets.get(field), (int, float)) or budgets[field] < 0:
            raise ConfigError(f"budgets.{field} is required")
    if budgets["episode_watchdog_seconds"] <= execution["policy_seconds"]:
        raise ConfigError("the watchdog must leave the episode its policy wall clock and boot")
    fields = ("source_root", "sim_python", "sam3_checkpoint")
    if robocasa(config):
        # the kitchen's interpreter, its site-packages (robocasa, robosuite) and its assets
        fields = ("sim_python", "sim_packages", "robocasa_assets", "sam3_checkpoint")
        for field in ("robot", "primitives", "split", "max_episode_steps"):
            if not execution.get(field):
                raise ConfigError(f"robocasa: execution.{field} is required")
    for field in fields:
        if not config.get(field):
            raise ConfigError(f"{field} is required")
    # expansions: {task: {"s<seed>l<layout>": {source, sha256, text}}}, every dev layout covered
    for task, expansions in (config.get("expansions") or {}).items():
        if task not in tasks:
            raise ConfigError(f"expansions: {task} is not a task of this campaign")
        for layout in dev["layouts"]:
            entry = expansions.get(f"s{dev['eval_seed']}l{layout}") or {}
            if not all(entry.get(field) for field in ("source", "sha256", "text")):
                raise ConfigError(
                    f"expansions.{task}: dev layout {layout} needs source, sha256 and text"
                )


def planned(config: dict[str, Any]) -> dict[str, int]:
    """Episode counts the protocol plans (before infra retries and replays)."""

    tasks = len(config["tasks"])
    dev = tasks * len(config["dev"]["layouts"]) * config["rounds"]
    test = tasks * len(config["test"]["layouts"]) * 2 if config.get("run_test", True) else 0
    return {
        "dev_episodes": dev,
        "test_episodes": test,
        "worker_sessions": tasks * (config["rounds"] - 1),
        "max_replays": tasks * (config["rounds"] - 1) * config["worker"]["max_replays"],
    }
