# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""PyRUA-Lean: a pyautogui-style robot library for code policies.

A policy is an ordinary Python file that defines ``run(robo)``.  ``robo`` is a
typed library (``robo.move_to(...)``, ``robo.pi0_pick(...)``,
``robo.segment(...)``) whose calls are executed by a real backend; the first
backend is RPent's LIBERO toolkit, so the very same primitives that RPent's
tool-calling agent uses are available as plain Python functions.
"""

from ._backend import Backend, CallRecord, EpisodeFinished, Ledger, ToolError
from ._robot import RobotBase
from .libero import (
    BackProjection,
    Contact,
    Gripper,
    LiberoRobot,
    Move,
    MovePose,
    Pick,
    RegionCenter,
    Release,
    Rotation,
    Segment,
    State,
)
from .prompt import PromptBundle, TaskCard, api_reference, knowledge_text, render_prompt
from .robots import ROBOTS, RobotAdapter, get_adapter
from .runner import PolicyLoadError, PolicyOutcome, load_policy, run_policy

__version__ = "0.2.0"

__all__ = [
    "BackProjection",
    "Backend",
    "CallRecord",
    "Contact",
    "EpisodeFinished",
    "Gripper",
    "Ledger",
    "LiberoRobot",
    "Move",
    "MovePose",
    "Pick",
    "PolicyLoadError",
    "PolicyOutcome",
    "PromptBundle",
    "ROBOTS",
    "RegionCenter",
    "Release",
    "RobotAdapter",
    "RobotBase",
    "Rotation",
    "Segment",
    "State",
    "TaskCard",
    "ToolError",
    "api_reference",
    "get_adapter",
    "knowledge_text",
    "load_policy",
    "render_prompt",
    "run_policy",
    "__version__",
]
