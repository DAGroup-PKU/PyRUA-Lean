# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Command line: render the prompt, dump a task card, generate and run policies.

``card`` and ``run`` start the RPent host in a separate process (one episode
per process, hard-killed on timeout); they must be invoked with RPent's
interpreter and environment.  ``api``, ``prompt``, ``summarize`` and
``compare`` need only this package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from ._env import setting
from ._robot import PRIMITIVE_SETS
from .evaluation.compare import main as compare_main
from .evaluation.metrics import load_episode_result, summarize
from .evaluation.rpent import parse_rpent_run
from .prompt import TaskCard, api_reference, render_prompt
from .runner import run_host

HOST_MODULE = "pyrualean.hosts.rpent_{robot}"
#: Extra wall-clock allowed for RPent's servers to start before the policy
#: budget begins (env + Pi0.5 + SAM3 cold start is typically 2-3 minutes).
STARTUP_ALLOWANCE_S = 900.0


def _cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot", default="libero", help="libero | robocasa | robotwin")
    parser.add_argument(
        "--suite", default=None, help="LIBERO suite / RoboCasa split / RoboTwin task config"
    )
    parser.add_argument("--task", required=True, help="LIBERO task index or the robot's task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--rpent-root", default=None)
    parser.add_argument("--python", default=sys.executable, help="interpreter with RPent installed")


def _host_argv(args: argparse.Namespace, command: str) -> list[str]:
    argv = [
        args.python,
        "-m",
        HOST_MODULE.format(robot=args.robot),
        command,
        "--task",
        str(args.task),
        "--seed",
        str(args.seed),
        "--out",
        str(args.out),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    if args.suite:
        argv += ["--suite", args.suite]
    if args.robot == "libero":
        argv += ["--libero-type", args.libero_type]
    if args.cuda_device is not None:
        argv += ["--cuda-device", str(args.cuda_device)]
    if args.rpent_root:
        argv += ["--rpent-root", args.rpent_root]
    return argv


def cmd_card(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    code = run_host(
        _host_argv(args, "card"),
        timeout_s=STARTUP_ALLOWANCE_S,
        log_path=args.out / "host.log",
    )
    card = args.out / "card.json"
    if code != 0 or not card.is_file():
        print(f"card failed (exit {code}); see {args.out / 'host.log'}", file=sys.stderr)
        return 2
    print(card)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    policy = args.policy.resolve()
    argv = _host_argv(args, "run") + ["--policy", str(policy), "--timeout-s", str(args.timeout_s)]
    started = time.perf_counter()
    code = run_host(
        argv,
        timeout_s=args.timeout_s + STARTUP_ALLOWANCE_S,
        log_path=args.out / "host.log",
    )
    result_path = args.out / "result.json"
    if not result_path.is_file():
        result: dict[str, Any] = {
            "schema": "pyrualean-episode-v2",
            "system": "pyrualean",
            "backend": "libero",
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "policy": {"path": str(policy)},
            "environment_success": False,
            "status": "killed",
            "reason": f"host exited {code} without writing result.json",
            "error": None,
            "calls": None,
            "stateful_calls": None,
            "env_steps": None,
            "policy_wall_s": None,
            "total_wall_s": time.perf_counter() - started,
            "timeout_s": args.timeout_s,
            "notes": ["parent hard-killed the host or the host crashed early"],
        }
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    generation = args.generation or policy.parent / "generation.json"
    if Path(generation).is_file():
        result["generation"] = json.loads(Path(generation).read_text(encoding="utf-8"))
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {
        k: result.get(k)
        for k in ("status", "reason", "environment_success", "calls", "env_steps", "policy_wall_s")
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["status"] not in ("killed", "boot_error") else 2


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import generate_with_codex

    card = TaskCard.from_json(args.card)
    if args.transfer:
        card.transfer = True
    record = generate_with_codex(
        card,
        args.out,
        model=args.model,
        reasoning_effort=args.reasoning,
        budget_s=args.budget_s,
        codex_bin=args.codex_bin,
        timeout_s=args.timeout_s,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in (
                    "policy",
                    "policy_lines",
                    "wall_s",
                    "tool_events",
                    "input_tokens",
                    "output_tokens",
                )
            }
        )
    )
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    argv = [
        args.python,
        "-m",
        "pyrualean.play",
        "--robot",
        args.robot,
        "--arm",
        args.arm,
        "--task",
        str(args.task),
        "--seed",
        str(args.seed),
        "--out",
        str(args.out),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--model",
        args.model,
        "--reasoning",
        args.reasoning,
        "--budget-s",
        str(args.budget_s),
        "--codex-bin",
        args.codex_bin,
    ]
    if args.suite:
        argv += ["--suite", args.suite]
    if args.robot == "libero":
        argv += ["--libero-type", args.libero_type]
    if args.cuda_device is not None:
        argv += ["--cuda-device", str(args.cuda_device)]
    if args.rpent_root:
        argv += ["--rpent-root", args.rpent_root]
    if args.max_programs is not None:
        argv += ["--max-programs", str(args.max_programs)]
    if args.base_url:
        argv += ["--base-url", args.base_url]
    if args.api_key_env:
        argv += ["--api-key-env", args.api_key_env]
    argv += ["--images", args.images, "--feedback", args.feedback]
    if args.codex_home:
        argv += ["--codex-home", args.codex_home]
    if args.guides:
        argv += ["--guides"]
    if args.guides_dir:
        argv += ["--guides-dir", args.guides_dir]
    if args.max_turns is not None:
        argv += ["--max-turns", str(args.max_turns)]
    if args.vla_endpoint:
        argv += ["--vla-endpoint", args.vla_endpoint]
    if args.sam3_endpoint:
        argv += ["--sam3-endpoint", args.sam3_endpoint]
    if args.runtime == "claude":
        argv += ["--runtime", "claude", "--claude-cli", args.claude_cli]
    if args.primitives != "full":
        argv += ["--primitives", args.primitives]
    code = run_host(
        argv, timeout_s=args.budget_s + STARTUP_ALLOWANCE_S, log_path=args.out / "host.log"
    )
    result_path = args.out / "result.json"
    if not result_path.is_file():
        print(f"play failed (exit {code}); see {args.out / 'host.log'}", file=sys.stderr)
        return 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "arm",
                    "status",
                    "environment_success",
                    "tool_turns",
                    "env_steps",
                    "policy_wall_s",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] not in ("boot_error", "error") else 2


def _load_any(path: Path) -> Any:
    if path.is_dir():
        if (path / "result.json").is_file():
            return load_episode_result(path)
        return parse_rpent_run(path)
    if path.name == "result.json":
        return load_episode_result(path)
    return parse_rpent_run(path.parent, transcript=path)


def cmd_summarize(args: argparse.Namespace) -> int:
    runs = [_load_any(Path(p)) for p in args.paths]
    header = (
        f"{'system':17} {'task':22} {'seed':>4} {'ok':>3} {'status':10} {'turns':>5} "
        f"{'calls':>5} {'steps':>6} {'wall_s':>7} {'in_tok':>9} {'cached':>9} {'out_tok':>7}"
    )
    print(header)

    def _fmt(value: Any) -> str:
        return "-" if value is None else str(value)

    for run in runs:
        ok = "-" if run.environment_success is None else ("Y" if run.environment_success else "N")
        print(
            f"{run.system[:17]:17} {run.task[:22]:22} {_fmt(run.seed):>4} {ok:>3} "
            f"{_fmt(run.status)[:10]:10} {_fmt(run.turns):>5} {_fmt(run.calls):>5} "
            f"{_fmt(run.env_steps):>6} {(run.wall_time_s or 0):7.1f} "
            f"{_fmt(run.input_tokens):>9} {_fmt(run.cached_input_tokens):>9} "
            f"{_fmt(run.output_tokens):>7}"
        )
    if args.json:
        print(json.dumps(summarize(runs), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyrualean", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    api = sub.add_parser("api", help="print the generated API reference")
    api.add_argument("--primitives", choices=PRIMITIVE_SETS, default="full")

    prompt = sub.add_parser("prompt", help="print the full generation prompt")
    prompt.add_argument("--card", type=Path, default=None)
    prompt.add_argument("--budget-s", type=int, default=1200)
    prompt.add_argument("--system-only", action="store_true")
    prompt.add_argument("--transfer", action="store_true")
    prompt.add_argument("--primitives", choices=PRIMITIVE_SETS, default="full")

    card = sub.add_parser("card", help="boot the simulator and dump a task card")
    _cell_args(card)

    run = sub.add_parser("run", help="run a policy file for one cell (RPent host)")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--timeout-s", type=float, default=1200.0)
    run.add_argument("--generation", type=Path, default=None, help="generation.json to attach")
    _cell_args(run)

    gen = sub.add_parser("generate", help="generate policy.py from a task card with Codex")
    gen.add_argument("--card", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--model", default="gpt-6-astra")
    gen.add_argument("--reasoning", default="xhigh")
    gen.add_argument("--budget-s", type=int, default=1200)
    gen.add_argument("--codex-bin", default="codex")
    gen.add_argument("--timeout-s", type=float, default=3600.0)
    gen.add_argument(
        "--transfer",
        action="store_true",
        help="tell the model the program will run on other seeds (no hard-coded pixels)",
    )
    gen.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible Responses endpoint for Codex (e.g. https://host/v1)",
    )
    gen.add_argument(
        "--api-key-env",
        default=None,
        help="name of the environment variable holding the key for --base-url",
    )

    summ = sub.add_parser("summarize", help="tabulate result.json / RPent run dirs")
    summ.add_argument("paths", nargs="+")
    summ.add_argument("--json", action="store_true")

    play = sub.add_parser("play", help="run one code-arm episode with Codex (RPent host)")
    play.add_argument("--arm", choices=("cells", "program"), required=True)
    play.add_argument("--model", default="gpt-6-astra")
    play.add_argument("--reasoning", default="xhigh")
    play.add_argument("--budget-s", type=float, default=1200.0)
    play.add_argument("--max-programs", type=int, default=None)
    play.add_argument("--codex-bin", default=setting("CODEX_BIN", "codex"))
    play.add_argument("--base-url", default=None)
    play.add_argument("--api-key-env", default=None)
    play.add_argument("--images", choices=("on-motion", "on-demand", "none"), default="on-motion")
    play.add_argument("--feedback", choices=("pure", "rich"), default="rich")
    play.add_argument("--codex-home", default=setting("CODEX_HOME"))
    play.add_argument(
        "--guides",
        action="store_true",
        help="give the agent RPent's LIBERO operating guides in its workspace",
    )
    play.add_argument("--guides-dir", default=None, help="directory holding the three guides")
    play.add_argument("--max-turns", type=int, default=None, help="decision budget (tool calls)")
    play.add_argument("--vla-endpoint", default=None, help="host:port of a shared Pi0.5 server")
    play.add_argument("--sam3-endpoint", default=None, help="host:port of a shared SAM3 server")
    play.add_argument(
        "--runtime",
        choices=("codex", "claude"),
        default=setting("RUNTIME", "codex"),
        help="agent runtime: codex (Codex CLI) or claude (Claude Code via the Claude Agent SDK)",
    )
    play.add_argument(
        "--claude-cli",
        default=setting("CLAUDE_CLI", "claude"),
        help="claude executable for --runtime claude",
    )
    play.add_argument(
        "--primitives",
        choices=PRIMITIVE_SETS,
        default=setting("PRIMITIVES", "full"),
        help="primitive set: full, or no-vla (the VLA primitives hidden and refused)",
    )
    _cell_args(play)

    comp = sub.add_parser("compare", help="strict paired comparison", add_help=False)
    comp.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "api":
        print(api_reference(primitives=args.primitives))
        return 0
    if args.command == "prompt":
        card_obj = TaskCard.from_json(args.card) if args.card else None
        if card_obj is not None and args.transfer:
            card_obj.transfer = True
        bundle = render_prompt(card_obj, budget_s=args.budget_s, primitives=args.primitives)
        print(bundle.system if args.system_only else bundle.text())
        return 0
    if args.command == "card":
        return cmd_card(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    if args.command == "play":
        return cmd_play(args)
    if args.command == "compare":
        return compare_main(args.rest)
    parser.error("unknown command")
    return 2


__all__ = ["main"]
