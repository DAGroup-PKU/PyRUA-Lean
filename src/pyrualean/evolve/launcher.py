"""A fresh policy process per private instance; no user shell or credential files."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .splits import LayoutRegistry, json_hash, tree_hash


def execution_identity(config: dict[str, Any]) -> str:
    repo = Path(__file__).resolve().parents[3]
    return json_hash(
        {
            "execution": config["execution"],
            "policy": tree_hash(repo / "src"),
            "integration": tree_hash(repo / "src" / "xpolicylab_adapter"),
            "environment": tree_hash(Path(config["source_root"]) / "env_cfg"),
            "benchmark_task": tree_hash(Path(config["source_root"]) / "task"),
            "benchmark_client": tree_hash(Path(config["source_root"]) / "src"),
            "benchmark_environment": tree_hash(Path(config["source_root"]) / "env"),
            "benchmark_utils": tree_hash(Path(config["source_root"]) / "utils"),
            "benchmark_scripts": tree_hash(Path(config["source_root"]) / "scripts"),
            "sam3": config["sam3_sha256"],
        }
    )


def stop_tree(process: subprocess.Popen) -> None:
    descendants = []

    def visit(pid: int) -> None:
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            return
        for child in children:
            visit(int(child))
            descendants.append(int(child))

    visit(process.pid)
    for pid in [*descendants, process.pid]:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for pid in [*descendants, process.pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def launch_worker(config_path: Path, request: dict[str, Any], timeout_s: float) -> int:
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    request_file = output / "request.json"
    request_file.write_text(json.dumps(request, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "pyrualean.evolve.launcher",
        str(config_path),
        str(request_file),
    ]
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        (output / "worker.json").write_text(
            json.dumps({"pid": process.pid, "started": time.time()})
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_tree(process)
            (output / "supervisor.json").write_text(
                json.dumps({"status": "wall_timeout", "seconds": timeout_s})
            )
            return 124
        except BaseException:
            stop_tree(process)
            raise


def run_episode(config_path: Path, request_path: Path) -> None:
    from ..hosts.rpent_robodojo import Cell
    from ..play import play_episode

    config = json.loads(config_path.read_text())
    request = json.loads(request_path.read_text())
    if not config.get("launch_enabled"):
        raise ValueError("launch_enabled is false")
    identity = execution_identity(config)
    if identity != request["execution_sha256"]:
        raise ValueError("frozen execution changed")
    registry = LayoutRegistry(config["layout_manifest"])
    instance = registry.select(request["split"], request["task"], request["layout"])
    if instance["env_config_sha256"] != tree_hash(Path(config["source_root"]) / "env_cfg"):
        raise ValueError("layout validated against a different environment configuration")
    os.environ["SAM3_CHECKPOINT_PATH"] = config["sam3_checkpoint"]
    settings = config["execution"]
    output = Path(request["output"]) / "results" / "episode"
    output.mkdir(parents=True, exist_ok=False)
    result = play_episode(
        Cell(settings["suite"], request["task"], 0, cuda_device=config["gpu"]),
        output,
        arm="cells",
        max_attempts=1,
        model=settings["model"],
        budget_s=settings["wall_seconds"],
        max_turns=settings["max_cells"],
        runtime="openai",
        robot="robodojo_xpl_rgb",
        primitives="no-vla",
        images="on-demand",
        feedback="rich",
        library=request["view"],
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        boot_options={
            "source_root": config["source_root"],
            "eval_seed": request["eval_seed"],
            "private_layout": str(registry.file(instance)),
            "sim_python": config["sim_python"],
        },
        harness_options={
            "candidate": True,
            "max_attempts": settings["local_attempts"],
            "lesson_limit": settings.get("lesson_limit"),
            "skill_limit": settings.get("skill_limit"),
        },
        openai_options={
            "max_requests": settings["max_model_requests"],
            "max_tokens": settings["max_tokens"],
            "context_groups": settings.get("context_groups"),
            "extra_body": settings.get("extra_body", {}),
            "retries": settings["provider_retries"],
        },
    )
    result.update(
        seed=request["layout"],
        client_layout=0,
        split=request["split"],
        episode_id=Path(request["output"]).name,
        instance=registry.identity(instance),
        revision=request["revision"],
        execution_sha256=identity,
    )
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
    temporary.replace(output / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    run_episode(args.config, args.request)


if __name__ == "__main__":
    main()
