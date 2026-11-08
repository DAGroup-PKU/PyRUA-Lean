# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""XPolicyLab's own Pi_05 policy, served over openpi's websocket protocol.

    <openpi venv>/bin/python pi05_server.py --xpolicylab-root <XPolicyLab> \
        --checkpoint <.../RoboDojo-sim-arx_x5-joint-0/59999> --port <port>

Run by :mod:`.pi05_xpolicylab` with the interpreter of XPolicyLab's ``policy/Pi_05/openpi``
uv environment (JAX, openpi, numpy 2), which this library's environment cannot share a
process with.  This file is run by path and imports nothing of PyRUA-Lean.

The policy is ``policy/Pi_05/model.py``'s ``Model``, the object RoboDojo's own evaluation of
Pi_05 serves: one request is one RoboDojo observation as the evaluation client sends it
(``vision/<camera>/color``, ``state``, ``instruction``) and the answer is the action chunk
``Model.get_action`` returns, a list of the client's action dicts.  The state is packed and
the actions unpacked by XPolicyLab's ``pack_robot_state`` / ``unpack_robot_state``, arm by
arm, the layout the published checkpoints were trained on.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: XPolicyLab Pi_05's ``deploy.yml`` for the published RoboDojo checkpoints.
TRAIN_CONFIG_NAME = "pi05_base_aloha_full_sim_arx-x5_seed_0"
REPO_ID = "arx_x5_sim"


class Pi05Policy:
    """``Model`` of XPolicyLab's ``policy/Pi_05`` behind openpi's ``BasePolicy`` interface."""

    def __init__(
        self,
        checkpoint: str,
        *,
        env_cfg_type: str = "arx_x5",
        action_type: str = "joint",
        train_config_name: str = TRAIN_CONFIG_NAME,
        repo_id: str = REPO_ID,
    ) -> None:
        from XPolicyLab.policy.Pi_05.model import Model

        self.model = Model(
            {
                "task_name": "pyrualean",
                "env_cfg_type": env_cfg_type,
                "action_type": action_type,
                "model_path": checkpoint,
                "train_config_name": train_config_name,
                "repo_id": repo_id,
            }
        )

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.model.reset()
        self.model.update_obs(obs)
        actions = self.model.get_action()
        return {
            "actions": [
                {key: np.asarray(value, dtype=np.float64) for key, value in action.items()}
                for action in actions
            ]
        }

    def reset(self) -> None:
        self.model.reset()


def _watch_parent(pid: int, every_s: float = 2.0) -> None:
    """Exit once process ``pid`` has gone, however it went.

    The host that started this server is XPolicyLab's policy server, which its ``eval.sh``
    stops with a plain SIGTERM: no ``atexit`` of it runs, and this server, in a session of
    its own, would outlive it with the card's memory.
    """

    import os
    import threading
    import time

    def watch() -> None:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                logging.info("parent %d has gone: exiting", pid)
                os._exit(0)
            except PermissionError:
                pass  # alive, someone else's
            time.sleep(every_s)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xpolicylab-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--train-config-name", default=TRAIN_CONFIG_NAME)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--parent-pid", type=int, default=None, help="exit when this process is gone"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    if args.parent_pid is not None:
        _watch_parent(args.parent_pid)

    xpolicylab = Path(args.xpolicylab_root).resolve()
    # XPolicyLab imports itself as a package, and reads env_cfg/ from the directory above it
    bench = str(xpolicylab.parent)
    if bench not in sys.path:
        sys.path.insert(0, bench)
    from openpi.serving import websocket_policy_server

    policy = Pi05Policy(
        args.checkpoint,
        env_cfg_type=args.env_cfg_type,
        action_type=args.action_type,
        train_config_name=args.train_config_name,
        repo_id=args.repo_id,
    )
    logging.info("Pi_05 ready: %s", args.checkpoint)
    websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"policy": "XPolicyLab/Pi_05", "checkpoint": args.checkpoint},
    ).serve_forever()


if __name__ == "__main__":
    main()
