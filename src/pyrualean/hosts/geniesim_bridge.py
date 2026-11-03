# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Between GenieSim's benchmark client and a program that steps the robot.

GenieSim's benchmark owns the simulator and asks a policy for actions: it sends an
observation and waits for a chunk of actions over a WebSocket.  A PyRUA-Lean episode is
the other way round: a program calls a tool, the tool takes one native action and reads
what came of it.  :class:`Bridge` is the meeting point, and :class:`Server` is the
WebSocket end the benchmark talks to.

One request is answered with **one** action (:data:`CHUNK` = 1), never more.  The client
then takes exactly that action and asks again, so a tool sees the result of every native
action: ``bridge.step(action)`` returns the observation that followed it.  A longer chunk
would blind the program for the whole chunk, which is what makes a code policy possible
at all — and a chunk of one is what lets a tool stop the instant it has arrived.

The client has no receive timeout on this path and answers the server's pings from its
own thread, so ``get_action`` blocks for as long as a tool takes to produce its next
action.  Nothing is spent on waiting: the simulator is not stepped while a model thinks,
and no filler action is ever sent.

Every observation carries the three colour images, depth and its calibration (the reply
always asks for them), both arms' measured state and the end-effector poses.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Actions per reply.  One: the program sees the result of every native action, and the
#: client asks again after each one.
CHUNK = 1
#: The payload cameras and the keys this module gives them.
CAMERAS = ("head", "hand_left", "hand_right")
#: How long ``get_action`` may go without a tool producing anything before it answers a
#: hold action anyway.  Only a safety valve for a wedged program: the benchmark itself
#: waits as long as it takes, so this is not a thinking budget.
HOLD_AFTER_S = 600.0


class EpisodeOver(RuntimeError):
    """The benchmark client has ended the episode: no further observation will come."""


def _image(blob: dict[str, Any]) -> np.ndarray:
    """One JPEG payload frame as an (H, W, 3) uint8 RGB array."""

    from PIL import Image

    with Image.open(io.BytesIO(blob["image_data"])) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _depth(blob: dict[str, Any], scale_m: float) -> np.ndarray:
    """One depth payload frame as an (H, W) float array of metres along the optical axis."""

    height, width = int(blob["height"]), int(blob["width"])
    if blob.get("encoding") != "RAW_UINT16":
        raise ValueError(f"unsupported depth encoding {blob.get('encoding')!r}")
    raw = np.frombuffer(blob["image_data"], dtype=np.uint16, count=height * width)
    return raw.reshape(height, width).astype(np.float64) * scale_m


@dataclass(frozen=True)
class Camera:
    """One camera at the step the observation was taken.

    width / height: image size in pixels; pixel (row, col) has row 0 at the top.
    fx / fy / cx / cy: pinhole intrinsics at that resolution (None without calibration).
    optical_to_frame: 4x4 mapping an optical-frame point [x, y, z, 1] (metres, +x right,
        +y down, +z forward) into the robot frame; None without calibration.
    depth: (H, W) depth along the optical axis in metres, or None.
    """

    name: str
    width: int
    height: int
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None
    optical_to_frame: np.ndarray | None
    depth: np.ndarray | None


@dataclass(frozen=True)
class Observation:
    """One request from the benchmark, parsed.

    step: how many native actions the benchmark has taken so far (it is the count of
        requests it has made, all of them answered with one action).
    arm_joints: 14 joint angles, left arm then right (radians).
    gripper: the two grippers' measured openings, left then right, in the payload's
        convention: NEGATIVE when open (about -0.785) and 0 when closed on nothing.
        ``RobocoliseumRobot`` reports openings with the opposite sign; see its docstring.
    end_pose: {"base_link": {"left_arm": {"position", "orientation"}, ...}, "arm_base_link":
        ...} with quaternion order [x, y, z, w] as the payload gives it.
    instruction: the task instruction the benchmark generated (authoritative).
    """

    step: int
    instruction: str
    robot_type: str
    task_name: str
    episode_idx: int
    cameras: dict[str, Camera]
    arm_joints: list[float]
    gripper: list[float]
    waist: list[float]
    head: list[float]
    end_pose: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def camera(self, name: str) -> Camera:
        if name not in self.cameras:
            raise KeyError(f"camera {name!r} not in the observation; have {sorted(self.cameras)}")
        return self.cameras[name]

    def flange(self, arm: str, frame: str = "base_link") -> tuple[np.ndarray, np.ndarray]:
        """One arm's end-effector link pose as ``(position[3], quaternion xyzw[4])``."""

        poses = (self.end_pose or {}).get(frame) or {}
        pose = poses.get(f"{arm}_arm")
        if not pose:
            raise KeyError(f"no end pose for arm {arm!r} in frame {frame!r}")
        return (
            np.asarray(pose["position"], dtype=np.float64),
            np.asarray(pose["orientation"], dtype=np.float64),
        )


def parse_payload(params: dict[str, Any], step: int) -> Observation:
    """Turn one ``infer`` request's params into an :class:`Observation`."""

    images = params.get("images") or {}
    cameras: dict[str, Camera] = {}
    for name in CAMERAS:
        blob = images.get(name)
        if not isinstance(blob, dict) or blob.get("image_data") is None:
            continue
        colour = _image(blob)
        calibration = blob.get("calibration") or {}
        available = bool(calibration.get("available"))
        depth_blob = images.get(f"{name}_depth")
        depth = None
        if isinstance(depth_blob, dict) and depth_blob.get("image_data") is not None:
            scale = float(calibration.get("depth_scale_m") or 0.001)
            try:
                depth = _depth(depth_blob, scale)
            except (KeyError, ValueError) as exc:  # a malformed frame: work without it
                logger.warning("depth for %s unusable: %s", name, exc)
                depth = None
        cameras[name] = Camera(
            name=name,
            width=int(colour.shape[1]),
            height=int(colour.shape[0]),
            fx=float(calibration["fx"]) if available else None,
            fy=float(calibration["fy"]) if available else None,
            cx=float(calibration["cx"]) if available else None,
            cy=float(calibration["cy"]) if available else None,
            optical_to_frame=(
                np.asarray(calibration["optical_to_frame"], dtype=np.float64) if available else None
            ),
            depth=depth,
        )
    # The colour frames themselves are not part of the Camera (they are large and the
    # tools read them from :func:`colour_frames`); what a tool needs from a camera here
    # is its geometry.
    states = params.get("states") or {}
    return Observation(
        step=int(step),
        instruction=str(params.get("prompt") or ""),
        robot_type=str(params.get("robot_type") or ""),
        task_name=str(params.get("task_name") or ""),
        episode_idx=int(params.get("episode_idx") or 0),
        cameras=cameras,
        arm_joints=[float(v) for v in (states.get("arm_joint_states") or [])],
        gripper=[float(v) for v in (states.get("gripper_states") or [])],
        waist=[float(v) for v in (states.get("waist_joint_states") or [])],
        head=[float(v) for v in (states.get("head_joint_states") or [])],
        end_pose=states.get("end_pose") or {},
        raw={"images": images},
    )


def colour_frames(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """The three colour images of one request, as (H, W, 3) uint8 RGB arrays."""

    out: dict[str, np.ndarray] = {}
    for name, blob in (params.get("images") or {}).items():
        if not isinstance(blob, dict) or blob.get("image_data") is None:
            continue
        if name.endswith("_depth") or name not in CAMERAS:
            continue
        out[name] = _image(blob)
    return out


def hold_action(obs: Observation | None) -> dict[str, Any]:
    """One action that leaves the robot exactly as it is."""

    joints = list(obs.arm_joints) if obs is not None and len(obs.arm_joints) >= 14 else [0.0] * 14
    grips = list(obs.gripper) if obs is not None and len(obs.gripper) >= 2 else [0.0, 0.0]
    return {
        "kind": "JOINT_ABS",
        "left_arm": joints[:7],
        "right_arm": joints[7:14],
        "left_gripper": grips[0],
        "right_gripper": grips[1],
        "hold": True,
    }


class Bridge:
    """The benchmark client's episodes, one after another, met by the tools.

    One benchmark run plays several episodes over one connection.  They are delimited by
    the benchmark itself: when it ends an episode it makes one more request flagged
    ``episode_done`` (its scoring call), resets the scene, and the next ordinary request is
    the first of the next episode.  :attr:`episode` counts them from 1; :attr:`requests`
    counts the current episode's requests, so it restarts with each episode exactly as the
    benchmark's own step counter does.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._obs: Observation | None = None
        self._frames: dict[str, np.ndarray] = {}
        self._serial = 0
        self._action: dict[str, Any] | None = None
        self._finished = False
        self._over: dict[str, Any] | None = None
        self.requests = 0
        #: Episodes begun so far (1 = the first).
        self.episode = 0
        #: Episodes the benchmark has closed with its ``episode_done`` request.
        self.closed = 0
        self._between = True  # before the first episode, or after a closing request
        #: The gripper values of the last action served, held while the benchmark decides.
        self._last_grip: tuple[float, float] | None = None
        #: Open connections from this benchmark run, and when the last one closed.
        self.connections = 0
        self.disconnected_at: float | None = None
        self.peer: str | None = None
        #: Set when nobody will play any further episode: each one ends at its first request.
        self._draining = False
        #: Set when the run is over for good (its online session closed).
        self._ended = False

    # -- the client's side (the WebSocket handler calls these) --------------------------

    def update_obs(self, params: dict[str, Any]) -> Observation | None:
        """Take one request.  Returns its observation, or None for a closing request."""

        with self._cv:
            if params.get("episode_done"):
                # The benchmark has ended the episode and is scoring it.
                if not self._between:
                    self.closed = self.episode
                self._between = True
                if self._over is None:
                    self._over = {"episode_done": True}
                self._cv.notify_all()
                return None
            if self._between:
                # The first request of a new episode: nothing of the last one carries over.
                self.episode += 1
                self.requests = 0
                self._action = None
                self._finished = self._draining
                self._over = None
                self._last_grip = None
                self._between = False
            obs = parse_payload(params, self.requests)
            self._obs = obs
            self._frames = colour_frames(params)
            self.requests += 1
            self._serial += 1
            self._cv.notify_all()
            return obs

    def wait_episode(self, number: int, timeout_s: float, alive: Any = None) -> Observation | None:
        """The first observation of episode ``number``; None when the benchmark has gone.

        ``alive()`` false ends the wait early: a benchmark that exited plays no more
        episodes.
        """

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self.episode < number or self._between:
                if alive is not None and not alive():
                    return None
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 2.0))
            return self._obs

    def wait_closed(self, number: int, timeout_s: float, alive: Any = None) -> bool:
        """Wait for the benchmark's closing request of episode ``number``."""

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self.closed < number:
                if alive is not None and not alive():
                    return False
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 2.0))
            return True

    def _holding(self) -> dict[str, Any]:
        """Stay where the robot is, grippers as last commanded (a held object stays held)."""

        action = hold_action(self._obs)
        if self._last_grip is not None:
            action["left_gripper"], action["right_gripper"] = self._last_grip
        return action

    def get_action(self, timeout_s: float = HOLD_AFTER_S) -> dict[str, Any]:
        """The next native action, or a hold while there is none.

        A program that has finished does NOT end the episode: the reply is a hold, every
        step, until the benchmark ends the episode itself — at the check where the task
        passes, or when a stage's step budget runs out.  Ending it with ``episode_done``
        while the task checker is still running leaves that checker in the evaluator's
        slot, and the benchmark then silently never starts the next episode's checker
        (``ActionManager.start`` skips a slot that already holds an action of the same
        name): every later episode of the run would score zero.  Only a drained run (no
        program will play the rest) ends its episodes with ``episode_done``.
        """

        with self._cv:
            ready = self._cv.wait_for(lambda: self._action is not None or self._finished, timeout_s)
            if self._action is not None:
                action, self._action = self._action, None
                try:
                    self._last_grip = (
                        float(action["left_gripper"]),
                        float(action["right_gripper"]),
                    )
                except (KeyError, TypeError, ValueError):
                    pass
                self._cv.notify_all()
                return action
            if self._draining:
                return dict(self._holding(), **{"episode_done": True})
            if not ready:
                logger.warning("no tool produced an action in %.0f s; holding", timeout_s)
            return self._holding()

    def episode_end(self, info: dict[str, Any] | None = None) -> None:
        with self._cv:
            if self._over is None:
                self._over = dict(info or {})
            self._cv.notify_all()

    # -- the tools' side ----------------------------------------------------------------

    def first_obs(self, timeout_s: float = 600.0, alive: Any = None) -> Observation:
        """The first observation of the episode; ``alive()`` false means the client died."""

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._obs is None:
                if alive is not None and not alive():
                    raise RuntimeError("the benchmark exited before its first observation")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no observation from the benchmark in {timeout_s:.0f} s")
                self._cv.wait(min(left, 2.0))
            return self._obs

    def get_obs(self) -> Observation:
        with self._cv:
            if self._obs is None:
                raise RuntimeError("no observation yet")
            return self._obs

    def frames(self) -> dict[str, np.ndarray]:
        with self._cv:
            return dict(self._frames)

    @property
    def step(self) -> int:
        with self._cv:
            return self.requests

    def status(self) -> dict[str, Any]:
        with self._cv:
            status: dict[str, Any] = {"step": self.requests}
            if self._over is not None:
                status["over"] = True
                status["step_limit"] = self.requests
            return status

    @property
    def over(self) -> dict[str, Any] | None:
        with self._cv:
            return None if self._over is None else dict(self._over)

    def step_env(
        self, action: dict[str, Any], timeout_s: float = 600.0
    ) -> tuple[Observation, bool]:
        """Take one native action; returns the observation after it and whether it ended.

        Raises :class:`EpisodeOver` when the client had already ended the episode.
        """

        deadline = time.monotonic() + timeout_s
        with self._cv:
            if self._over is not None or self._finished:
                raise EpisodeOver("the episode is over")
            before = self._serial
            self._action = action
            self._cv.notify_all()
            while self._serial == before and self._over is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._action = None
                    raise TimeoutError(f"the benchmark did not answer in {timeout_s:.0f} s")
                self._cv.wait(min(left, 2.0))
            if self._serial == before:
                raise EpisodeOver("the episode ended on this action")
            assert self._obs is not None
            return self._obs, False

    def connected(self, grace_s: float = 0.0) -> bool:
        """Whether the benchmark is connected (or left less than ``grace_s`` ago).

        The client reconnects after a transient network error, so a run is only gone once
        it has stayed disconnected for a while.
        """

        with self._cv:
            if self._ended:
                return False
            if self.connections > 0:
                return True
            if self.disconnected_at is None:
                return self.requests == 0 and self.episode == 0  # not connected yet
            return time.monotonic() - self.disconnected_at < grace_s

    def _connection(self, delta: int) -> None:
        with self._cv:
            self.connections += delta
            if self.connections <= 0:
                self.connections = 0
                self.disconnected_at = time.monotonic()
            self._cv.notify_all()

    def drain(self) -> None:
        """No program will play another episode: end the current one and every later one."""

        with self._cv:
            self._draining = True
            self._finished = True
            self._cv.notify_all()

    def finish(self) -> None:
        """The program has no more actions: the client ends the episode and scores it."""

        with self._cv:
            self._finished = True
            self._cv.notify_all()

    def abandon(self) -> None:
        with self._cv:
            self._finished = True
            if self._over is None:
                self._over = {"success": False, "abandoned": True}
            self._cv.notify_all()

    def end_run(self, reason: str) -> None:
        """The run is over for good (the platform closed its session): the episode in play
        ends, its tools stop at their next action, and :meth:`connected` is false from now."""

        with self._cv:
            self._ended = True
            self._finished = True
            if self._over is None:
                self._over = {"run_ended": reason}
            self._cv.notify_all()


def envelope(action: dict[str, Any]) -> dict[str, Any]:
    """One native action as the reply the benchmark client parses."""

    left_arm = [float(v) for v in action["left_arm"]]
    right_arm = [float(v) for v in action["right_arm"]]
    left_grip = [float(action["left_gripper"])]
    right_grip = [float(action["right_gripper"])]
    body: dict[str, Any] = {
        "left_arm": {"kind": action.get("kind", "JOINT_ABS"), "values": [left_arm] * CHUNK},
        "right_arm": {"kind": action.get("kind", "JOINT_ABS"), "values": [right_arm] * CHUNK},
        "left_effector": [left_grip] * CHUNK,
        "right_effector": [right_grip] * CHUNK,
        # Depth and its calibration cost one round trip to switch on, so they are always
        # asked for: a program that measures must never find them missing.
        "need_depth": True,
    }
    if action.get("head") is not None:
        body["head"] = {"values": [[float(v) for v in action["head"]]] * CHUNK}
    if action.get("waist") is not None:
        body["waist"] = {
            "kind": "JOINT_ABS",
            "values": [[float(v) for v in action["waist"]]] * CHUNK,
        }
    if action.get("wheels") is not None:
        # [dx, dy, dyaw] relative to where the base is when the action arrives (POSE_DELTA,
        # anchored once per reply, and every reply here is one action).
        body["wheels"] = {
            "kind": "POSE_DELTA",
            "values": [[float(v) for v in action["wheels"]]] * CHUNK,
        }
    if action.get("episode_done"):
        body["episode_done"] = True
    return {"result": body}


class Hub:
    """One :class:`Bridge` per benchmark run, for servers several runs connect to.

    A run is identified by the address it connects FROM together with the port it
    connects TO — never by the connection itself: the benchmark client reconnects after a
    transient error, and a reconnect must land on the same run's bridge (its episode, its
    step counter, its pending action).  The address alone is not enough: a cluster
    masquerades its pods' traffic to the node's address (measured: two pods on one node
    arrive from the same IP), so concurrent workers each get their own port and the pair
    (address, port) names the run.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._bridges: dict[str, Bridge] = {}
        self._new: list[str] = []

    def bridge_for(self, peer: str) -> Bridge:
        with self._cv:
            bridge = self._bridges.get(peer)
            if bridge is None:
                bridge = Bridge()
                bridge.peer = peer
                self._bridges[peer] = bridge
                self._new.append(peer)
                logger.info("new benchmark run from %s", peer)
                self._cv.notify_all()
            return bridge

    def next_run(self, timeout_s: float) -> tuple[str, Bridge] | None:
        """The next run to connect (address, bridge), or None after ``timeout_s``."""

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while not self._new:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 2.0))
            peer = self._new.pop(0)
            return peer, self._bridges[peer]

    def runs(self) -> dict[str, Bridge]:
        with self._cv:
            return dict(self._bridges)


async def reply(bridge: Bridge, message: bytes) -> bytes | None:
    """The answer to one of the benchmark's requests, or None for one that wants none.

    ``bridge.get_action`` blocks until a tool has an action, so it runs off the event loop:
    the connection's pings are still answered meanwhile.
    """

    import msgpack

    req = msgpack.unpackb(message, raw=False)
    if req.get("method") != "infer":
        logger.warning("ignoring method %r", req.get("method"))
        return None
    params = req.get("params") or {}
    if bridge.update_obs(params) is None:
        # The benchmark's closing request of an episode: its client pops one action after
        # every request, so it still gets a (hold) answer.
        return msgpack.packb(envelope(dict(hold_action(bridge.get_obs()))))
    action = await asyncio.get_running_loop().run_in_executor(None, bridge.get_action)
    return msgpack.packb(envelope(action))


class Server:
    """GenieSim's ``infer`` WebSocket, served from a thread of this process.

    Anything that is not an :meth:`asyncio.AbstractEventLoop`-friendly task runs in the
    loop; the per-request handler blocks on the program's next action, so it is run in
    an executor and the loop stays free to answer the client's pings (it drops a
    connection whose ping goes unanswered for 60 s).
    """

    def __init__(self, target: Bridge | Hub, host: str, port: int) -> None:
        #: One run (a Bridge) or several told apart by address (a Hub).
        self.bridge = target
        self.host = str(host)
        self.port = int(port)
        self._loop = asyncio.new_event_loop()
        self._stop: asyncio.Event | None = None
        self._server: Any = None
        self._thread = threading.Thread(target=self._run, name="geniesim-bridge", daemon=True)
        self.error: BaseException | None = None
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(30.0)

    # -- the thread ---------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 - reported to the host
            self.error = exc
            self._ready.set()
        finally:
            with contextlib.suppress(Exception):
                self._loop.close()

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve as ws_serve
        except ImportError:  # websockets < 14
            from websockets.server import serve as ws_serve

        self._stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        # Each run blocks one executor thread while its program thinks: size the pool for
        # many concurrent runs rather than the default (CPU count + 4).
        from concurrent.futures import ThreadPoolExecutor

        loop.set_default_executor(ThreadPoolExecutor(max_workers=128, thread_name_prefix="bridge"))

        async def handler(ws: Any) -> None:
            peer = getattr(ws, "remote_address", None)
            target = self.bridge
            if isinstance(target, Hub):
                bridge = target.bridge_for(f"{peer[0] if peer else '?'}@{self.port}")
            else:
                bridge = target
            bridge._connection(+1)
            logger.info("benchmark connected: %s", peer)
            try:
                async for message in ws:
                    answer = await reply(bridge, message)
                    if answer is not None:
                        await ws.send(answer)
            except Exception as exc:  # noqa: BLE001 - one client, log and close
                logger.warning("benchmark connection ended: %s: %s", type(exc).__name__, exc)
            finally:
                bridge._connection(-1)
                logger.info("benchmark disconnected: %s", peer)

        async with ws_serve(
            handler, self.host, self.port, ping_interval=20, ping_timeout=60, max_size=None
        ) as server:
            self._server = server
            self._ready.set()
            logger.info("bridge serving on ws://%s:%d", self.host, self.port)
            await self._stop.wait()

    # -- shutdown -----------------------------------------------------------------------

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        try:
            self._loop.call_soon_threadsafe(lambda: self._stop and self._stop.set())
        except RuntimeError:
            pass
        self._thread.join(timeout=10)

    def alive(self) -> bool:
        return self._thread.is_alive() and self.error is None


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status a refused WebSocket handshake answered with, if that is what failed."""

    response = getattr(exc, "response", None)  # websockets >= 14: InvalidStatus
    status = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    return int(status) if status is not None else None


class TunnelClient:
    """The online platform's tunnel: this process dials the gateway, and the benchmark's
    requests come through it, from a thread of this process.

    RoboColiseum runs the benchmark on its own machines and relays its requests over one
    WebSocket that each agent process dials, ``<endpoint>?job=<uuid>&agent=<id>`` with
    ``Authorization: Bearer <job token>`` (the platform's ``challenge-run-agent`` guide).
    Text frames are JSON control frames: ``warmup`` (answered with ``ready``), ``queued``,
    ``session_open`` / ``session_close`` (one session per case, one at a time) and
    ``drain`` (no more sessions: close and do not redial); any other type is ignored.
    Binary frames carry a session's requests and the answers: a 4-byte big-endian length,
    the session id, then the payload, which is the very request a :class:`Server` gets.  An
    empty payload (the warmup's round-trip) is answered with an empty one.  A connection
    lost before ``drain`` is redialled with the same agent id: the gateway keeps the open
    session for :data:`RECONNECT_WINDOW_S`.

    Every session gets its own :class:`Bridge`; :meth:`next_session` hands them out.
    """

    #: How long the gateway keeps a session after an unclean close, for a redial.
    RECONNECT_WINDOW_S = 30.0
    #: Dials before giving up when no connection has carried a frame yet.
    FIRST_DIALS = 5

    def __init__(
        self, endpoint: str, job_uuid: str, token: str, agent_id: str | None = None
    ) -> None:
        import uuid
        from urllib.parse import urlencode

        self.agent_id = agent_id or str(uuid.uuid4())
        query = urlencode({"job": job_uuid, "agent": self.agent_id})
        self.url = f"{endpoint}{'&' if '?' in endpoint else '?'}{query}"
        self._token = token
        self._cv = threading.Condition()
        self._sessions: dict[str, Bridge] = {}
        self._new: list[str] = []
        self._connected = False
        #: The platform said there is nothing more to do.
        self.drained = False
        #: Why the tunnel gave up (None while it runs, and after a drain).
        self.error: str | None = None
        self._done = False
        self._loop = asyncio.new_event_loop()
        self._stop: asyncio.Event | None = None
        self._ws: Any = None
        self._thread = threading.Thread(target=self._run, name="geniesim-tunnel", daemon=True)
        self._thread.start()

    # -- the program's side -------------------------------------------------------------

    def next_session(self, timeout_s: float | None = None) -> tuple[str, Bridge] | None:
        """The next case's session (id, bridge); None once the tunnel is done (drained, or
        given up: see :attr:`error`) or after ``timeout_s``."""

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while not self._new:
                if self._done:
                    return None
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return None
                self._cv.wait(2.0 if left is None else min(left, 2.0))
            sid = self._new.pop(0)
            return sid, self._sessions.get(sid) or self._closed_bridge(sid)

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(lambda: self._stop and self._stop.set())
        self._thread.join(timeout=15)

    def alive(self) -> bool:
        return self._thread.is_alive()

    # -- sessions -----------------------------------------------------------------------

    @staticmethod
    def _closed_bridge(sid: str) -> Bridge:
        bridge = Bridge()
        bridge.peer = sid
        bridge.end_run("session closed before it was played")
        return bridge

    def _open(self, sid: str) -> Bridge:
        with self._cv:
            bridge = self._sessions.get(sid)
            if bridge is None:
                bridge = Bridge()
                bridge.peer = sid
                if self._connected:
                    bridge._connection(+1)
                self._sessions[sid] = bridge
                self._new.append(sid)
                logger.info("tunnel: session %s opened", sid)
                self._cv.notify_all()
            return bridge

    def _close(self, sid: str, reason: str) -> None:
        with self._cv:
            bridge = self._sessions.pop(sid, None)
        if bridge is not None:
            bridge.end_run(reason)
            logger.info("tunnel: session %s closed (%s)", sid, reason)

    def _link(self, up: bool) -> None:
        """The tunnel connected or dropped: every open session's run with it."""

        with self._cv:
            self._connected = up
            bridges = list(self._sessions.values())
        for bridge in bridges:
            bridge._connection(+1 if up else -1)

    def _finish(self, error: str | None) -> None:
        with self._cv:
            self.error = error
            self._done = True
            sids = list(self._sessions)
            self._cv.notify_all()
        for sid in sids:
            self._close(sid, "drain" if error is None else "tunnel lost")

    # -- the thread ---------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        error: str | None = None
        try:
            error = self._loop.run_until_complete(self._dial())
        except BaseException as exc:  # noqa: BLE001 - reported through error
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._finish(error)
            with contextlib.suppress(Exception):
                # Requests still being served, and the connection's reader, end here.
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            with contextlib.suppress(Exception):
                self._loop.close()

    async def _dial(self) -> str | None:
        """Dial, serve, redial; the reason it gave up, or None after a drain or a stop."""

        try:
            from websockets.asyncio.client import connect

            headers = "additional_headers"
        except ImportError:  # websockets < 14
            from websockets.client import connect

            headers = "extra_headers"
        self._stop = asyncio.Event()
        dials = 0
        lost_at: float | None = None  # when a connection that carried frames was lost
        while not self._stop.is_set():
            dials += 1
            carried = False
            try:
                async with connect(
                    self.url,
                    **{headers: {"Authorization": f"Bearer {self._token}"}},
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=None,
                ) as ws:
                    logger.info("tunnel: connected as agent %s", self.agent_id)
                    self._ws = ws
                    self._link(True)
                    try:
                        carried = await self._serve(ws)
                    finally:
                        self._ws = None
                        self._link(False)
            except Exception as exc:  # noqa: BLE001 - a failed dial is retried below
                if _http_status(exc) == 401:
                    return (
                        "the gateway refused the token (401): a wrong token, another "
                        "account's job, a job already over, or a redial too late"
                    )
                logger.warning("tunnel: %s: %s", type(exc).__name__, exc)
            if self.drained or self._stop.is_set():
                return None
            if carried:
                dials, lost_at = 0, time.monotonic()
            elif lost_at is None:
                # Closed at once after the handshake: over the job's parallelism, or the
                # gateway not ready yet.
                logger.warning("tunnel: closed before any frame (over the job's parallelism?)")
            if lost_at is not None and time.monotonic() - lost_at > self.RECONNECT_WINDOW_S:
                return f"the tunnel stayed lost for more than {self.RECONNECT_WINDOW_S:.0f} s"
            if lost_at is None and dials >= self.FIRST_DIALS:
                return f"no connection held after {dials} dials"
            await asyncio.sleep(min(5.0, 0.5 * 2 ** max(0, dials - 1)))
        return None

    async def _serve(self, ws: Any) -> bool:
        """Read the connection until it closes; whether it carried any frame."""

        from websockets.exceptions import ConnectionClosed

        carried = False
        pending: set[asyncio.Task[None]] = set()
        stop = asyncio.ensure_future(self._stop.wait()) if self._stop else None
        try:
            async for message in ws:
                carried = True
                if isinstance(message, str):
                    await self._control(ws, message)
                    if self.drained:
                        await ws.close()
                        break
                else:
                    # A request blocks until the program's next action: served beside the
                    # reading, so control frames (a session closing, a drain) still arrive.
                    task = asyncio.ensure_future(self._data(ws, bytes(message)))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                if stop is not None and stop.done():
                    await ws.close()
                    break
        except ConnectionClosed as exc:
            logger.warning("tunnel: connection closed: %s", exc)
        finally:
            if stop is not None:
                stop.cancel()
        return carried

    async def _control(self, ws: Any, text: str) -> None:
        try:
            frame = json.loads(text)
        except ValueError:
            logger.warning("tunnel: unreadable control frame %r", text[:200])
            return
        kind = frame.get("type") if isinstance(frame, dict) else None
        sid = str(frame.get("session_id") or "") if isinstance(frame, dict) else ""
        if kind == "warmup":
            # Nothing to load: the model is remote and the tools load on first use.
            await ws.send(json.dumps({"type": "ready"}))
        elif kind == "queued":
            logger.info("tunnel: queued (%s): waiting for a free simulator", frame.get("reason"))
        elif kind == "session_open" and sid:
            self._open(sid)
        elif kind == "session_close" and sid:
            self._close(sid, str(frame.get("reason") or "session closed"))
        elif kind == "drain":
            logger.info("tunnel: drain: the job has no more cases")
            self.drained = True
            with self._cv:
                sids = list(self._sessions)
            for open_sid in sids:
                self._close(open_sid, "drain")
        else:
            logger.info("tunnel: ignoring control frame %r", kind)

    async def _data(self, ws: Any, frame: bytes) -> None:
        from websockets.exceptions import ConnectionClosed

        if len(frame) < 4:
            logger.warning("tunnel: data frame of %d bytes dropped", len(frame))
            return
        (length,) = struct.unpack(">I", frame[:4])
        if 4 + length > len(frame):
            logger.warning("tunnel: data frame with an overrunning session id dropped")
            return
        sid = frame[4 : 4 + length].decode("utf-8", "replace")
        payload = frame[4 + length :]
        if not payload:
            answer: bytes | None = b""  # the warmup round-trip
        else:
            bridge = self._sessions.get(sid)
            if bridge is None:
                logger.warning("tunnel: request for session %s before its session_open", sid)
                bridge = self._open(sid)
            try:
                answer = await reply(bridge, payload)
            except Exception:  # noqa: BLE001 - one bad request must not end the tunnel
                logger.exception("tunnel: request of session %s failed", sid)
                return
        if answer is None:
            return
        with contextlib.suppress(ConnectionClosed):
            await ws.send(frame[: 4 + length] + answer)


__all__ = [
    "Bridge",
    "CHUNK",
    "CAMERAS",
    "Camera",
    "EpisodeOver",
    "Hub",
    "Observation",
    "Server",
    "TunnelClient",
    "colour_frames",
    "envelope",
    "hold_action",
    "parse_payload",
    "reply",
]
