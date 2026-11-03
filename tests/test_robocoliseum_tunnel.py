"""The online tunnel against a scripted gateway that speaks the platform's protocol."""

from __future__ import annotations

import asyncio
import io
import json
import socket
import struct
import threading
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

msgpack = pytest.importorskip("msgpack")
pytest.importorskip("websockets")
pytest.importorskip("PIL")

from pyrualean.hosts import geniesim_bridge as bridge_mod  # noqa: E402
from pyrualean.hosts import geniesim_robocoliseum as host  # noqa: E402

JOB, TOKEN, AGENT = "job-1", "tok-1", "agent-1"


def _jpeg() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (8, 6), (90, 120, 30)).save(out, format="JPEG")
    return out.getvalue()


def request(done: bool = False, task: str = "pick_block_color") -> bytes:
    image = {"encoding": "JPEG", "image_data": _jpeg(), "height": 6, "width": 8}
    params = {
        "images": {"head": image, "hand_left": image, "hand_right": image},
        "states": {"arm_joint_states": [0.1] * 14, "gripper_states": [-0.785, -0.785]},
        "prompt": "pick up the red block",
        "robot_type": "G2_omnipicker",
        "task_name": task,
        "episode_done": done,
    }
    return msgpack.packb({"method": "infer", "params": params})


def frame(sid: str, payload: bytes) -> bytes:
    raw = sid.encode()
    return struct.pack(">I", len(raw)) + raw + payload


def unframe(data: bytes) -> tuple[str, bytes]:
    (length,) = struct.unpack(">I", data[:4])
    return data[4 : 4 + length].decode(), data[4 + length :]


Script = Callable[[Any, int], Awaitable[None]]


class Gateway:
    """A WebSocket server on a free port that runs ``script(ws, n)`` on its n-th connection."""

    def __init__(self, script: Script, refuse: bool = False) -> None:
        self.script, self.refuse = script, refuse
        self.dials: list[dict[str, Any]] = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.failure: BaseException | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._main(),), daemon=True
        )
        self._thread.start()
        assert self._ready.wait(10)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/api/challenge/tunnel"

    async def _main(self) -> None:
        from websockets.asyncio.server import serve

        def check(connection: Any, request: Any) -> Any:
            query = parse_qs(urlparse(request.path).query)
            self.dials.append({"query": query, "auth": request.headers.get("Authorization")})
            if self.refuse:
                return connection.respond(HTTPStatus.UNAUTHORIZED, "no\n")
            return None

        async def handler(ws: Any) -> None:
            try:
                await self.script(ws, len(self.dials))
            except BaseException as exc:  # noqa: BLE001 - surfaced by the test
                self.failure = exc
                raise

        self._stop = asyncio.Event()
        async with serve(handler, "127.0.0.1", self.port, process_request=check, max_size=None):
            self._ready.set()
            await self._stop.wait()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)


async def handshake(ws: Any) -> None:
    await ws.send(json.dumps({"type": "warmup"}))
    assert json.loads(await ws.recv()) == {"type": "ready"}


def program(bridge: bridge_mod.Bridge, actions: int) -> threading.Thread:
    """A program that takes ``actions`` hold actions in the episode the bridge is in, then
    finishes (the robot holds still from then on); a closed session ends it early."""

    def run() -> None:
        obs = bridge.wait_episode(1, 10)
        assert obs is not None
        try:
            for _ in range(actions):
                obs, _ = bridge.step_env(bridge_mod.hold_action(obs), timeout_s=10)
        except bridge_mod.EpisodeOver:
            pass
        bridge.finish()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_session_lifecycle_and_drain() -> None:
    async def script(ws: Any, n: int) -> None:
        await handshake(ws)
        await ws.send(frame("", b""))  # the warmup round-trip
        assert unframe(await ws.recv()) == ("", b"")
        await ws.send(b"\x00\x01")  # too short: dropped, the tunnel stays up
        await ws.send(json.dumps({"type": "some_future_frame"}))  # ignored
        await ws.send(json.dumps({"type": "session_open", "session_id": "s1"}))
        for _ in range(2):
            await ws.send(frame("s1", request()))
            sid, payload = unframe(await ws.recv())
            answer = msgpack.unpackb(payload, raw=False)
            assert sid == "s1"
            assert answer["result"]["left_arm"]["kind"] == "JOINT_ABS"
        await ws.send(json.dumps({"type": "session_close", "session_id": "s1"}))
        await ws.send(json.dumps({"type": "drain"}))
        await ws.wait_closed()

    gateway = Gateway(script)
    tunnel = bridge_mod.TunnelClient(gateway.url, JOB, TOKEN, agent_id=AGENT)
    try:
        opened = tunnel.next_session(10)
        assert opened is not None
        sid, bridge = opened
        assert sid == "s1"
        program(bridge, 2).join(10)
        assert tunnel.next_session(10) is None
        assert tunnel.drained and tunnel.error is None
        assert not bridge.connected(grace_s=1e9)
        assert bridge.over == {"run_ended": "session closed"}
        assert gateway.dials[0]["query"] == {"job": [JOB], "agent": [AGENT]}
        assert gateway.dials[0]["auth"] == f"Bearer {TOKEN}"
        assert gateway.failure is None
    finally:
        tunnel.stop()
        gateway.close()


def test_redial_keeps_the_open_session() -> None:
    async def script(ws: Any, n: int) -> None:
        if n == 1:
            await handshake(ws)
            await ws.send(json.dumps({"type": "session_open", "session_id": "s1"}))
            await ws.send(frame("s1", request()))
            await ws.recv()
            await ws.close(code=1011)  # unclean: the agent redials
            return
        await ws.send(frame("s1", request()))  # the same session, restored
        assert unframe(await ws.recv())[0] == "s1"
        await ws.send(json.dumps({"type": "drain"}))
        await ws.wait_closed()

    gateway = Gateway(script)
    tunnel = bridge_mod.TunnelClient(gateway.url, JOB, TOKEN, agent_id=AGENT)
    try:
        opened = tunnel.next_session(10)
        assert opened is not None
        _, bridge = opened
        program(bridge, 1).join(15)
        assert tunnel.next_session(15) is None
        assert tunnel.drained and tunnel.error is None
        assert [d["query"]["agent"] for d in gateway.dials] == [[AGENT], [AGENT]]
        assert bridge.requests == 2  # one bridge across both connections
        assert gateway.failure is None
    finally:
        tunnel.stop()
        gateway.close()


def test_a_refused_token_ends_the_tunnel() -> None:
    async def script(ws: Any, n: int) -> None:  # never reached
        raise AssertionError("refused handshakes have no connection")

    gateway = Gateway(script, refuse=True)
    tunnel = bridge_mod.TunnelClient(gateway.url, JOB, "wrong")
    try:
        assert tunnel.next_session(10) is None
        assert tunnel.error is not None and "401" in tunnel.error
        assert len(gateway.dials) == 1
    finally:
        tunnel.stop()
        gateway.close()


def test_serve_online_plays_each_episode_of_a_case(tmp_path: Path) -> None:
    async def script(ws: Any, n: int) -> None:
        await handshake(ws)
        await ws.send(json.dumps({"type": "session_open", "session_id": "case/7"}))
        for _ in range(3):  # the program's action, then holds once it has finished
            await ws.send(frame("case/7", request(task="sort_fruit")))
            await ws.recv()
        await ws.send(frame("case/7", request(done=True)))  # the benchmark ends the episode
        await ws.recv()
        await ws.send(json.dumps({"type": "session_close", "session_id": "case/7"}))
        await ws.send(json.dumps({"type": "drain"}))
        await ws.wait_closed()

    played: list[tuple[str, Path]] = []

    def play_one(cell: host.Cell, episode_dir: Path, session: host.Session) -> dict[str, Any]:
        toolkit = session.next_toolkit(episode_dir)
        toolkit.bridge.step_env(bridge_mod.hold_action(toolkit.bridge.get_obs()), timeout_s=10)
        toolkit.conclude()
        played.append((cell.task, episode_dir))
        return {"status": "finished", "env_steps": toolkit.env_steps()}

    gateway = Gateway(script)
    tunnel = bridge_mod.TunnelClient(gateway.url, JOB, TOKEN)
    try:
        assert host.serve_online(tunnel, tmp_path, play_one) == 1
        assert played == [("sort_fruit", tmp_path / "case_7" / "episode_001")]
        rows = [json.loads(line) for line in (tmp_path / "online.jsonl").read_text().splitlines()]
        assert [(r["session"], r["episode"], r["task"], r["status"]) for r in rows] == [
            ("case/7", 1, "sort_fruit", "finished")
        ]
        assert rows[0]["instruction"] == "pick up the red block"
        assert gateway.failure is None
    finally:
        tunnel.stop()
        gateway.close()
