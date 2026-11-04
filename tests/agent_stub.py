"""Loopback HTTP stub for hermetic tests of the agent chat client (no real provider)."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK_SCHEMA_PATH = ROOT / "tests/fixtures/task_expert_failure_review_v1.json"
# A fake, deliberately non key-shaped test credential (secret scanners must not fire on it).
FAKE_KEY = "fake-test-credential-0123456789abcdef"

VALID_PROPOSAL: dict[str, Any] = {
    "diagnosis": [
        {
            "failure_mode": "right gripper closes before bilateral contact",
            "supporting_evidence": "v2 slot 1: inner/outer-wall contact check failed",
            "basis": "observed_in_evidence",
            "confidence": "medium",
        }
    ],
    "recommended_changes": [
        {
            "change": "wait for bilateral contact before commanding close",
            "addresses": "right gripper closes before bilateral contact",
            "rationale": "the failed grasp closed with one finger off the wall",
            "risk": "slower cycle time",
        }
    ],
    "invariants": ["no object pose writes or hidden attachment"],
    "falsification_tests": [
        {
            "hypothesis": "early close causes the right-arm grasp failure",
            "procedure": "run seeds 0-9 with the contact wait and log finger forces",
            "observable": "bilateral contact force at close command",
            "refuted_if": "grasp still fails with bilateral contact confirmed",
        }
    ],
}


def load_task_schema() -> dict[str, Any]:
    return json.loads(TASK_SCHEMA_PATH.read_text(encoding="utf-8"))


def chat_body(
    content: str,
    *,
    response_id: str = "chatcmpl-test-1",
    model: str = "gpt-test-2026",
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


@dataclass
class Reply:
    status: int = 200
    body: Any = None  # dict/list -> JSON; str/bytes -> raw
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    content_type: str | None = None


def ok_reply(proposal: Any = None, **kwargs: Any) -> Reply:
    content = json.dumps(VALID_PROPOSAL if proposal is None else proposal)
    return Reply(200, chat_body(content, **kwargs))


class StubServer:
    """Serves scripted replies in arrival order and records every request."""

    def __init__(self, script: list[Reply] | None = None) -> None:
        self.script = list(script or [])
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                with server._lock:
                    server.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "body": body,
                        }
                    )
                    reply = server.script.pop(0) if server.script else Reply(
                        500, {"error": {"message": "stub script exhausted"}}
                    )
                if reply.delay:
                    time.sleep(reply.delay)
                if isinstance(reply.body, (dict, list)):
                    payload = json.dumps(reply.body).encode("utf-8")
                    content_type = reply.content_type or "application/json"
                elif isinstance(reply.body, str):
                    payload = reply.body.encode("utf-8")
                    content_type = reply.content_type or "text/plain; charset=utf-8"
                else:
                    payload = reply.body or b""
                    content_type = reply.content_type or "application/octet-stream"
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    for name, value in reply.headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up (timeout test)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)


def files_containing(root: Path, needle: str) -> list[Path]:
    """Every file under ``root`` whose bytes contain ``needle``."""

    hits = []
    for path in root.rglob("*"):
        if path.is_file() and needle.encode("utf-8") in path.read_bytes():
            hits.append(path)
    return hits
