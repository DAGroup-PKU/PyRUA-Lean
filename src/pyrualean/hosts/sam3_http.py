# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SAM3 as one local HTTP service, shared by every policy process on this machine.

:mod:`.sam3_local` loads the model inside the process that asks.  A GenieSim evaluation runs
one policy process per benchmark run at a time (one per cluster worker, or as many as an
online job's parallelism), and each copy of the 848M-parameter model holds about 10 GB of GPU
memory (measured: 10.4 GB on one RTX 5090), so they share one: the model runs once, behind
this server, and the policy processes hold a :class:`Sam3Http` client with the same
``segment`` signature and the same :class:`~.sam3_local.Segmented` answer.  The server runs
:class:`.sam3_local.LocalSam3` itself — the same calls and settings as the in-process path.
It loads that one file and nothing else of the package, so it may run in an environment of
its own (the ``sam3`` package pins ``numpy<2``).

  python src/pyrualean/hosts/sam3_http.py --port 8790 \\
      --checkpoint /path/to/sam3.pt          # prints "SAM3 ready" once loaded
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from typing import Any

import numpy as np

#: The default endpoint, on this machine.
DEFAULT_ENDPOINT = "http://127.0.0.1:8790"
#: One request waits at most this long: the model serialises requests, eight workers may queue.
TIMEOUT_S = 60.0


def _segmented_cls() -> Any:
    try:
        from .sam3_local import Segmented
    except ImportError:  # run as a script by the server
        from sam3_local import Segmented  # type: ignore[no-redef]
    return Segmented


def _pack_mask(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "shape": list(mask.shape),
        "bits": base64.b64encode(np.packbits(mask).tobytes()).decode(),
    }


def _unpack_mask(packed: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(v) for v in packed["shape"])
    bits = np.frombuffer(base64.b64decode(packed["bits"]), dtype=np.uint8)
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


class Sam3Http:
    """A client of the server below, answering like ``LocalSam3.segment``."""

    def __init__(self, endpoint: str | None = None, timeout_s: float = TIMEOUT_S) -> None:
        endpoint = endpoint or DEFAULT_ENDPOINT
        if "://" not in endpoint:
            endpoint = "http://" + endpoint
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.endpoint + "/health", timeout=5) as resp:
                return json.loads(resp.read()).get("ok") is True
        except (OSError, ValueError):
            return False

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
        top_k: int = 1,
    ) -> Any:
        """``LocalSam3.segment``'s answer: the best mask.  With ``top_k`` > 1 and a text
        prompt, ``self.last_candidates`` also holds up to that many instances above
        ``min_score``, best first — one name can match several objects (measured: "orange
        juice bottle in the shopping basket" scores the bottle on the SHELF highest; the
        words that place it are not what SAM3 ranks by)."""

        segmented = _segmented_cls()
        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        self.last_candidates = []
        body = json.dumps(
            {
                "top_k": int(top_k),
                "shape": list(pixels.shape),
                "pixels": base64.b64encode(pixels.tobytes()).decode(),
                "text_prompt": text_prompt,
                "point": None if point is None else [int(point[0]), int(point[1])],
                "min_score": float(min_score),
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/segment", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                answer = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 400:
                raise ValueError(detail) from exc
            return segmented(False, reason=f"SAM3 server error {exc.code}: {detail}")
        except (OSError, ValueError) as exc:
            return segmented(False, reason=f"SAM3 server unreachable at {self.endpoint}: {exc}")
        mask = _unpack_mask(answer["mask"]) if answer.get("mask") else None
        self.last_candidates = [
            {"score": c["score"], "box": c.get("box"), "mask": _unpack_mask(c["mask"])}
            for c in answer.get("candidates") or []
        ]
        return segmented(
            bool(answer.get("found")),
            score=answer.get("score"),
            box=answer.get("box"),
            mask=mask,
            reason=answer.get("reason"),
        )


def serve(port: int, checkpoint: str | None, host: str = "127.0.0.1") -> None:
    """Load the model and answer ``POST /segment`` until killed."""

    import importlib.util
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    # sam3_local needs nothing of the package around it: load the file itself, so this
    # environment does not need what the rest of pyrualean imports.
    path = Path(__file__).with_name("sam3_local.py")
    spec = importlib.util.spec_from_file_location("sam3_local", path)
    local = importlib.util.module_from_spec(spec)
    sys.modules["sam3_local"] = local
    assert spec.loader is not None
    spec.loader.exec_module(local)
    model = local.LocalSam3(checkpoint)
    served = {"n": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - quiet
            return

        def _reply(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/health":
                self._reply(200, {"ok": True, "served": served["n"]})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/segment":
                self._reply(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                shape = tuple(int(v) for v in req["shape"])
                pixels = np.frombuffer(base64.b64decode(req["pixels"]), dtype=np.uint8).reshape(
                    shape
                )
                out = model.segment(
                    pixels,
                    text_prompt=req.get("text_prompt"),
                    point=req.get("point"),
                    min_score=float(req.get("min_score", 0.2)),
                )
            except (ValueError, KeyError, TypeError) as exc:
                self._reply(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - reported to the client
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            candidates = []
            top_k = int(req.get("top_k") or 1)
            if top_k > 1 and req.get("text_prompt"):
                candidates = all_instances(
                    model, pixels, req["text_prompt"], float(req.get("min_score", 0.2)), top_k
                )
            with lock:
                served["n"] += 1
            self._reply(
                200,
                {
                    "candidates": candidates,
                    "found": bool(out.found),
                    "score": None if out.score is None else float(out.score),
                    "box": out.box,
                    "reason": out.reason,
                    "mask": None if out.mask is None else _pack_mask(out.mask),
                },
            )

    def all_instances(
        model: Any, pixels: np.ndarray, text: str, min_score: float, top_k: int
    ) -> list[dict[str, Any]]:
        """Every instance the text prompt scores at least ``min_score``, best first — the
        same call ``LocalSam3.segment`` makes, without keeping only the top one."""

        import warnings

        with model._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = model._state_for(np.ascontiguousarray(pixels[..., :3].astype(np.uint8)))
            with model._autocast():
                out = model._processor.set_text_prompt(prompt=text.strip(), state=state)
        masks = local._array(out.get("masks")) if out.get("masks") is not None else None
        scores = local._array(out.get("scores")) if out.get("scores") is not None else None
        boxes = out.get("boxes")
        boxes = local._array(boxes) if boxes is not None else None
        if masks is None or scores is None:
            return []
        if masks.ndim == 4:
            masks = masks[:, 0]
        scores = scores.reshape(-1)
        found = []
        for index in np.argsort(-scores)[:top_k]:
            if float(scores[index]) < min_score:
                break
            mask = np.asarray(masks[index]) > 0
            if not mask.any():
                continue
            box = None
            if boxes is not None and boxes.ndim >= 2 and index < boxes.shape[0]:
                box = [float(v) for v in boxes[index].reshape(-1)[:4]]
            found.append({"score": float(scores[index]), "box": box, "mask": _pack_mask(mask)})
        return found

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SAM3 ready on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--checkpoint", default=None, help="else $SAM3_CHECKPOINT_PATH")
    args = parser.parse_args()
    serve(args.port, args.checkpoint, args.host)
