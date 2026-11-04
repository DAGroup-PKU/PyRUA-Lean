"""Wall-clock bounded carrier upload and remote byte verification."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


class UploadError(RuntimeError):
    pass


def curl(args, url, remaining):
    if remaining <= 0:
        raise UploadError("upload exchange deadline")
    # URLs/proxy credentials stay in stdin, not process arguments or logs.
    config = "url = " + json.dumps(url) + "\n"
    proxy = os.environ.get("SEEDANCE_API_PROXY")
    if proxy:
        config += "proxy = " + json.dumps(proxy) + "\n"
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--connect-timeout",
        "20",
        "--max-time",
        str(remaining),
        "--config",
        "-",
    ]
    if not proxy:
        command += ["--noproxy", "*"]
    try:
        result = subprocess.run(
            command + args, input=config.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=remaining + 2
        )
    except subprocess.TimeoutExpired:
        raise UploadError("carrier transfer wall-clock deadline") from None
    if result.returncode:
        raise UploadError(f"carrier transfer failed: curl code {result.returncode}")
    return result.stdout


def exchange(path, expected, folder, host, timeout=90):
    """Upload to a tempfile.org-compatible host (POST /api/upload/local, GET /<id>/download)."""
    deadline = time.monotonic() + timeout
    host = host.rstrip("/")
    response = curl(
        ["--request", "POST", "--form", f"files=@{path};type=video/mp4", "--form", "expiryHours=24"],
        host + "/api/upload/local",
        deadline - time.monotonic(),
    )
    try:
        identifier = json.loads(response)["files"][0]["id"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise UploadError("malformed carrier upload receipt") from None
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise UploadError("invalid carrier identifier")
    url = f"{host}/{identifier}/download"
    partial = Path(folder) / "upload-verify.partial"
    try:
        curl(["--output", str(partial)], url, deadline - time.monotonic())
        digest = hashlib.sha256()
        size = 0
        with partial.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != expected:
            raise UploadError("remote carrier hash mismatch")
        return url, size
    finally:
        partial.unlink(missing_ok=True)
