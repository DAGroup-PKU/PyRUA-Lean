"""Check the repository for what must not be published: ``python tools/check_release.py [ROOT]``.

Every text file outside git-ignored places (``.git``, ``.cache``, ``runs``, built criteria) is
searched for absolute machine paths, IP addresses and the shapes of API keys and private keys.
Exits 1 and prints file:line for each hit.  Run it before every push; CI runs it as well.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".cache",
    "runs",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
}
PATTERNS = {
    "absolute path": re.compile(
        r"(?<![\w.])/(?:home|pfs|mnt|data|scratch|nfs|gpfs|lustre)/[\w.-]+"
    ),
    "IP address": re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "API key": re.compile(
        r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{20,}|nvapi-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
        r"ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|xox[abp]-[A-Za-z0-9-]{20,})"
    ),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer token": re.compile(r"Bearer [A-Za-z0-9._-]{30,}"),
}
#: Version-like dotted numbers that are not addresses.
NOT_IP = re.compile(r"\b(?:\d+\.){3}\d+\b(?=[\w-])|(?<=[\w=-])\b(?:\d+\.){3}\d+")


def files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(root).parts):
            yield path


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1])
    hits = 0
    for path in files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    if name == "IP address" and NOT_IP.search(
                        line[max(0, match.start() - 1) : match.end() + 1]
                    ):
                        continue
                    hits += 1
                    print(f"{path.relative_to(root)}:{number}: {name}: {match.group(0)[:60]}")
    print(f"{hits} finding(s)" if hits else "clean")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
