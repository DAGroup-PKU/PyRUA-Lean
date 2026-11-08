"""Fill a config template's ``${NAME}`` / ``${NAME:-default}`` from the environment.

    python tools/render_config.py TEMPLATE.json OUT.json

The templates in ``configs/`` name machine paths and model names this way (see
``paths.env.example``); ``source tools/env.sh`` first.  A name that is unset and has no default
stops the rendering: a config never runs with an empty path.  Only JSON strings are filled;
API keys are never part of a config (a config names the variable that holds a key).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Unset(KeyError):
    """A placeholder whose variable is unset and has no default."""


def fill(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):

        def one(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if env.get(name):
                return env[name]
            if default is not None:
                return default
            raise Unset(name)

        return PLACEHOLDER.sub(one, value)
    if isinstance(value, list):
        return [fill(item, env) for item in value]
    if isinstance(value, dict):
        return {key: fill(item, env) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("template", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    try:
        config = fill(json.loads(args.template.read_text()), dict(os.environ))
    except Unset as exc:
        sys.stderr.write(f"{args.template}: ${{{exc.args[0]}}} is unset (paths.env?)\n")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
