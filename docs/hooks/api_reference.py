"""MkDocs hook: render a robot's API reference into the page that asks for it.

A page holds a line ``<!-- api-reference: <robot> -->`` (``primitives=no-vla`` may follow the robot's name; the
line may be indented, inside an admonition); at build time it becomes the reference
``pyrualean.prompt.api_reference`` generates from the live robot class, the text the agent reads.
The docs therefore cannot drift from the code, for the same reason the prompt cannot.
"""

import re
import sys
from pathlib import Path

# MkDocs restores sys.path once the hook is loaded: import the package now, while src/ is on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyrualean.prompt import api_reference  # noqa: E402
from pyrualean.robots import get_adapter  # noqa: E402

MARKER = re.compile(r"^([ \t]*)<!-- api-reference: ([\w-]+)(?: primitives=([\w-]+))? -->$", re.M)


def _render(match: re.Match) -> str:
    indent, robot, primitives = match.group(1), match.group(2), match.group(3) or "full"
    reference = api_reference(get_adapter(robot).robot_cls, primitives=primitives)
    block = f"```text\n{reference.rstrip()}\n```"
    return "\n".join(indent + line if line else line for line in block.splitlines())


def on_page_markdown(markdown, page, config, files):
    return MARKER.sub(_render, markdown)
