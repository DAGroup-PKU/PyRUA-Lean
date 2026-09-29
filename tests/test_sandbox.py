"""The policy sandbox: guarded imports and open, removed builtins, static audit."""

from __future__ import annotations

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.runner import PolicyLoadError, load_policy
from pyrualean.sandbox import ALLOWED_MODULES, audit_source, make_builtins


@pytest.fixture
def cells(robo, tmp_path):
    return ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", feedback="pure")


def test_cells_cannot_import_os_or_escape_the_workspace(cells, tmp_path):
    denied = cells.execute_tool("python", {"code": "import os\nprint(os.getcwd())"})
    assert "not available inside the policy sandbox" in denied.result["exception"]
    outside = cells.execute_tool("python", {"code": "open('../secret.txt', 'w')"})
    assert "only open files inside its workspace" in outside.result["exception"]
    absolute = cells.execute_tool("python", {"code": "open('/etc/hostname').read()"})
    assert "only open files inside its workspace" in absolute.result["exception"]
    inside = cells.execute_tool(
        "python", {"code": "open('note.txt', 'w').write('x'); print(open('note.txt').read())"}
    )
    assert inside.result["stdout"] == "x\n" and (cells.workspace / "note.txt").read_text() == "x"
    allowed = cells.execute_tool(
        "python", {"code": "import json, re, collections\nprint(json.dumps({'a': 1}))"}
    )
    assert allowed.result["stdout"] == '{"a": 1}\n' and "exception" not in allowed.result
    no_eval = cells.execute_tool("python", {"code": "eval('1+1')"})
    assert "NameError" in no_eval.result["exception"]
    exited = cells.execute_tool("python", {"code": "exit()"})
    assert "NameError" in exited.result["exception"]
    assert cells.robo.state().step >= 0  # the toolkit is still alive


def test_audit_flags_reach_for_host_handles_and_paths(cells):
    probe = cells.execute_tool(
        "python",
        {"code": "print(dir(robo)); x = robo._backend; y = robo.backend; import os"},
    )
    assert "exception" in probe.result
    flags = cells.turns[-1]["sandbox_flags"]
    assert "probe:dir" in flags and "private-attr:_backend" in flags
    assert "host-attr:backend" in flags and "import:os" in flags
    assert cells.sandbox_flags == [{"turn": 0, "flags": flags}]
    clean = cells.execute_tool("python", {"code": "m = robo.state(); print(m.eef_pos)"})
    assert cells.turns[-1]["sandbox_flags"] == [] and "exception" not in clean.result
    assert audit_source("open('../states.json')") == ["path:../states.json"]
    assert set(audit_source("robo.__class__.__subclasses__()")) == {
        "escape:__class__",
        "escape:__subclasses__",
    }
    assert audit_source("def f(:") == []


def test_program_arm_is_sandboxed_but_workspace_imports_work(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws")
    denied = kit.execute_tool(
        "run_program", {"code": "import subprocess\ndef run(robo):\n    pass\n"}
    )
    assert "not available inside the policy sandbox" in denied.result["exception"]
    assert kit.turns[-1]["sandbox_flags"] == ["import:subprocess"]
    (kit.workspace / "sbhelper.py").write_text("VALUE = 3\n")
    ok = kit.execute_tool(
        "run_program", {"code": "from sbhelper import VALUE\ndef run(robo):\n    print(VALUE)\n"}
    )
    assert ok.result["stdout"] == "3\n" and kit.turns[-1]["sandbox_flags"] == []


def test_load_policy_with_sandbox_builtins(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    policy = tmp_path / "policy.py"
    policy.write_text("import os\ndef run(robo):\n    pass\n")
    with pytest.raises(PolicyLoadError, match="policy sandbox"):
        load_policy(policy, builtins=make_builtins(ws))
    policy.write_text("import math\ndef run(robo):\n    return math.pi\n")
    assert load_policy(policy, builtins=make_builtins(ws))(None) == pytest.approx(3.14159, abs=1e-4)
    assert "numpy" in ALLOWED_MODULES and "os" not in ALLOWED_MODULES


def test_cells_refuse_python_internals_before_running(cells):
    from pyrualean.sandbox import BLOCKED_NAMES, blocked_names

    patched = cells.execute_tool(
        "python", {"code": "MARK = 1\nrobo.state.__func__.__globals__['x'] = 1"}
    )
    text = patched.result["exception"]
    assert text.startswith("SandboxError") and "__func__" in text and "__globals__" in text
    assert "stdout" not in patched.result  # the cell never ran
    flags = cells.turns[-1]["sandbox_flags"]
    assert "escape:__globals__" in flags and "escape:__func__" in flags
    later = cells.execute_tool("python", {"code": "print(MARK)"})
    assert "NameError" in later.result["exception"]
    by_string = cells.execute_tool("python", {"code": "getattr(robo.state, '__globals__')"})
    assert "SandboxError" in by_string.result["exception"]
    assert "escape:__globals__" in cells.turns[-1]["sandbox_flags"]
    assert blocked_names("def f(:") == [] and blocked_names("x = 1") == []
    assert blocked_names("a.__code__; b = '__code__'; __self__") == ["__code__", "__self__"]
    assert "globals" not in BLOCKED_NAMES  # `'x' in globals()` reads the cell's own namespace
    assert cells.robo.state().step >= 0


def test_program_arm_refuses_python_internals_before_import(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws")
    refused = kit.execute_tool(
        "run_program",
        {"code": "print('module ran')\ndef run(robo):\n    type(robo).__subclasses__()\n"},
    )
    assert "SandboxError" in refused.result["exception"] and "stdout" not in refused.result
    assert kit.turns[-1]["sandbox_flags"] == ["escape:__subclasses__"]
