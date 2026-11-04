"""Each job has one entry point, in tools/, so a run pinned to a checkout keeps running that checkout's code."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def tool(cwd, *args, **env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('DATAFARM_') and k != 'PYTHONPATH'}
    return subprocess.run([sys.executable, str(ROOT / 'tools' / args[0]), *args[1:]], capture_output=True, text=True,
                          cwd=cwd, env={**clean, **env}, timeout=120)


def test_ledger_and_expert_tools_run_from_their_own_checkout(tmp_path):
    r = tool(tmp_path, 'datafarm_runs.py', '--root', str(tmp_path / 'ledger'), 'status', '--no-probe')
    assert r.returncode == 0, r.stderr
    r = tool(tmp_path, 'expert_harness.py', '--help')
    assert r.returncode == 0 and 'tools/expert_harness.py' in r.stdout, r.stderr


def test_ledger_tool_has_no_site_scheduler_defaults():
    text = (ROOT / 'tools/datafarm_runs.py').read_text()
    assert '.scheduler' not in text and 'DATAFARM_SCHEDULER_BASE' not in text


def test_no_installed_alias_for_a_checkout_bound_tool():
    scripts = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['scripts']
    assert 'datafarm-expert' not in scripts
    from datafarm.cli import make_parser
    assert 'runs' not in make_parser()._subparsers._group_actions[0].choices


def test_tool_names_use_underscores():
    assert not [p.name for p in (ROOT / 'tools').iterdir() if p.is_file() and ('-' in p.name or not p.suffix)]
