"""Environment settings: ``PYRUALEAN_*`` variables, read by the package."""

from __future__ import annotations

from pyrualean._env import setting, setting_var
from pyrualean.generate import provider_overrides


def test_setting_reads_the_pyrualean_variable_else_the_default(monkeypatch):
    monkeypatch.delenv("PYRUALEAN_PRIMITIVES", raising=False)
    assert setting("PRIMITIVES", "full") == "full"
    assert setting_var("PRIMITIVES") == "PYRUALEAN_PRIMITIVES"
    monkeypatch.setenv("PYRUALEAN_PRIMITIVES", "no-vla")
    assert setting("PRIMITIVES", "full") == "no-vla"
    monkeypatch.setenv("PYRUALEAN_PRIMITIVES", "")  # set but empty still counts as set
    assert setting("PRIMITIVES", "no-vla") == ""


def test_codex_retries_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("PYRUALEAN_CODEX_RETRIES", "7")
    joined = " ".join(provider_overrides("https://gw.example.com", "KEY"))
    assert "request_max_retries=7" in joined and "stream_max_retries=7" in joined
