"""Loading generated files and running them under the watchdog."""

from __future__ import annotations

import time

import pytest

from pyrualean import LiberoRobot, PolicyLoadError, load_policy, run_policy


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_two_codebases_with_the_same_module_name_do_not_collide(tmp_path):
    first = _write(tmp_path / "gen1", "policy.py", "def run(robo):\n    return 'one'\n")
    second = _write(tmp_path / "gen2", "policy.py", "def run(robo):\n    return 'two'\n")
    assert load_policy(first)(None) == "one"
    assert load_policy(second)(None) == "two"
    assert load_policy(first)(None) == "one"


def test_missing_entrypoint_and_import_errors_are_load_errors(tmp_path):
    with pytest.raises(PolicyLoadError, match="callable 'run'"):
        load_policy(_write(tmp_path, "nope.py", "x = 1\n"))
    with pytest.raises(PolicyLoadError, match="importing"):
        load_policy(_write(tmp_path, "bad.py", "import definitely_missing_module\n"))
    with pytest.raises(PolicyLoadError):
        load_policy(tmp_path / "absent.py")


def test_run_policy_reports_completion_errors_and_finish(robo, backend):
    def ok(r):
        r.move_to([0.05, -0.10, 0.6])

    def boom(r):
        raise RuntimeError("boom")

    def finishes(r):
        r.move_to([0.05, -0.10, 0.6])
        r.pi0_pick("pick up the black bowl")
        r.set_gripper(1)
        r.move_to([0.05, 0.20, 0.55], gripper=1)
        r.release()
        r.move_to([0.0, 0.0, 0.68])  # raises EpisodeFinished("terminated")

    assert run_policy(ok, robo).status == "completed"
    outcome = run_policy(boom, robo)
    assert outcome.status == "error" and outcome.error_type == "RuntimeError"
    assert "boom" in (outcome.traceback or "")
    outcome = run_policy(finishes, robo)
    assert outcome.status == "finished" and outcome.reason == "terminated"
    assert backend.solved()


def test_run_policy_timeout_stops_a_busy_policy(backend):
    robo = LiberoRobot(backend)

    def forever(r):
        while True:
            r.set_gripper(1, steps=1)
            time.sleep(0.01)

    started = time.perf_counter()
    outcome = run_policy(forever, robo, timeout_s=0.3)
    assert outcome.status == "timeout"
    assert time.perf_counter() - started < 5
    assert backend.cancelled


def test_policy_except_exception_cannot_swallow_episode_finished(robo):
    def stubborn(r):
        r.move_to([0.05, -0.10, 0.6])
        r.pi0_pick("pick up the black bowl")
        r.set_gripper(1)
        r.move_to([0.05, 0.20, 0.55], gripper=1)
        r.release()
        for _ in range(3):
            try:
                r.move_to([0.0, 0.0, 0.68])
            except Exception:
                pass

    outcome = run_policy(stubborn, robo)
    assert outcome.status == "finished"
