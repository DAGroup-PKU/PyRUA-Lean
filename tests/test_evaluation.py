"""Episode accounting: crashes are failures, steps are comparable, pairs are strict."""

from __future__ import annotations

import json

from pyrualean.evaluation import (
    EpisodeMetrics,
    compare_metric_files,
    env_steps_from_states,
    load_episode_result,
    parse_rpent_run,
    summarize,
)


def _result(tmp_path, name, **overrides):
    payload = {
        "schema": "pyrualean-episode-v2",
        "system": "pyrualean",
        "backend": "libero",
        "suite": "libero_object_swap",
        "task": 2,
        "seed": 0,
        "environment_success": True,
        "status": "finished",
        "calls": 9,
        "stateful_calls": 6,
        "env_steps": 180,
        "policy_wall_s": 40.0,
        "startup_s": 120.0,
        "error": None,
        "notes": [],
    }
    payload.update(overrides)
    directory = tmp_path / name
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(payload))
    return directory


def test_crashed_and_timed_out_episodes_count_as_failures(tmp_path):
    runs = [
        load_episode_result(_result(tmp_path, "ok")),
        load_episode_result(
            _result(
                tmp_path,
                "crash",
                seed=1,
                environment_success=False,
                status="error",
                error={"type": "KeyError", "message": "x"},
            )
        ),
        load_episode_result(
            _result(tmp_path, "timeout", seed=2, environment_success=False, status="timeout")
        ),
    ]
    summary = summarize(runs)
    assert summary["native_sr"] == 1 / 3
    assert summary["native_sr_denominator"] == 3
    assert summary["unknown_outcome_episodes"] == 0
    assert summary["statuses"] == {"finished": 1, "error": 1, "timeout": 1}
    assert "KeyError: x" in runs[1].notes


def test_env_steps_from_states_counts_chunks_and_servo_steps():
    steps = [
        {"step_idx": 0},
        # converged servo: RPent records the final check as a step (12 -> 11)
        {"command": {"action": "move_to"}, "result": {"steps_used": 12, "final_dist_m": 0.009}},
        # exhausted budget: every recorded step was executed
        {
            "command": {"action": "move_to", "max_steps": 5},
            "result": {"steps_used": 5, "final_dist_m": 0.2},
        },
        {"command": {"action": "pi0_pick"}, "result": {"chunks_used": 3}},
        {"command": {"action": "set_gripper", "steps": 8}, "result": {"steps": 8}},
        {"command": {"action": "release"}, "result": {"steps_used": 4}},
    ]
    assert env_steps_from_states(steps) == 11 + 5 + 15 + 8 + 4
    assert env_steps_from_states(steps, chunk_steps=10) == 11 + 5 + 30 + 8 + 4


def test_parse_rpent_run_uses_native_termination_not_agent_claim(tmp_path):
    (tmp_path / "transcript_object_swap_t2_s0.json").write_text(
        json.dumps(
            {
                "suite": "libero_object_swap",
                "task": 2,
                "seed": 0,
                "finish": {"status": "success", "summary": "done"},
                "stats": {
                    "tool_calls": 44,
                    "turns_used": 18,
                    "elapsed_s": 286.3,
                    "total_input_tokens": 1390523,
                    "total_cached_input_tokens": 1300608,
                    "total_output_tokens": 4553,
                    "total_reasoning_output_tokens": 568,
                },
            }
        )
    )
    (tmp_path / "states.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"step_idx": 0, "terminated": False},
                    {
                        "step_idx": 1,
                        "terminated": False,
                        "command": {"action": "move_to"},
                        "result": {"steps_used": 20},
                    },
                    {
                        "step_idx": 2,
                        "terminated": False,
                        "command": {"action": "pi0_pick"},
                        "result": {"chunks_used": 2},
                    },
                ]
            }
        )
    )
    run = parse_rpent_run(tmp_path)
    assert run.system == "rpent" and run.task == "libero_object_swap:2" and run.seed == 0
    assert run.environment_success is False
    assert run.status == "success"
    assert any("claimed success" in note for note in run.notes)
    assert run.calls == 44 and run.stateful_calls == 2 and run.env_steps == 30
    assert run.total_tokens == 1390523 + 4553


def test_parse_rpent_run_without_finish_is_no_finish(tmp_path):
    (tmp_path / "transcript_x.json").write_text(
        json.dumps(
            {
                "suite": "libero_object_swap",
                "task": 2,
                "seed": 1,
                "finish": None,
                "stats": {"tool_calls": 46, "turns_used": 11, "elapsed_s": 534.1},
            }
        )
    )
    (tmp_path / "states.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"step_idx": 0, "terminated": False},
                    {
                        "step_idx": 1,
                        "terminated": False,
                        "command": {"action": "release"},
                        "result": {"steps_used": 20},
                    },
                ]
            }
        )
    )
    run = parse_rpent_run(tmp_path)
    assert run.status == "no_finish" and run.environment_success is False
    assert any("without a finish" in note for note in run.notes)


def test_compare_pairs_strictly_by_backend_task_seed(tmp_path):
    ok = _result(tmp_path, "code", task=2, seed=0)
    rpent = tmp_path / "rpent.json"
    rpent.write_text(
        json.dumps(
            {
                "runs": [
                    EpisodeMetrics(
                        system="rpent",
                        backend="libero",
                        task="libero_object_swap:2",
                        seed=0,
                        environment_success=False,
                        calls=44,
                        env_steps=300,
                        wall_time_s=286.3,
                    ).to_dict(),
                    EpisodeMetrics(
                        system="rpent",
                        backend="libero",
                        task="libero_object_swap:2",
                        seed=7,
                        environment_success=True,
                    ).to_dict(),
                ]
            }
        )
    )
    report = compare_metric_files([ok, rpent])
    assert report["pairing"]["valid"] is False  # seed 7 unmatched
    assert report["pairing"]["matched_episodes"] == 1
    pair = report["pairs"][0]
    assert pair["delta"]["native_success_delta"] == 1
    assert pair["delta"]["env_steps"] == 180 - 300
    assert report["pairing"]["unmatched_keys"][0]["seed"] == 7


def test_env_steps_from_states_trusts_a_native_action_counter():
    steps = [
        {"command": {"action": "reset"}, "state": {"episode_status": {"take_action_cnt": 0}}},
        {
            "command": {"action": "lingbot_act"},
            "state": {"episode_status": {"take_action_cnt": 100}},
        },
        {
            "command": {"action": "move_to"},
            "result": {"steps_used": 7},
            "state": {"episode_status": {"take_action_cnt": 108}},
        },
    ]
    assert env_steps_from_states(steps) == 108


def test_parse_rpent_run_reads_robotwin_transcripts(tmp_path):
    (tmp_path / "transcript_robotwin_beat_block_hammer_s100000.json").write_text(
        json.dumps(
            {
                "env": "robotwin",
                "task_name": "beat_block_hammer",
                "requested_seed": 100000,
                "task_config": "demo_randomized",
                "elapsed_s": 230.0,
                "finish": {"status": "success", "summary": "done"},
                "stats": {
                    "tool_calls": 8,
                    "turns_used": 4,
                    "elapsed_s": 230.0,
                    "total_input_tokens": 184000,
                    "total_output_tokens": 616,
                },
            }
        )
    )
    (tmp_path / "states.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_idx": 0,
                        "command": {"action": "reset"},
                        "terminated": False,
                        "state": {"episode_status": {"take_action_cnt": 0}},
                    },
                    {
                        "step_idx": 1,
                        "command": {"action": "lingbot_act"},
                        "terminated": True,
                        "state": {"episode_status": {"take_action_cnt": 108, "eval_success": True}},
                    },
                ]
            }
        )
    )
    rec = parse_rpent_run(tmp_path)
    assert rec.backend == "robotwin" and rec.seed == 100000
    assert rec.task == "demo_randomized:beat_block_hammer"
    assert rec.environment_success is True and rec.env_steps == 108 and rec.stateful_calls == 1


def test_env_steps_from_states_rebuilds_robocasa_runs():
    vla_only = [
        {"step_idx": 0, "extras": {"vla_desync": True}},
        {
            "command": {"action": "rldx_skill"},
            "result": {"steps_applied": 208, "status": "success"},
            "extras": {"vla_desync": False},
        },
    ]
    assert env_steps_from_states(vla_only) == 208
    scripted = [
        {"step_idx": 0, "extras": {"vla_desync": True}},
        {"command": {"action": "move_to"}, "result": {"steps": 20}, "extras": {}},
        {"command": {"action": "navigate_to"}, "result": {"steps": 30}, "extras": {}},
        {"command": {"action": "move_to"}, "result": {"steps": 10}, "extras": {}},
        {"command": {"action": "release", "steps": 8}, "result": {}, "extras": {}},
        {"command": {"action": "move_to"}, "result": {"error": "planning failed"}, "extras": {}},
    ]
    # jacobian (9) + 20, heading (6) + 30, jacobian again (9) + 10, release 8; the error is skipped
    assert env_steps_from_states(scripted) == 9 + 20 + 6 + 30 + 9 + 10 + 8


def test_parse_rpent_run_normalises_claude_sdk_token_stats(tmp_path):
    (tmp_path / "transcript_goal_swap_t7_s0.json").write_text(
        json.dumps(
            {
                "suite": "libero_goal_swap",
                "task": 7,
                "seed": 0,
                "finish": {"status": "success", "summary": "done"},
                "stats": {
                    "backend": "claude_agent_sdk",
                    "tool_calls": 6,
                    "turns_used": 4,
                    "elapsed_s": 120.0,
                    "total_input_tokens": 2000,
                    "total_cache_read_input_tokens": 90000,
                    "total_cache_creation_input_tokens": 8000,
                    "total_output_tokens": 500,
                },
            }
        )
    )
    (tmp_path / "states.json").write_text(
        json.dumps(
            {"steps": [{"step_idx": 0, "terminated": False}, {"step_idx": 1, "terminated": True}]}
        )
    )
    rec = parse_rpent_run(tmp_path)
    assert rec.input_tokens == 2000 + 90000 + 8000 and rec.cached_input_tokens == 90000
    assert rec.output_tokens == 500 and rec.total_tokens == 100500


def test_claude_stream_usage_counts_each_model_request_once(tmp_path):
    from pyrualean.evaluation.rpent import claude_stream_usage

    usage_a = {"input_tokens": 2, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10}
    usage_b = {"input_tokens": 2, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 0}
    tool_result = {"content": [{"type": "text", "text": '{"terminated": false}'}]}
    lines = [
        {"type": "AssistantMessage", "message_id": "a", "usage": {**usage_a, "output_tokens": 3}},
        {"type": "AssistantMessage", "message_id": "a", "usage": {**usage_a, "output_tokens": 9}},
        {"type": "UserMessage", "content": [tool_result]},
        {"type": "AssistantMessage", "message_id": "b", "usage": {**usage_b, "output_tokens": 5}},
        {
            "type": "AssistantMessage",
            "message_id": "sub",
            "parent_tool_use_id": "x",
            "usage": {"input_tokens": 999},
        },
    ]
    (tmp_path / "claude_goal_swap_t7_s0.txt.stream.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    usage = claude_stream_usage(tmp_path)
    assert usage == {
        "input": 4,
        "cache_read": 300,
        "cache_creation": 10,
        "output": 14,
        "requests": 2,
        "exact_output": 0,
    }
    (tmp_path / "transcript_goal_swap_t7_s0.json").write_text(
        json.dumps(
            {
                "suite": "libero_goal_swap",
                "task": 7,
                "seed": 0,
                "finish": {"status": "success", "summary": "done"},
                "stats": {
                    "backend": "claude_agent_sdk",
                    "tool_calls": 2,
                    "turns_used": 2,
                    "elapsed_s": 30.0,
                    "total_input_tokens": 6,
                    "total_cache_read_input_tokens": 900,
                    "total_cache_creation_input_tokens": 30,
                    "total_output_tokens": 40,
                },
            }
        )
    )
    states = {"steps": [{"step_idx": 0, "terminated": True}]}
    (tmp_path / "states.json").write_text(json.dumps(states))
    rec = parse_rpent_run(tmp_path)
    assert rec.input_tokens == 314 and rec.cached_input_tokens == 300 and rec.output_tokens == 14
    assert any("SDK stream" in note for note in rec.notes)


def test_claude_stream_usage_takes_output_from_the_transcript(tmp_path):
    from pyrualean.evaluation.rpent import claude_stream_usage

    def usage(cache_read, output):
        return {
            "input_tokens": 2,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": 0,
            "output_tokens": output,
        }

    stream = [
        {"type": "SystemMessage", "subtype": "init", "session_id": "sid-1"},
        {"type": "AssistantMessage", "message_id": "a", "usage": usage(100, 3)},
        {"type": "AssistantMessage", "message_id": "b", "usage": usage(200, 5)},
    ]
    lines = "\n".join(json.dumps(e) for e in stream) + "\n"
    (tmp_path / "claude_x.txt.stream.jsonl").write_text(lines)
    assert claude_stream_usage(tmp_path)["output"] == 8
    assert claude_stream_usage(tmp_path)["exact_output"] == 0

    def entry(message_id, output, **extra):
        message = {"id": message_id, "usage": {"output_tokens": output}}
        return {"type": "assistant", "message": message, **extra}

    transcript = [
        entry("a", 3),
        entry("a", 240),
        entry("sub", 999, isSidechain=True),
        entry("b", 120),
    ]
    lines = "\n".join(json.dumps(e) for e in transcript) + "\n"
    (tmp_path / "claude_x.txt.transcript.jsonl").write_text(lines)
    result = claude_stream_usage(tmp_path)
    assert result["output"] == 360 and result["exact_output"] == 1 and result["requests"] == 2


def test_parse_rpent_run_keys_robocasa_cells_like_the_code_arm(tmp_path):
    record = {"task_name": "CloseFridge", "split": "target", "seed": 2, "stats": {"tool_calls": 1}}
    (tmp_path / "transcript_CloseFridge_target_s2.json").write_text(json.dumps(record))
    steps = {"steps": [{"step_idx": 0, "terminated": True}]}
    (tmp_path / "states.json").write_text(json.dumps(steps))
    rec = parse_rpent_run(tmp_path)
    assert (rec.backend, rec.task, rec.seed) == ("robocasa", "target:CloseFridge", 2)
