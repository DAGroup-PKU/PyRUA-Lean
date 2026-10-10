"""Offline checks of the OpenAI chat-completions runtime (``pyrualean.openai_runtime``).

The provider is a scripted ``transport`` (no network): each call returns the
next canned chat completion.  The robot is the fake LIBERO backend of
``conftest.py`` behind a real ``ArmToolkit``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pyrualean import openai_runtime as orun
from pyrualean.arms import ArmToolkit


def _reply(
    tool_calls: list[tuple[str, str, dict[str, Any] | str]] | None = None,
    text: str | None = None,
    reasoning: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
            for call_id, name, args in tool_calls
        ]
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": usage
        or {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 600},
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


class ScriptedTransport:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Any]:
        # ``body["messages"]`` is the runtime's live list: snapshot it.
        self.requests.append({"url": url, "body": copy.deepcopy(body), "timeout_s": timeout_s})
        if not self.replies:
            raise AssertionError("transport called more times than scripted")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return item
        return 200, item


@pytest.fixture
def card_images(tmp_path: Path) -> list[Path]:
    from PIL import Image

    paths = []
    for name in ("agentview_high.png", "wrist_high.png"):
        path = tmp_path / "card" / name
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (1024, 1024), (10, 20, 30)).save(path)
        paths.append(path)
    return paths


def _toolkit(robo: Any, tmp_path: Path, **kwargs: Any) -> ArmToolkit:
    return ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", **kwargs)


def _run(
    toolkit: ArmToolkit, transport: ScriptedTransport, tmp_path: Path, images: list[Path], **kw
):
    return orun.run_openai_agent(
        "PROMPT TEXT",
        images,
        toolkit,
        model="test-model",
        base_url="http://provider.test/v1",
        api_key="k",
        max_turns=kw.pop("max_turns", None),
        budget_s=kw.pop("budget_s", 600.0),
        out_dir=tmp_path / "out",
        transport=transport,
        sleep=lambda s: None,
        **kw,
    )


def test_tools_spec_converts_to_openai_functions(robo, tmp_path):
    tools = orun.openai_tools(_toolkit(robo, tmp_path).get_tools_spec())
    assert [t["function"]["name"] for t in tools] == ["python", "finish"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["required"] == ["code"]


def test_full_episode_with_images_and_finish(robo, backend, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path, images="on-demand", feedback="pure")
    code_pick = (
        "bowl = robo.segment('the black bowl')\n"
        "robo.move_to([bowl.world_xyz[0], bowl.world_xyz[1], 0.60])\n"
        "pick = robo.pi0_pick('pick up the bowl', max_chunks=8)\n"
        "print(pick.success)\n"
        "robo.show('agentview')\n"
    )
    code_place = (
        "plate = robo.segment('the white plate')\n"
        "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.60], gripper=robo.CLOSE)\n"
        "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.47], gripper=robo.CLOSE)\n"
        "print(robo.release())\n"
    )
    transport = ScriptedTransport(
        [
            _reply([("c1", "python", {"code": code_pick})], text="Picking.", reasoning="think 1"),
            _reply([("c2", "python", {"code": code_place})], text="Placing."),
            _reply([("c3", "finish", {"status": "success", "summary": "done"})], text="Done."),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images, max_turns=40)

    assert generation["status"] == "completed" and generation["returncode"] == 0
    assert backend.solved() and toolkit.finish["status"] == "success"
    assert generation["requests"] == 3
    assert generation["tool_calls"] == {"python": 2, "finish": 1}
    assert generation["input_tokens"] == 3000 and generation["cached_input_tokens"] == 1800
    assert generation["output_tokens"] == 150 and generation["reasoning_output_tokens"] == 60
    assert generation["request_profile"] == [
        1000,
        2000,
        3000,
    ]  # cumulative, like the Claude runtime

    # Request shape: first user message carries the prompt and both card images.
    first = transport.requests[0]["body"]
    assert first["model"] == "test-model" and first["tool_choice"] == "auto"
    assert transport.requests[0]["url"].endswith("/v1/chat/completions")
    user0 = first["messages"][0]
    assert user0["role"] == "user" and user0["content"][0] == {
        "type": "text",
        "text": "PROMPT TEXT",
    }
    assert [p["type"] for p in user0["content"][1:]] == ["image_url", "image_url"]
    assert user0["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    # Second request: assistant (with its tool call), tool result text, then the
    # on-demand image as a separate user message naming the call.
    second = transport.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "tool", "user"]
    assert second[1]["tool_calls"][0]["function"]["name"] == "python"
    assert "reasoning_content" not in second[1]  # not replayed to the provider
    assert second[2]["tool_call_id"] == "c1"
    tool_text = json.loads(second[2]["content"])
    assert tool_text["cell"] == 1 and "True" in tool_text["stdout"]
    assert tool_text["images"] == ["agentview_high.png"]
    assert second[3]["content"][0]["text"] == orun.IMAGE_NOTE.format(
        sources="c1 (agentview_high.png)"
    )
    assert second[3]["content"][1]["type"] == "image_url"

    # Outputs: the transcript keeps reasoning and references saved PNGs.
    out = tmp_path / "out"
    recorded = json.loads((out / "messages.json").read_text())
    assert recorded[1]["reasoning_content"] == "think 1"
    refs = [
        p["image_ref"]
        for m in recorded
        for p in (m["content"] if isinstance(m["content"], list) else [])
        if p.get("type") == "image_ref"
    ]
    assert len(refs) == 3 and all(Path(r).is_file() for r in refs)
    assert (out / "last_message.md").read_text() == "Done."
    events = [json.loads(line) for line in (out / "openai_events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events].count("request") == 3
    assert [e for e in events if e["type"] == "tool"][0]["images"] == 1


def test_no_tool_call_is_nudged_then_ends(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport([_reply(text="Let me think."), _reply(text="Still thinking.")])
    generation = _run(toolkit, transport, tmp_path, card_images, nudges=1)
    assert generation["status"] == "completed" and generation["requests"] == 2
    second = transport.requests[1]["body"]["messages"]
    assert second[-1] == {"role": "user", "content": orun.NUDGE_TEXT}
    assert toolkit.finish is None


def test_decision_budget_ends_episode_and_allows_final_finish(robo, backend, tmp_path, card_images):
    # max_turns=2: the third python call is refused by the toolkit (which ends the
    # episode on the robot and tells the model to call finish); one more reply is
    # allowed so finish can be recorded.
    toolkit = _toolkit(robo, tmp_path, max_turns=2)
    transport = ScriptedTransport(
        [
            _reply([("c1", "python", {"code": "print(1)"})]),
            _reply([("c2", "python", {"code": "print(2)"})]),
            _reply([("c3", "python", {"code": "print(3)"})]),
            _reply([("c4", "finish", {"status": "stuck", "summary": "out of calls"})]),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images, max_turns=2)
    assert toolkit.turn_budget_exhausted and toolkit.finish["status"] == "stuck"
    assert generation["requests"] == 4 and generation["tool_calls"] == {"python": 3, "finish": 1}
    assert [t["ok"] for t in toolkit.turns] == [True, True, False, True]
    refused = json.loads(transport.requests[3]["body"]["messages"][-1]["content"])
    assert "call budget exhausted" in refused["error"]
    assert not backend.solved()


def test_budget_exhausted_without_finish_stops_after_one_more_reply(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path, max_turns=1)
    transport = ScriptedTransport(
        [
            _reply([("c1", "python", {"code": "print(1)"})]),
            _reply([("c2", "python", {"code": "print(2)"})]),  # refused: budget spent
            _reply([("c3", "python", {"code": "print(3)"})]),  # the one allowed extra reply
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images, max_turns=1)
    assert generation["requests"] == 3 and toolkit.finish is None
    assert [t["ok"] for t in toolkit.turns] == [True, False, False]
    assert generation["status"] == "completed"


def test_bad_arguments_are_reported_not_fatal(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [
            _reply([("c1", "python", "{not json")]),
            _reply([("c2", "finish", {"status": "failure", "summary": "x"})]),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images)
    second = transport.requests[1]["body"]["messages"]
    assert "not valid JSON" in second[2]["content"]
    assert generation["status"] == "completed" and toolkit.finish["status"] == "failure"


def test_provider_errors_retry_then_fail(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport([(503, {"error": "busy"}), RuntimeError("boom"), (500, "bad")])
    generation = _run(toolkit, transport, tmp_path, card_images, retries=2)
    assert generation["status"] == "error" and generation["returncode"] == 2
    assert generation["requests"] == 3
    assert len(generation["warnings"]) == 2 and len(generation["errors"]) == 1  # 2 retried, 1 fatal


def test_recovered_provider_error_is_a_warning_not_an_error(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [
            (503, {"error": "busy"}),
            _reply([("c1", "finish", {"status": "failure", "summary": "x"})]),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images)
    assert generation["status"] == "completed" and generation["errors"] == []
    assert len(generation["warnings"]) == 1 and generation["requests"] == 2


def test_rate_limits_are_waited_out_beyond_the_retries(robo, tmp_path, card_images, monkeypatch):
    toolkit = _toolkit(robo, tmp_path)
    monkeypatch.setattr(orun.random, "random", lambda: 0.5)  # no jitter
    waits: list[float] = []
    transport = ScriptedTransport(
        [(429, "")] * 8 + [_reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    generation = orun.run_openai_agent(
        "PROMPT TEXT",
        card_images,
        toolkit,
        model="test-model",
        base_url="http://provider.test/v1",
        api_key="k",
        max_turns=None,
        budget_s=3600.0,
        out_dir=tmp_path / "out",
        transport=transport,
        sleep=waits.append,
        retries=2,
    )
    assert generation["status"] == "completed" and generation["errors"] == []
    assert generation["requests"] == 9 and len(generation["warnings"]) == 5  # first five kept
    assert waits == [5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 120.0, 120.0]


def test_client_error_is_not_retried(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport([(400, {"error": "context too long"})])
    generation = _run(toolkit, transport, tmp_path, card_images, retries=5)
    assert generation["status"] == "error" and generation["requests"] == 1
    assert len(generation["errors"]) == 1 and generation["warnings"] == []


def test_wall_clock_timeout_stops_episode(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    stopped: list[str] = []
    transport = ScriptedTransport([_reply([("c1", "python", {"code": "print(1)"})])])
    generation = orun.run_openai_agent(
        "PROMPT",
        card_images,
        toolkit,
        model="m",
        base_url="http://provider.test/v1/",
        api_key=None,
        max_turns=None,
        budget_s=0.0,
        out_dir=tmp_path / "out",
        transport=transport,
        on_timeout=lambda: stopped.append("yes"),
    )
    assert generation["status"] == "timeout" and generation["returncode"] is None
    assert stopped == ["yes"] and transport.requests == []


def test_images_can_be_downscaled_with_a_note(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [_reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    generation = _run(toolkit, transport, tmp_path, card_images, image_max_side=512)
    user0 = transport.requests[0]["body"]["messages"][0]["content"]
    assert "downscaled by 0.500" in user0[0]["text"]
    import base64
    import io

    from PIL import Image

    data = base64.b64decode(user0[1]["image_url"]["url"].split(",", 1)[1])
    assert Image.open(io.BytesIO(data)).size == (512, 512)
    assert generation["options"]["image_max_side"] == 512


def test_extra_body_and_sampling_options_are_sent(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [_reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    _run(
        toolkit,
        transport,
        tmp_path,
        card_images,
        temperature=0.7,
        max_tokens=1234,
        extra_body={"reasoning_effort": "high"},
    )
    body = transport.requests[0]["body"]
    assert body["temperature"] == 0.7 and body["max_tokens"] == 1234
    assert body["reasoning_effort"] == "high" and body["tool_choice"] == "auto"


def test_tool_choice_required_is_sent(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [_reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    generation = _run(toolkit, transport, tmp_path, card_images, tool_choice="required")
    assert transport.requests[0]["body"]["tool_choice"] == "required"
    assert generation["options"]["tool_choice"] == "required"


def test_two_tool_calls_in_one_reply_keep_the_message_order_legal(robo, tmp_path, card_images):
    # Images may not sit between a `tool` reply and the next one: every tool message of a turn
    # must follow its `tool_calls` directly, so the images wait for the last one.
    toolkit = _toolkit(robo, tmp_path, images="on-demand", feedback="pure")
    show = "robo.show('agentview')\nprint('looked')\n"
    transport = ScriptedTransport(
        [
            _reply([("c1", "python", {"code": show}), ("c2", "python", {"code": "print(2)"})]),
            _reply([("c3", "finish", {"status": "failure", "summary": "x"})]),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images)
    second = transport.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "tool", "tool", "user"]
    assert [m.get("tool_call_id") for m in second[2:4]] == ["c1", "c2"]
    assert "c1 (agentview_high.png)" in second[4]["content"][0]["text"]
    assert generation["tool_calls"]["python"] == 2


def test_provider_without_tool_call_ids_still_answers_the_right_call(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    reply = _reply([("", "python", {"code": "print(1)"})])
    reply["choices"][0]["message"]["tool_calls"][0].pop("id")
    transport = ScriptedTransport(
        [reply, _reply([("c2", "finish", {"status": "failure", "summary": "x"})])]
    )
    _run(toolkit, transport, tmp_path, card_images)
    second = transport.requests[1]["body"]["messages"]
    assistant, tool = second[1], second[2]
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]


def test_finish_with_unparsable_arguments_does_not_end_the_episode(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [
            _reply([("c1", "finish", '{"status": "succ')]),  # truncated arguments
            _reply([("c2", "finish", {"status": "failure", "summary": "really done"})]),
        ]
    )
    generation = _run(toolkit, transport, tmp_path, card_images)
    assert generation["requests"] == 2
    assert toolkit.finish["summary"] == "really done"
    assert "not valid JSON" in transport.requests[1]["body"]["messages"][2]["content"]


def test_text_only_reply_replays_with_a_string_content(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    reply = _reply(text=None)  # a thinking model that spent its budget reasoning
    transport = ScriptedTransport(
        [reply, _reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    _run(toolkit, transport, tmp_path, card_images, nudges=1)
    assistant = transport.requests[1]["body"]["messages"][1]
    assert assistant["content"] == "" and "tool_calls" not in assistant


def test_repeated_unparsable_replies_stop_the_episode(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport([_reply([(f"c{i}", "python", "{bad")]) for i in range(8)])
    generation = _run(toolkit, transport, tmp_path, card_images)
    assert generation["requests"] == 4  # three tolerated, the fourth ends it
    assert any("did not parse" in w for w in generation["warnings"])


def test_max_tokens_zero_is_not_sent(robo, tmp_path, card_images):
    toolkit = _toolkit(robo, tmp_path)
    transport = ScriptedTransport(
        [_reply([("c1", "finish", {"status": "failure", "summary": "x"})])]
    )
    _run(toolkit, transport, tmp_path, card_images, max_tokens=0)
    assert "max_tokens" not in transport.requests[0]["body"]


def test_a_long_tool_call_is_cut_off_at_the_wall_clock(robo, tmp_path, card_images):
    # The loop is the host thread, so a cell that keeps driving the robot must still be stopped at
    # the deadline: the timer sets the episode's stop reason and the next *stateful* primitive
    # raises EpisodeFinished.  (A cell that only computes still runs to its end, exactly as under
    # the Codex and Claude runtimes - neither puts a timeout around a single cell.)
    toolkit = _toolkit(robo, tmp_path)
    busy = (
        "import time\n"
        "for step in range(40):\n"
        "    robo.move_to([0.0, 0.0, 0.68 - 0.001 * step])\n"
        "    time.sleep(0.05)\n"
    )
    transport = ScriptedTransport([_reply([("c1", "python", {"code": busy})])])
    stopped: list[str] = []

    def on_timeout() -> None:
        stopped.append("yes")
        toolkit.robo._stop_episode("timeout")

    generation = orun.run_openai_agent(
        "PROMPT",
        card_images,
        toolkit,
        model="m",
        base_url="http://provider.test/v1",
        api_key=None,
        max_turns=None,
        budget_s=0.4,
        out_dir=tmp_path / "out",
        transport=transport,
        sleep=lambda s: None,
        on_timeout=on_timeout,
    )
    assert stopped == ["yes"]  # the hook fires once, from the timer
    # The cell needs ~2 s for its 40 moves; the stop cuts it at the first move past 0.4 s.
    assert generation["elapsed_s"] < 1.5, generation["elapsed_s"]
    assert toolkit.turns[-1]["tool"] == "python"
