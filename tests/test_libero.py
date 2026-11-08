"""Behaviour of the typed LIBERO API over an RPent-shaped backend."""

from __future__ import annotations

import pytest

from pyrualean import EpisodeFinished, LiberoRobot, Move, Pick, Segment, State, ToolError


def test_state_and_task_come_from_the_envelope(robo, backend):
    state = robo.state()
    assert isinstance(state, State)
    assert state.task == backend.TASK
    assert state.eef_pos == (0.0, 0.0, 0.68)
    assert state.gripper_opening == pytest.approx(0.08)
    assert state.object_names == ("akita_black_bowl_1", "plate_1")
    assert robo.task == backend.TASK
    assert not robo.done


def test_move_to_returns_a_typed_result_with_reached(robo, backend):
    result = robo.move_to([0.05, -0.10, 0.60])
    assert isinstance(result, Move)
    assert result.reached and result.final_dist_m == 0.0
    assert backend.calls[-1][0] == "move_to"
    assert backend.calls[-1][1]["xyz"] == [0.05, -0.10, 0.60]
    assert backend.calls[-1][1]["gripper"] == -1.0
    stalled = robo.move_to([0.05, -0.10, 1.5], max_steps=5)  # far in z: fake stalls
    assert stalled.reached is False


def test_planar_moves_over_0_30_m_are_refused_before_dispatch(robo, backend):
    before = len(backend.calls)
    with pytest.raises(ValueError, match="0.3"):
        robo.move_to([0.4, 0.0, 0.68])
    assert len(backend.calls) == before + 1  # only the initial state() lookup


def test_full_pick_and_place_terminates_and_freezes_motion(robo, backend):
    bowl = robo.segment("the black bowl")
    assert isinstance(bowl, Segment) and bowl.found and bowl.world_xyz == backend.BOWL
    robo.move_to([bowl.world_xyz[0], bowl.world_xyz[1], 0.6])
    pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
    assert isinstance(pick, Pick) and pick.success
    assert robo.state().gripper_opening == pytest.approx(0.03)
    robo.set_gripper(LiberoRobot.CLOSE, steps=8)
    plate = robo.segment("the plate")
    robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.55], gripper=LiberoRobot.CLOSE)
    release = robo.release()
    assert release.terminated and robo.done and backend.solved()
    with pytest.raises(EpisodeFinished) as info:
        robo.move_to([0.0, 0.0, 0.68])
    assert info.value.reason == "terminated"
    # Read-only calls still work after termination.
    assert robo.state().terminated


def test_carrying_with_open_gripper_drops_the_object(robo, backend):
    robo.move_to([0.05, -0.10, 0.6])
    robo.pi0_pick("pick up the black bowl")
    robo.move_to([0.05, 0.20, 0.55])  # default gripper=-1 opens the fingers
    assert backend.holding is False
    assert robo.release().terminated is False


def test_backend_error_payload_becomes_tool_error(robo, backend):
    backend.error_next = {"error": "simulated failure", "traceback": "..."}
    with pytest.raises(ToolError, match="simulated failure") as info:
        robo.set_gripper(1)
    assert info.value.tool == "set_gripper"
    assert robo._ledger.records[-1].ok is False


def test_library_side_argument_validation(robo, backend):
    with pytest.raises(ValueError, match="exactly one"):
        robo.rotate_wrist()
    with pytest.raises(ValueError):
        robo.move_to([0.0, 0.0])
    with pytest.raises(ValueError):
        robo.move_to([0.0, 0.0, 0.5], gripper=3)
    assert all(name != "rotate_wrist" for name, _ in backend.calls)


def test_segment_not_found_is_a_result_not_an_error(robo):
    result = robo.segment("the purple unicorn")
    assert result.found is False and result.world_xyz is None
    assert "no mask" in (result.error or "")


def test_back_project_out_of_bounds_raises_tool_error(robo):
    point = robo.back_project(500, 600)
    assert point.world_xyz == (0.5, 0.6, 0.42)
    with pytest.raises(ToolError, match="out of bounds"):
        robo.back_project(5000, 0)
    region = robo.region_center((100, 200), (100, 200))
    assert region.center_xyz == (0.05, 0.20, 0.42)


def test_images_and_world_maps_decode(robo):
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    image = robo.image("agentview")
    assert image.shape == (1024, 1024, 3) and image.dtype == np.uint8
    assert robo.image("wrist", resolution="low").shape == (256, 256, 3)
    assert robo.world_map("agentview").shape == (1024, 1024, 3)
    with pytest.raises(ValueError):
        robo.image("rear")


def test_ledger_records_every_call_with_env_steps(robo, backend):
    robo.state()
    robo.move_to([0.05, -0.10, 0.6])
    robo.pi0_pick("pick up the black bowl", max_chunks=8)
    records = robo._ledger.records
    assert [r.tool for r in records] == ["view_env_state", "move_to", "pi0_pick"]
    assert records[0].stateful is False and records[1].stateful is True
    assert records[1].env_steps and records[2].env_steps == 20
    assert robo._ledger.env_steps() == backend.frames
    assert robo._ledger.stateful_calls() == 2


def test_cancellation_surfaces_as_episode_finished(robo, backend):
    robo._stop_episode("timeout")
    with pytest.raises(EpisodeFinished) as info:
        robo.move_to([0.0, 0.0, 0.5])
    assert info.value.reason == "timeout" and backend.cancelled


def test_state_orientation_helpers():
    state = State(
        step=0,
        task="t",
        eef_pos=(0, 0, 0),
        eef_quat=(0.0, 0.0, 0.0, 1.0),
        gripper_qpos=(0.04, -0.04),
        gripper_opening=0.08,
        object_names=(),
        terminated=False,
        truncated=False,
    )
    assert state.eef_yaw == pytest.approx(0.0)
    assert abs(state.eef_pitch) == pytest.approx(3.141592653589793)
