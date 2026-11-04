import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - core environment without numpy
    np = None

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


@unittest.skipIf(np is None, "numpy is required")
class TLayoutTest(unittest.TestCase):
    def setUp(self):
        from datafarm.training.dataset import SEEDANCE_T_LAYOUT_1280X1080

        self.layout = SEEDANCE_T_LAYOUT_1280X1080
        frame = np.zeros((1080, 1280, 3), dtype=np.uint8)
        frame[:720] = 200
        frame[720:, :640] = 60
        frame[720:, 640:] = 120
        self.frame = frame

    def test_split_uses_head720_not_three_quarters(self):
        head, left, right = self.layout.split(self.frame)
        self.assertEqual(head.shape, (720, 1280, 3))
        self.assertEqual(left.shape, (360, 640, 3))
        self.assertEqual(right.shape, (360, 640, 3))
        # A 3H/4 split would leak wrist pixels into the head view.
        self.assertTrue((head == 200).all())
        self.assertTrue((left == 60).all())
        self.assertTrue((right == 120).all())

    def test_rejects_undeclared_carrier_size(self):
        with self.assertRaises(ValueError):
            self.layout.split(np.zeros((1080, 1440, 3), dtype=np.uint8))

    def test_seam_report_accepts_declared_boundaries(self):
        from datafarm.training.dataset import layout_seam_report

        self.assertTrue(layout_seam_report(self.frame, self.layout)["passed"])

    def test_seam_report_accepts_sharper_content_edge(self):
        from datafarm.training.dataset import layout_seam_report

        edged = self.frame.copy()
        edged[720:, 369:380] = 255  # bright stick sharper than the wrist seam
        edged[300:720] = 0  # straight head-view horizon sharper than the head seam
        report = layout_seam_report(edged, self.layout)
        self.assertTrue(report["passed"], report)
        self.assertNotEqual(report["strongest_bottom_col_seam"], 640)

    def test_seam_report_uses_best_frame(self):
        from datafarm.training.dataset import layout_seam_report

        flat = self.frame.copy()
        flat[720:] = 200  # generated frame where all three views share one colour
        self.assertFalse(layout_seam_report(flat, self.layout)["passed"])
        self.assertTrue(layout_seam_report([flat, self.frame], self.layout)["passed"])

    def test_seam_report_rejects_misplaced_boundary_over_frames(self):
        from datafarm.training.dataset import layout_seam_report

        shifted = self.frame.copy()
        shifted[720:, :700] = 60
        self.assertFalse(layout_seam_report([shifted, shifted], self.layout)["passed"])

    def test_seam_report_rejects_misplaced_head_boundary(self):
        from datafarm.training.dataset import layout_seam_report

        shifted = self.frame.copy()
        shifted[:810] = 200
        self.assertFalse(layout_seam_report(shifted, self.layout)["passed"])


class _FakeFk:
    def __init__(self, dh_is_offset=1):
        del dh_is_offset

    def CalFK(self, joints):
        # millimetres and degrees, like the official SDK
        return [[1000.0 * joints[0], 0.0, 0.0, 0.0, 0.0, 180.0 * joints[1] / 3.141592653589793]]


@unittest.skipIf(np is None, "numpy is required")
class PiperEefEncoderTest(unittest.TestCase):
    def record(self, index, left_gripper):
        qpos = [0.01 * index, 0.1, 0, 0, 0, 0, 0, 0]
        return {
            "frame_index": index,
            "actual": {"left_qpos": qpos, "right_qpos": qpos},
            "command": {"left": {"gripper_command": left_gripper}, "right": {"gripper_command": 1.0}},
        }

    def test_action_is_next_pose_with_current_gripper(self):
        from datafarm.training.dataset import PiperEefEncoder

        encoder = PiperEefEncoder(fk_factory=_FakeFk)
        state, action = encoder.records_to_state_action(
            [self.record(0, 0.0), self.record(1, 1.0), self.record(2, 0.5)]
        )
        self.assertEqual(state.shape, (2, 14))
        np.testing.assert_allclose(action[:, 0], state[:, 0] + 0.01, atol=1e-6)
        np.testing.assert_allclose(action[:, 6], state[:, 6])
        np.testing.assert_allclose(state[:, 6], [0.0, 0.105], atol=1e-6)
        self.assertAlmostEqual(float(state[0, 5]), 0.1, places=5)


def _sdk_rpy(matrix):
    """Piper SDK matrix -> RPY (roll=atan2(R21,R22), pitch=atan2(-R20,..), yaw=atan2(R10,R00))."""

    return np.array([
        np.arctan2(matrix[2, 1], matrix[2, 2]),
        np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0])),
        np.arctan2(matrix[1, 0], matrix[0, 0]),
    ])


@unittest.skipIf(np is None, "numpy is required")
class GeodesicAngleTest(unittest.TestCase):
    def test_matches_known_rotation_and_round_trips_sdk_convention(self):
        from datafarm.training.dataset import _rotation_matrices, geodesic_angle

        rpy = np.array([2.4, 1.2, 2.3])
        np.testing.assert_allclose(_sdk_rpy(_rotation_matrices(rpy)), rpy, atol=1e-9)
        self.assertAlmostEqual(float(geodesic_angle(rpy, rpy)), 0.0, places=7)
        self.assertAlmostEqual(float(geodesic_angle(np.array([0, 0, 0.01]), np.zeros(3))), 0.01, places=9)

    def test_no_gimbal_lock_gain_near_pitch_82_degrees(self):
        from datafarm.training.dataset import _rotation_matrices, geodesic_angle

        home = np.array([2.49, 1.434, 2.43])
        small = 0.002
        # About the tool z axis: near pitch 90 deg roll and yaw must swing ~1/cos(pitch) times further.
        tilt = np.array([[np.cos(small), -np.sin(small), 0], [np.sin(small), np.cos(small), 0], [0, 0, 1]])
        moved = _sdk_rpy(_rotation_matrices(home) @ tilt)
        self.assertGreater(float(np.max(np.abs(moved - home))), 3 * small)
        self.assertAlmostEqual(float(geodesic_angle(moved, home)), small, places=6)


@unittest.skipIf(np is None or not HAS_PYARROW, "numpy and pyarrow are required")
class DatasetAuditTest(unittest.TestCase):
    HOME = [0.05, 0.0, 0.19, 2.3, 1.3, 2.3, 0.105, 0.05, 0.0, 0.19, 2.4, 1.4, 2.4, 0.105]

    def write_episode(self, root, state, action):
        import pyarrow as pa
        import pyarrow.parquet as parquet

        path = Path(root) / "data" / "chunk-000" / "episode_000000.parquet"
        path.parent.mkdir(parents=True)
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
        })
        parquet.write_table(table, path)

    def contract(self):
        return {
            "schema_version": "eef_terminal_home_contract/1.1.0",
            "canonical_home_state_14d": self.HOME,
            "max_translation_error_m": 0.005,
            "max_rotation_error_rad": 0.035,
            "max_gripper_error_m": 0.002,
            "stationary_tail_frames": 10,
            "max_stationary_tail_translation_m": 0.002,
            "max_stationary_tail_rotation_rad": 0.005,
            "max_stationary_tail_gripper_m": 0.0005,
        }

    def test_terminal_home_and_action_shift_pass(self):
        from datafarm.training.dataset import audit_action_shift, audit_terminal_home

        state = np.tile(np.asarray(self.HOME, dtype=np.float32), (20, 1))
        state[:5, 0] += np.linspace(0.05, 0.01, 5, dtype=np.float32)
        action = np.vstack([state[1:], state[-1:]])
        with tempfile.TemporaryDirectory() as root:
            self.write_episode(root, state, action)
            self.assertEqual(audit_terminal_home(Path(root), self.contract())["passed"], 1)
            self.assertEqual(audit_action_shift(Path(root))["episodes"], 1)

    def test_terminal_home_rejects_episode_ending_away_from_home(self):
        from datafarm.training.dataset import audit_terminal_home

        state = np.tile(np.asarray(self.HOME, dtype=np.float32), (20, 1))
        state[:, 0] += 0.02
        with tempfile.TemporaryDirectory() as root:
            self.write_episode(root, state, state)
            with self.assertRaises(ValueError):
                audit_terminal_home(Path(root), self.contract())

    def settling_state(self, drift_m):
        state = np.tile(np.asarray(self.HOME, dtype=np.float32), (20, 1))
        state[-10:, 0] += np.linspace(drift_m, 0.0, 10, dtype=np.float32)
        return state

    def test_geometric_tail_accepts_small_settling(self):
        from datafarm.training.dataset import audit_terminal_home

        state = self.settling_state(0.001)
        with tempfile.TemporaryDirectory() as root:
            self.write_episode(root, state, state)
            result = audit_terminal_home(Path(root), self.contract())
            self.assertEqual(result["passed"], 1)
            self.assertAlmostEqual(result["observed_maxima"]["stationary_tail_translation_m"], 0.001, places=5)

    def test_geometric_tail_rejects_larger_settling(self):
        from datafarm.training.dataset import audit_terminal_home

        state = self.settling_state(0.003)
        with tempfile.TemporaryDirectory() as root:
            self.write_episode(root, state, state)
            with self.assertRaises(ValueError):
                audit_terminal_home(Path(root), self.contract())


class TrainingStateModeTest(unittest.TestCase):
    def load_tool(self):
        from datafarm.training import launch
        return launch

    @unittest.skipIf(np is None, "numpy is required")
    def test_state_mode_must_be_explicit(self):
        tool = self.load_tool()
        with self.assertRaises(ValueError):
            tool.state_conditioning({})
        self.assertEqual(tool.state_conditioning({"state_conditioning": "none"}), "none")
        
    @unittest.skipIf(np is None, "numpy is required")
    def test_submit_rewrites_fsdp_devices(self):
        tool = self.load_tool()
        self.assertEqual(
            tool.replace_fsdp_devices(["train", "--fsdp-devices", "8"], 4),
            ["train", "--fsdp-devices", "4"],
        )


if __name__ == "__main__":
    unittest.main()
