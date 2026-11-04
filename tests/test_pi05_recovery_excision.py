from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - core environment without numpy
    np = None

from datafarm.training.dataset import (
    RECOVERY_EXCISION_PHASES, recovery_excisions, retained_record_indices,
)


def payload(frames=60, start=20, end=44, splice_gap=0.0, phases=RECOVERY_EXCISION_PHASES, prefix="object_2"):
    """A left-arm trajectory whose detour returns to the pre-detour joint state."""
    records = []
    for index in range(frames):
        left = [0.10 + 0.001 * index] * 6 + [0.04]
        phase = None
        if start <= index <= end:
            share = (end - start + 1) // max(len(phases), 1)
            phase = f"{prefix}_{phases[min((index - start) // max(share, 1), len(phases) - 1)]}"
            left = [0.50 + 0.001 * index] * 6 + [0.04]  # off to the misaligned pose
            if index == end:  # recentred on the pre-detour target
                left = [0.10 + 0.001 * (start - 1) + splice_gap] * 6 + [0.04]
        records.append({
            "frame_index": index, "execution_phase": phase,
            "actual": {"left_qpos": left, "right_qpos": [0.0] * 7},
        })
    return {"records": records}


class RecoveryExcisionTest(unittest.TestCase):
    def test_off_by_default(self):
        import inspect
        from datafarm.training.dataset import materialize_episode
        self.assertIs(inspect.signature(materialize_episode).parameters["excise_recovery"].default, False)

    def test_detour_range_covers_the_whole_recovery(self):
        cuts = recovery_excisions(payload())
        self.assertEqual(len(cuts), 1)
        self.assertEqual((cuts[0]["start_frame"], cuts[0]["end_frame"], cuts[0]["dropped_frames"]), (20, 44, 25))

    def test_retained_indices_splice_the_two_visits_together(self):
        data = payload()
        retained = retained_record_indices(data, recovery_excisions(data))
        self.assertEqual(len(retained), 60 - 25)
        self.assertEqual(retained[19:21], [19, 45])  # the detour is gone, the trajectory continues
        self.assertNotIn(30, retained)

    def test_no_recovery_means_no_excision(self):
        data = payload()
        for record in data["records"]:
            record["execution_phase"] = None
        self.assertEqual(recovery_excisions(data), ())
        self.assertEqual(len(retained_record_indices(data, ())), 60)

    def test_refuses_to_splice_when_the_arm_did_not_return(self):
        with self.assertRaises(ValueError) as caught:
            recovery_excisions(payload(splice_gap=0.05))
        self.assertIn("teleport", str(caught.exception))

    def test_incomplete_recovery_is_an_error(self):
        with self.assertRaises(ValueError):
            recovery_excisions(payload(phases=("misaligned_approach", "retract")))

    @unittest.skipIf(np is None, "numpy is required")
    def test_state_action_pairs_skip_the_detour(self):
        from datafarm.training.dataset import PiperEefEncoder

        data = payload()
        retained = retained_record_indices(data, recovery_excisions(data))
        poses = np.array([data["records"][i]["actual"]["left_qpos"] for i in retained])
        # the splice step is no larger than an ordinary step
        steps = np.abs(np.diff(poses[:, 0]))
        self.assertLessEqual(float(steps[19]), float(steps.max()))
        self.assertIsNotNone(PiperEefEncoder)


if __name__ == "__main__":
    unittest.main()
