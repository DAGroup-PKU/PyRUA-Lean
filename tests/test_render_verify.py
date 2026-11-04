"""Hermetic tests for the spec-driven Render verifier (datafarm.verify).

Synthetic T-layout clips: a brown "cup" square moves across the head panel and a
dark "robot" bar sweeps the wrist panels.  The generated clip re-colours
everything slightly (as a renderer would); the real clip has the target colours.
"""

import importlib.util
import unittest
from dataclasses import replace

HAS_CV2 = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None

SPEC = {
    "schema_version": "datafarm.render-verifier-spec/1.0.0",
    "spec_id": "synthetic-v1",
    "task_family": "synthetic",
    "task_ir_roles": ["cup", "robot"],
    "layout_size": [320, 270],
    "fps": 30.0,
    "work_scale": 1.0,
    "stride": 2,
    "panels": [
        {"name": "head", "box": [0, 0, 320, 180], "motion": "fixed"},
        {"name": "left_wrist", "box": [0, 180, 160, 90], "motion": "moving"},
        {"name": "right_wrist", "box": [160, 180, 160, 90], "motion": "moving"},
    ],
    "entities": [
        {"role": "cup", "carrier_keys": [{"lower": [5, 80, 60], "upper": [25, 255, 220]}],
         "target_keys": [{"lower": [3, 60, 40], "upper": [28, 255, 240]}], "min_area_frac": 0.004},
        {"role": "robot", "carrier_keys": [{"lower": [0, 0, 0], "upper": [179, 40, 70]}],
         "target_keys": [{"lower": [0, 0, 0], "upper": [179, 255, 60]}], "checks": ["missing"],
         "adaptive": False, "min_area_frac": 0.004},
    ],
    "realism": {"mode": "sim2real"},
}

CUP_CARRIER = (40, 90, 150)   # BGR brown
CUP_TARGET = (45, 100, 165)
TABLE = (200, 200, 200)


def make_frames(count=90, cup=CUP_CARRIER, table=TABLE, robot=(45, 45, 45), delete=(), extra=(),
                freeze=(), shift=0, noise=0):
    import numpy as np

    rng = np.random.default_rng(0)
    frames = []
    for index in range(count):
        t = max(0, index - shift)
        if index in freeze:
            t = min(freeze)
        frame = np.full((270, 320, 3), table, np.uint8)
        x = 40 + t * 2
        if index not in delete:
            frame[60:110, x:x + 40] = cup
        if index in extra:
            frame[120:170, 250:290] = cup
        wx = 20 + (t * 3) % 100
        frame[200:250, wx:wx + 30] = robot
        frame[200:250, 180 + (t * 2) % 100:210 + (t * 2) % 100] = robot
        if noise:
            frame = np.clip(frame.astype(int) + rng.integers(-noise, noise + 1, frame.shape), 0, 255).astype(np.uint8)
        frames.append(frame)
    return np.stack(frames)


def clip_from_arrays(frames_full, *, fps, layout_size=None, work_scale=0.5, gray_scale=0.25, stride=2):
    """Build a clip from full-rate BGR frames already in layout coordinates."""

    import cv2
    import numpy as np

    from datafarm.verify.media import Clip, _sizes

    frames_full = np.asarray(frames_full, np.uint8)
    height, width = frames_full.shape[1:3]
    layout_size = layout_size or (width, height)
    work_size, gray_size = _sizes(layout_size, work_scale, gray_scale)
    work, gray = [], []
    for index, frame in enumerate(frames_full):
        if (frame.shape[1], frame.shape[0]) != tuple(layout_size):
            frame = cv2.resize(frame, tuple(layout_size), interpolation=cv2.INTER_AREA)
        if index % stride == 0:
            work.append(cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA))
        gray.append(cv2.cvtColor(cv2.resize(frame, gray_size, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY))
    return Clip(
        frames=np.stack(work), gray=np.stack(gray), stride=stride, fps=fps,
        frame_count=len(frames_full), native_size=(width, height), layout_size=tuple(layout_size),
        work_scale=work_scale, gray_scale=gray_scale,
    )


@unittest.skipUnless(HAS_CV2, "opencv and numpy are required")
class RenderVerifierTest(unittest.TestCase):
    def setUp(self):
        from datafarm.verify.spec import spec_from_dict

        self.spec = spec_from_dict(SPEC)
        kw = dict(fps=30.0, work_scale=1.0, stride=2)
        self.clip = lambda frames: clip_from_arrays(frames, **kw)
        self.carrier = self.clip(make_frames())
        self.real = self.clip(make_frames(cup=CUP_TARGET, noise=3))

    def run_checks(self, generated, **kwargs):
        from datafarm.verify.verifier import run_checks

        return run_checks(self.spec, self.carrier, self.clip(generated), self.real, **kwargs)

    def verdicts(self, receipt):
        return {c["name"]: c["verdict"] for c in receipt["checks"]}

    def test_faithful_render_passes(self):
        receipt = self.run_checks(make_frames(cup=CUP_TARGET, noise=3))
        self.assertEqual(receipt["verdict"], "pass", receipt["failed"])

    def test_deleted_entity_fails_entity_check(self):
        receipt = self.run_checks(make_frames(cup=CUP_TARGET, delete=range(30, 50)))
        self.assertEqual(self.verdicts(receipt)["entity_cup"], "fail")

    def test_hallucinated_entity_fails_entity_check(self):
        receipt = self.run_checks(make_frames(cup=CUP_TARGET, extra=range(30, 50)))
        self.assertEqual(self.verdicts(receipt)["entity_cup"], "fail")

    def test_persistent_compact_duplicate_cannot_disable_its_own_detector(self):
        # Unlike a short injection, an always-present duplicate enters the
        # median background-leak statistic and used to cause abstention.
        receipt = self.run_checks(make_frames(cup=CUP_TARGET, extra=range(90)))
        self.assertEqual(self.verdicts(receipt)["entity_cup"], "fail")

    def test_entity_coloured_background_still_abstains(self):
        from datafarm.verify.checks import Context, check_entity

        cup = replace(self.spec.entities[0], checks=("unsupported:head",), adaptive=False)
        spec = replace(self.spec, entities=(cup, *self.spec.entities[1:]))
        for shape in ("broad", "thin_border"):
            with self.subTest(shape=shape):
                frames = make_frames(cup=CUP_TARGET)
                if shape == "broad":
                    frames[:, 0:55, :] = CUP_TARGET
                else:
                    frames[:, :, 0:15] = CUP_TARGET
                result = check_entity(Context(spec, self.carrier, self.clip(frames)), cup)
                self.assertEqual(result.verdict, "abstain")

    def test_broader_presence_key_recognises_dim_real_object_but_not_absence(self):
        from datafarm.verify.spec import ColorKey

        cup = replace(self.spec.entities[0], presence_keys=(ColorKey((3, 20, 40), (28, 255, 240)),))
        self.spec = replace(self.spec, entities=(cup, *self.spec.entities[1:]))
        muted = (100, 110, 120)  # Same hue as target, below the strict S>=60 key.
        present = self.run_checks(make_frames(cup=muted))
        self.assertEqual(self.verdicts(present)["entity_cup"], "pass")
        absent = self.run_checks(make_frames(cup=muted, delete=range(30, 50)))
        self.assertEqual(self.verdicts(absent)["entity_cup"], "fail")

    def test_presence_key_does_not_weaken_duplicate_detection(self):
        from datafarm.verify.spec import ColorKey

        cup = replace(self.spec.entities[0], presence_keys=(ColorKey((3, 20, 40), (28, 255, 240)),))
        self.spec = replace(self.spec, entities=(cup, *self.spec.entities[1:]))
        frames = make_frames(cup=(100, 110, 120))
        frames[30:50, 120:170, 250:290] = CUP_TARGET
        duplicate = self.run_checks(frames)
        self.assertEqual(self.verdicts(duplicate)["entity_cup"], "fail")

    def test_edge_reference_filters_flat_colour_blob_but_keeps_textured_duplicate(self):
        from datafarm.verify.checks import Context, check_entity

        cup = replace(self.spec.entities[0], adaptive=False,
                      unsupported_edge_reference_fraction={"head": 0.6})
        self.spec = replace(self.spec, entities=(cup, *self.spec.entities[1:]))
        faithful = make_frames(cup=CUP_TARGET)
        for frame in range(len(faithful)):
            x = 40 + frame * 2
            faithful[frame, 70:82, x + 5:x + 35] = (0, 30, 100)
        flat = faithful.copy()
        flat[30:50, 120:170, 250:290] = CUP_TARGET
        duplicate = faithful.copy()
        for frame in range(30, 50):
            x = 40 + frame * 2
            duplicate[frame, 120:170, 250:290] = faithful[frame, 60:110, x:x + 40]

        def check(frames):
            return check_entity(Context(self.spec, self.carrier, self.clip(frames)), cup)

        self.assertEqual(check(faithful).verdict, "pass")
        flat_result = check(flat)
        self.assertEqual(flat_result.verdict, "pass")
        self.assertGreater(flat_result.metrics["head"]["unsupported_low_edge_discarded"], 0)
        self.assertEqual(check(duplicate).verdict, "fail")

    def test_shadow_exposure_estimator_preserves_real_bright_defects(self):
        import numpy as np
        from datafarm.verify.checks import _background_exposure_shift

        judged = np.ones((10, 10), bool)
        no_far_background = np.zeros((10, 10), bool)
        broad_shadow = np.array([-52.0] * 70 + [0.0] * 30, np.float32).reshape(10, 10)
        self.assertEqual(_background_exposure_shift(broad_shadow, judged, no_far_background),
                         (0.0, True))
        global_darkening = np.full((10, 10), -50.0, np.float32)
        self.assertEqual(_background_exposure_shift(global_darkening, judged, no_far_background),
                         (-50.0, False))
        bright_blob = np.array([70.0] * 10 + [0.0] * 90, np.float32).reshape(10, 10)
        self.assertEqual(_background_exposure_shift(bright_blob, judged, no_far_background),
                         (0.0, False))
        # No source motion: an unrelated darkening must not get shadow treatment.
        far_background = np.ones((10, 10), bool)
        self.assertEqual(_background_exposure_shift(broad_shadow, judged, far_background),
                         (-52.0, False))

    def test_persistent_bright_background_patch_still_fails(self):
        frames = make_frames(cup=CUP_TARGET)
        frames[30:50, 10:70, 220:300] = (255, 255, 255)
        receipt = self.run_checks(frames)
        self.assertEqual(self.verdicts(receipt)["background_stability"], "fail")

    def test_departing_initial_shadow_can_return_to_nearby_surface_brightness(self):
        from datafarm.verify.spec import Threshold
        frames = make_frames(cup=CUP_TARGET)
        frames[:30, 120:150, 100:150] = (155, 155, 155)
        self.spec = replace(self.spec, thresholds={**self.spec.thresholds,
            "background_mean_drift_max": Threshold(60, "coarse exposure rule")})
        receipt = self.run_checks(frames, families=["background"])
        self.assertEqual(self.verdicts(receipt)["background_stability"], "pass")
        # A new light brighter than the opening surface is still a defect.
        frames[30:, 120:150, 100:150] = (255, 255, 255)
        receipt = self.run_checks(frames, families=["background"])
        self.assertEqual(self.verdicts(receipt)["background_stability"], "fail")

    def test_background_static_fraction_limit_skips_unmeasurable_frames(self):
        from datafarm.verify.spec import Threshold

        carrier = make_frames()
        # Most of the fixed head view changes in the source.  The narrow strip
        # still matching frame zero cannot establish whether a large generated
        # shadow is a background defect.
        carrier[30:50, :180, 50:] = (80, 80, 80)
        generated = carrier.copy()
        generated[30:50, 20:150, :35] = (255, 255, 255)
        self.carrier = self.clip(carrier)
        self.spec = replace(self.spec, thresholds={
            **self.spec.thresholds,
            "background_min_static_fraction": Threshold(0.10, "synthetic low-background control"),
        })
        receipt = self.run_checks(generated, families=["background"])
        check = receipt["checks"][0]
        self.assertEqual(check["verdict"], "pass")
        self.assertGreater(check["metrics"]["head"]["skipped_low_static_frames"], 0)

        # The same rule must still catch an unrelated bright patch when the
        # source leaves enough of the background visible.
        self.carrier = self.clip(make_frames())
        bright = make_frames(cup=CUP_TARGET)
        bright[30:50, 10:70, 220:300] = (255, 255, 255)
        receipt = self.run_checks(bright, families=["background"])
        self.assertEqual(self.verdicts(receipt)["background_stability"], "fail")

    def test_event_local_motion_does_not_confuse_nearby_motion_with_event(self):
        from datafarm.verify.spec import EventSpec, Threshold

        base = make_frames(count=120)
        source = base[[min(max(t - 42, 0), 25) for t in range(120)]]
        delayed = source[[max(0, t - 12) for t in range(120)]]
        self.carrier = self.clip(source)
        for minimum in (0.01, 0.1):
            with self.subTest(minimum=minimum):
                thresholds = {**self.spec.thresholds,
                              "event_local_motion_min": Threshold(minimum, "synthetic event control")}
                static = EventSpec("post-motion-static", 82, ("cup", "robot"), ("head",))
                self.spec = replace(self.spec, events=(static,), thresholds=thresholds)
                receipt = self.run_checks(delayed, families=["event"])
                self.assertEqual(self.verdicts(receipt)["event_alignment"], "abstain")
                skipped = receipt["checks"][0]["metrics"][static.event_id]["skipped_panels"]
                self.assertEqual(skipped[0]["reason"], "insufficient_source_motion_at_event")

                moving = EventSpec("visible-motion", 48, ("cup", "robot"), ("head",))
                self.spec = replace(self.spec, events=(moving,))
                receipt = self.run_checks(delayed, families=["event"])
                self.assertEqual(self.verdicts(receipt)["event_alignment"], "fail")

    def test_disconnected_colour_sliver_does_not_enlarge_entity(self):
        frames = make_frames(cup=CUP_TARGET, noise=3)
        for index in range(20):
            x = 40 + index * 2
            frames[index, 60:110, x + 60:x + 62] = CUP_TARGET
        receipt = self.run_checks(frames)
        self.assertEqual(self.verdicts(receipt)["entity_cup"], "pass")

    def test_connected_entity_growth_still_fails(self):
        frames = make_frames(cup=CUP_TARGET, noise=3)
        for index in range(20):
            x = 40 + index * 2
            frames[index, 60:110, x + 40:x + 65] = CUP_TARGET
        receipt = self.run_checks(frames)
        self.assertEqual(self.verdicts(receipt)["entity_cup"], "fail")

    def test_instance_core_ignores_attached_colour_wisp_but_catches_real_growth_and_copy(self):
        cup = replace(self.spec.entities[0], instance_open_px_work=7)
        tuned = replace(self.spec, entities=(cup, *self.spec.entities[1:]))

        def entity_verdict(frames, spec):
            from datafarm.verify.checks import Context, check_entity
            return check_entity(Context(spec, self.carrier, self.clip(frames)), spec.entities[0]).verdict

        wispy = make_frames(cup=CUP_TARGET)
        for index in range(20):
            x = 40 + index * 2
            wispy[index, 84:88, x + 40:x + 65] = CUP_TARGET
        self.assertEqual(entity_verdict(wispy, self.spec), "fail")
        self.assertEqual(entity_verdict(wispy, tuned), "pass")

        enlarged = make_frames(cup=CUP_TARGET)
        copied = make_frames(cup=CUP_TARGET)
        for index in range(20):
            x = 40 + index * 2
            enlarged[index, 60:110, x + 40:x + 65] = CUP_TARGET
            copied[index, 60:110, x + 80:x + 120] = CUP_TARGET
            copied[index, 84:88, x + 40:x + 80] = CUP_TARGET
        self.assertEqual(entity_verdict(enlarged, tuned), "fail")
        self.assertEqual(entity_verdict(copied, tuned), "fail")

    def test_frozen_frames_fail_temporal_check(self):
        receipt = self.run_checks(make_frames(cup=CUP_TARGET, freeze=range(40, 60)))
        self.assertEqual(self.verdicts(receipt)["temporal_consistency"], "fail")

    def test_source_supported_dark_panel_is_not_a_camera_flash(self):
        from datafarm.verify.checks import Context, check_temporal
        from datafarm.verify.spec import Threshold
        import numpy as np

        carrier = np.full((30, 270, 320, 3), 200, np.uint8)
        generated = np.full_like(carrier, 140)
        for i in range(30):
            height = max(0, min(90, (i - 8) * 16))
            carrier[i, 180:180 + height, 160:320] = 188
            generated[i, 180:180 + height, 160:320] = 20
            carrier[i, 255:270, 160:320] = 35  # stationary dark gripper in the wrist view
            generated[i, 255:270, 160:320] = 30
        original = check_temporal(Context(self.spec, self.clip(carrier), self.clip(generated)))
        self.assertEqual(original.verdict, "fail", original.metrics)
        tuned = replace(self.spec, thresholds={"temporal_source_motion_share_min": Threshold(0.70)})
        result = check_temporal(Context(tuned, self.clip(carrier), self.clip(generated)))
        self.assertEqual(result.verdict, "pass", result.metrics)
        self.assertGreater(result.metrics["right_wrist"]["source_supported_steps"], 0)

        flashed = generated.copy()
        flashed[14, 180:270, 160:320] = np.maximum(
            flashed[14, 180:270, 160:320].astype(np.int16) - 35, 0).astype(np.uint8)
        flash = check_temporal(Context(tuned, self.clip(carrier), self.clip(flashed)))
        self.assertEqual(flash.verdict, "fail", flash.metrics)

    def test_carrier_copy_fails_realism(self):
        receipt = self.run_checks(make_frames(table=(200, 200, 200)))
        self.assertIn(self.verdicts(receipt)["realism_vs_real_world"], ("fail",))

    def test_missing_real_reference_abstains(self):
        from datafarm.verify.verifier import run_checks

        receipt = run_checks(self.spec, self.carrier, self.clip(make_frames(cup=CUP_TARGET)), None)
        self.assertEqual(self.verdicts(receipt)["realism_vs_real_world"], "abstain")
        self.assertEqual(receipt["verdict"], "abstain")

    def test_chunked_without_seams_is_rejected_by_spec_validation(self):
        from datafarm.verify.spec import validate_spec

        problems = validate_spec(replace(self.spec, chunked=True))
        self.assertTrue(any(p.startswith("P3") for p in problems))

    def test_unmapped_task_role_is_rejected(self):
        from datafarm.verify.spec import validate_spec

        problems = validate_spec(replace(self.spec, task_ir_roles=("cup", "robot", "lid")))
        self.assertTrue(any("lid" in p for p in problems))

    def test_ood_requires_declared_axes(self):
        from datafarm.verify.spec import validate_spec

        spec = replace(self.spec, realism=replace(self.spec.realism, mode="ood"))
        self.assertTrue(any(p.startswith("P4") for p in validate_spec(spec)))

    def test_ood_lighting_ignores_global_colour_cast_but_keeps_entities(self):
        import numpy as np
        from datafarm.verify.verifier import run_checks

        spec = replace(self.spec, realism=replace(self.spec.realism, mode="ood",
                                                   perturbed_axes=("lighting",)))
        tinted = np.clip(make_frames(cup=CUP_TARGET, noise=3).astype(int) * np.array([1.0, 1.0, 0.7]), 0, 255)
        receipt = run_checks(spec, self.carrier, self.clip(tinted.astype(np.uint8)), self.real)
        realism = next(c for c in receipt["checks"] if c["name"] == "realism_vs_real_world")
        for regions in realism["metrics"]["generated"].values():
            for scores in regions.values():
                self.assertNotIn("color", scores)

    def test_semantic_check_abstains_without_model(self):
        import os
        from dataclasses import replace as dc_replace
        from datafarm.verify.checks import Context
        from datafarm.verify.semantic import _MODEL, check_semantic_unsupported

        saved = os.environ.pop("DATAFARM_DINOV2", None)
        _MODEL.clear()
        try:
            robot = dc_replace(self.spec.entities[1], checks=("missing", "semantic_unsupported"))
            ctx = Context(self.spec, self.carrier, self.clip(make_frames(cup=CUP_TARGET)))
            self.assertEqual(check_semantic_unsupported(ctx, robot).verdict, "abstain")
        finally:
            if saved is not None:
                os.environ["DATAFARM_DINOV2"] = saved

    def test_sub_check_panel_restriction(self):
        from datafarm.verify.spec import EntitySpec

        entity = EntitySpec("robot", (), (), checks=("missing", "unsupported:head"))
        self.assertEqual(entity.active_checks("head"), {"missing", "unsupported"})
        self.assertEqual(entity.active_checks("left_wrist"), {"missing"})


if __name__ == "__main__":
    unittest.main()
