"""A verifier spec is calibrated only on the full suite with nothing silently dropped."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrate_render_verifier", ROOT / "tools/calibrate_render_verifier.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

SUITE = ["entity_delete", "view_swap", "colour_shift"]
POSITIVE = {"name": "positive_control", "verdict": "pass"}


def caught(name):
    return {"name": name, "caught": True}


def test_full_suite_all_caught_is_calibrated():
    result = tool.summarize([POSITIVE, *map(caught, SUITE)], suite=SUITE, partial=False, waivers={})
    assert result["calibrated"] and result["caught"] == 3 and result["total"] == 3


def test_unconstructible_mutation_needs_an_explicit_waiver():
    results = [POSITIVE, caught("entity_delete"), caught("colour_shift"),
               {"name": "view_swap", "status": "not_constructible"}]
    refused = tool.summarize(results, suite=SUITE, partial=False, waivers={})
    assert not refused["calibrated"] and refused["unwaived_not_constructible"] == ["view_swap"]
    waived = tool.summarize(results, suite=SUITE, partial=False, waivers={"view_swap": "single view spec"})
    assert waived["calibrated"] and waived["constructed"] == 2 and waived["total"] == 3


def test_partial_run_missed_defect_or_failed_control_never_calibrates():
    assert not tool.summarize([POSITIVE, caught("entity_delete")], suite=SUITE, partial=True,
                              waivers={})["calibrated"]
    missed = [POSITIVE, caught("entity_delete"), caught("view_swap"), {"name": "colour_shift", "caught": False}]
    assert tool.summarize(missed, suite=SUITE, partial=False, waivers={})["missed"] == ["colour_shift"]
    failed_control = [{"name": "positive_control", "verdict": "fail"}, *map(caught, SUITE)]
    assert not tool.summarize(failed_control, suite=SUITE, partial=False, waivers={})["calibrated"]
    with pytest.raises(ValueError, match="unknown counterfactuals"):
        tool.summarize([POSITIVE], suite=SUITE, partial=False, waivers={"typo": "x"})
