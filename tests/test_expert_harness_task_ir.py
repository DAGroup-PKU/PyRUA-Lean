from __future__ import annotations

import copy
import unittest

from datafarm.expert.task_ir import validate_stage_graph


def hanging_ir() -> dict:
    return {
        "demand": "Hang the mug on the rack by its handle.",
        "roles": {"mug": {"binding": "self.mug", "kind": "object"}, "rack": {"binding": "self.rack", "kind": "support"}},
        "stage_graph": {
            "stages": [
                {"id": "S1_grasp", "skill": "grasp", "arm": "left", "role": "mug", "frame": "contact_point:0",
                 "exit": ["mug_lifted"], "guard": [], "recovery": "try contact points 1,2 then other approach distance"},
                {"id": "S2_handover", "skill": "handover", "arm": "both", "role": "mug", "frame": "functional_point:0",
                 "exit": ["right_holds"], "guard": ["mug_lifted"], "recovery": "re-grasp from middle pose"},
                {"id": "S3_hang", "skill": "place", "arm": "right", "role": "rack", "frame": "functional_point:0",
                 "exit": ["on_hook"], "guard": ["right_holds"], "recovery": "retry with align constraint"},
            ],
            "sync": [{"wait_for": "S1_grasp", "before": "S2_handover", "rule": "left holds mug at middle pose"}],
        },
        "predicates": {
            "mug_lifted": {"kind": "exit", "expr": "mug z - start z", "tolerance": 0.05},
            "right_holds": {"kind": "guard", "expr": "right gripper closed and touching mug", "tolerance": "grip<0.2"},
            "on_hook": {"kind": "terminal", "expr": "|handle_xy - hook_mid_xy|", "tolerance": 0.02, "judge_criterion": "c1"},
        },
        "forbidden_shortcuts": ["no object pose writes", "no judge access"],
        "skill_evidence": [{"skill": "grasp", "evidence": "dev rollouts 20/20"}],
        "strategy_space": [{"param": "grasp_cp", "values": [0, 1], "reason": "handle orientation varies"}],
    }


class TaskIRStageGraphTests(unittest.TestCase):
    def test_valid_graph_and_judge_coverage(self) -> None:
        self.assertEqual(validate_stage_graph(hanging_ir(), judge_criteria=["c1"]), [])

    def test_uncovered_judge_criterion_and_unknown_refs_are_reported(self) -> None:
        ir = hanging_ir()
        ir["stage_graph"]["stages"][2]["exit"] = ["on_hook", "missing_pred"]
        problems = validate_stage_graph(ir, judge_criteria=["c1", "c2"])
        self.assertTrue(any("missing_pred" in p for p in problems))
        self.assertTrue(any("'c2'" in p and "not covered" in p for p in problems))

    def test_stage_without_exit_or_recovery_and_bad_arm(self) -> None:
        ir = hanging_ir()
        st = ir["stage_graph"]["stages"][0]
        st["exit"], st["recovery"], st["arm"] = [], "", "middle"
        problems = validate_stage_graph(ir)
        for needle in ("exit predicate", "recovery", "arm must be"):
            self.assertTrue(any(needle in p for p in problems), needle)

    def test_terminal_predicate_must_name_criterion_and_tolerance(self) -> None:
        ir = hanging_ir()
        del ir["predicates"]["on_hook"]["judge_criterion"]
        del ir["predicates"]["mug_lifted"]["tolerance"]
        problems = validate_stage_graph(ir)
        self.assertTrue(any("judge criterion" in p for p in problems))
        self.assertTrue(any("tolerance" in p for p in problems))

    def test_sync_order_and_missing_keys(self) -> None:
        ir = hanging_ir()
        ir["stage_graph"]["sync"] = [{"wait_for": "S3_hang", "before": "S1_grasp", "rule": "x"}]
        self.assertTrue(any("contradicts stage order" in p for p in validate_stage_graph(ir)))
        bad = copy.deepcopy(ir)
        del bad["forbidden_shortcuts"]
        self.assertEqual(validate_stage_graph(bad), ["missing required key 'forbidden_shortcuts'"])

if __name__ == "__main__":
    unittest.main()
