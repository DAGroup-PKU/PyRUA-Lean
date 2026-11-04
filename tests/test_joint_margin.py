import pytest

def test_audit_tracks_phase_dwell_and_nominal_violation(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'tools/audit_joint_margin_trace.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    q=[0,1,-1,0,0,0];bad=q.copy();bad[4]=1.1
    p=tmp_path/'t.json';p.write_text(json.dumps({'binding':{'sample_hz':10,'physical_joint_limits_rad':[[-1,1]]*6,'policy':{'target_joint': 4, 'target_margin_deg': 5}},'records':[{'step':i,'phase':'a' if i<2 else 'b','left':v,'right':q,'commands':{}} for i,v in enumerate([q,bad,bad,q])]}))
    r=m.audit(p);assert not r['joint_target_met'];assert r['arms']['left']['actual']['beyond_limits_frames'][4]==2
    assert r['arms']['left']['actual']['bands']['1']['longest_seconds'][4]==pytest.approx(.2)
    assert r['arms']['left']['phases']['b']['bands']['1']['longest_seconds'][4]==pytest.approx(.1)

@pytest.mark.parametrize("defect", ["actual_j3", "command_j2", "missing_command", "clock_gap", "none"])
def test_full_joint_verdict_is_not_one_joint_verdict(tmp_path, defect):
    import json
    from datafarm.simulation.joint_margin_audit import audit
    q = [0.] * 6
    records = [{"step": i, "phase": "home", "left": q.copy(), "right": q.copy(),
                "commands": {a: {"qpos_rad": q.copy()} for a in ("left", "right")}}
               for i in range(3)]
    if defect == "actual_j3": records[0]["left"][2] = 1.000001
    if defect == "command_j2": records[1]["commands"]["right"]["qpos_rad"][1] = -1.000001
    if defect == "missing_command": del records[1]["commands"]["left"]
    if defect == "clock_gap": records[-1]["step"] = 3
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"binding": {"sample_hz": 250, "physical_joint_limits_rad": [[-1,1]]*6,
         "policy": {"target_joint": 4, "target_margin_deg": 5}}, "records": records}))
    result = audit(p)
    assert result["joint_target_met"]
    assert result["recorded_trace_passed"] == (defect == "none")

