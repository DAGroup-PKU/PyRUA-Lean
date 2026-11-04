import json
import sys
import pytest
from datafarm.simulation.usdcraft import plan, run, collect, read


def setup(tmp_path):
    root=tmp_path/'upstream';(root/'usdcraft').mkdir(parents=True)
    (root/'usdcraft/cli.py').write_text('# fixture')
    (root/'pyproject.toml').write_text('# fixture')
    image=tmp_path/'real.png';image.write_bytes(b'fixture only, upstream decodes actual images')
    config=tmp_path/'config.json';config.write_text(json.dumps({'checkout':str(root),'model':'model-x','reasoning':'low','launcher':[sys.executable,str(root/'fake.py')]}))
    obs=tmp_path/'observation.json';obs.write_text(json.dumps({'object_id':'door','image':str(image),'description':'Observed hinged door; dimensions inferred'}))
    return root,config,obs,tmp_path/'build'


def exports(root):
    record=root/'data/records/r1';record.mkdir(parents=True)
    model=record/'model.py';model.write_text('raise RuntimeError("must not execute")')
    mat=root/'data/cache/record_materialization/r1';mat.mkdir(parents=True)
    (mat/'mesh.stl').write_text('fixture')
    (mat/'model.urdf').write_text('<robot name="door"><link name="base"><visual><geometry><mesh filename="mesh.stl"/></geometry></visual></link><joint name="hinge" type="revolute"><axis xyz="0 0 1"/><limit lower="0" upper="1.5"/></joint></robot>')
    return {'status':'generation_ready','record_id':'r1','model_path':str(model),'provider':'codex-cli'}


def test_native_routes_and_mesh_requirement(tmp_path):
    root,c,o,b=setup(tmp_path);p=plan(c,o,b)
    assert p['mode']=='generation' and '--direct' in p['argv'] and p['cost']['amount_usd'] is None
    mesh=tmp_path/'mesh.glb';mesh.write_bytes(b'fixture');v=read(o);v.update(mesh=str(mesh),front_axis='-Y',meters_per_unit=.001);o.write_text(json.dumps(v))
    p=plan(c,o,tmp_path/'recon');assert p['mode']=='reconstruction' and '--front-axis=-Y' in p['argv']


def test_plan_reports_the_upstream_commit(tmp_path):
    root,c,o,b=setup(tmp_path);p=plan(c,o,b)
    assert p['upstream']['tested'] is False and len(p['upstream']['tested_commit'])==40


def test_checkout_can_use_the_binding(tmp_path, monkeypatch):
    root,c,o,b=setup(tmp_path);monkeypatch.setenv('DATAFARM_USDCRAFT',str(root))
    cfg=read(c);cfg['checkout']='${USDCRAFT}';c.write_text(json.dumps(cfg))
    assert plan(c,o,b)['cwd']==str(root.resolve())


def test_changed_input_never_submits(tmp_path):
    root,c,o,b=setup(tmp_path);plan(c,o,b);o.write_text('{}')
    with pytest.raises(ValueError,match='Input changed'):run(b)
    assert not (b/'execution.json').exists()


def test_execute_bind_and_no_duplicate(tmp_path):
    root,c,o,b=setup(tmp_path);payload=exports(root)
    (root/'fake.py').write_text('import json\nfrom pathlib import Path\np=Path("count");p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint('+repr(json.dumps(payload))+')\n')
    plan(c,o,b);assert run(b)['status']=='exported-unqualified';run(b)
    assert (root/'count').read_text()=='1'
    asset=read(b/'asset.json');assert asset['qualification']=='unverified'
    assert any(x['path'].endswith('mesh.stl') for x in asset['artifacts'])


def test_failure_not_retried(tmp_path):
    root,c,o,b=setup(tmp_path);(root/'fake.py').write_text('raise SystemExit(2)')
    plan(c,o,b);assert run(b)['status']=='failed';assert run(b)['exit_code']==2
    assert not (b/'asset.json').exists()


def test_missing_dependency_and_external_model_rejected(tmp_path):
    root,c,o,b=setup(tmp_path);p=plan(c,o,b);payload=exports(root)
    (root/'data/cache/record_materialization/r1/mesh.stl').unlink()
    with pytest.raises(FileNotFoundError):collect(p,payload)
    payload['model_path']=str(o)
    with pytest.raises(ValueError,match='outside canonical'):collect(p,payload)


def test_timeout_preserves_unknown_cost(tmp_path):
    root,c,o,b=setup(tmp_path);cfg=read(c);cfg['timeout_seconds']=.05;c.write_text(json.dumps(cfg));(root/'fake.py').write_text('import time;time.sleep(20)')
    plan(c,o,b)
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):run(b)
    s=read(b/'execution.json');assert s['status']=='failed' and s['cost']['amount_usd'] is None


def test_concurrent_worker_and_prior_unknown_not_resubmitted(tmp_path):
    import fcntl
    root,c,o,b=setup(tmp_path);plan(c,o,b)
    with (b/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):run(b)
    (b/'execution.json').write_text(json.dumps({'status':'running','pid':99999999}))
    assert run(b)['status']=='running'
    assert not (b/'stdout.json').exists()


def test_mesh_escape_and_ready_state_not_trusted(tmp_path):
    root,c,o,b=setup(tmp_path);p=plan(c,o,b);payload=exports(root)
    payload['status']='complete'
    with pytest.raises(ValueError,match='ready native'):collect(p,payload)
    payload['status']='generation_ready'
    mesh=root/'data/cache/record_materialization/r1/mesh.stl';mesh.unlink();mesh.symlink_to(o)
    with pytest.raises(ValueError,match='external symlink'):collect(p,payload)


def test_relative_model_path_resolves_against_upstream_root(tmp_path, monkeypatch):
    root,c,o,b=setup(tmp_path);p=plan(c,o,b);payload=exports(root)
    payload['model_path']='data/records/r1/model.py'
    monkeypatch.chdir(tmp_path)  # a process cwd outside the upstream checkout
    assert collect(p,payload)['record_id']=='r1'
