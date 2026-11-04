import json
import pytest
from datafarm.agent.native_codex import save
from datafarm.agent.progress import publish, refresh_external_jobs
from datafarm.agent.cost_budget import TaskBudget
from datafarm.simulation.usdcraft import run_for_task, plan
from test_usdcraft_bridge import setup, exports


def test_publication_does_not_count_claims_or_heartbeats(tmp_path):
    work=tmp_path/'work';(work/'.runtime').mkdir(parents=True)
    output=tmp_path/'status.json'
    save(work/'.runtime/status-output.json',{'path':str(output),'task_id':'one'})
    save(work/'state.json',{'status':'working','phase':'contact-probe','completed_episodes':5})
    publish(work,{'status':'driving'})
    first=json.loads(output.read_text())['tasks']['one']
    publish(work,{'status':'driving'})
    second=json.loads(output.read_text())['tasks']['one']
    assert first['last_substantive_at']==second['last_substantive_at']
    assert second['completed_episodes']==0 and second['reported_completed_episodes']==5
    save(work/'.runtime/control.json',{'last_result':{'status':'complete','valid':True,'episodes':5}})
    publish(work,{'status':'complete'})
    assert json.loads(output.read_text())['tasks']['one']['completed_episodes']==5


def test_external_monitor_exact_id_unknown_and_failure(tmp_path):
    state={'status':'working','external_jobs':[{'id':'original','status':'running','status_file':'job.json'}]}
    save(tmp_path/'state.json',state)
    assert refresh_external_jobs(tmp_path)==state
    save(tmp_path/'job.json',{'id':'wrong','status':'complete'})
    assert refresh_external_jobs(tmp_path)==state
    save(tmp_path/'job.json',{'id':'original','status':'finished','exit_code':2})
    assert refresh_external_jobs(tmp_path)['external_jobs'][0]['status']=='failed'
    save(tmp_path/'job.json',{'id':'original','status':'exported-unqualified','exit_code':0})
    assert refresh_external_jobs(tmp_path)['external_jobs'][0]['status']=='finished'


def task_policy(work,mode):
    save(work/'cost-policy.json',{'schema':'datafarm.task-cost-policy/2','mode':mode,'target_usd':30,
        'limit_usd':50,'scope':'all-model-work-through-white-data','on_excess':'warn'})
    save(work/'state.json',{'status':'working','external_jobs':[]})
    (work/'sessions').mkdir()


def test_task_bound_asset_unknown_cost_and_no_duplicate(tmp_path):
    root,c,o,b=setup(tmp_path);payload=exports(root)
    (root/'fake.py').write_text('print('+repr(json.dumps(payload))+')')
    task_policy(tmp_path,'advisory');plan(c,o,b)
    assert run_for_task(b,tmp_path)['status']=='exported-unqualified'
    assert TaskBudget(tmp_path).status()['total_usd'] is None
    assert TaskBudget(tmp_path).status()['unknown_sessions']==1
    assert len(refresh_external_jobs(tmp_path)['external_jobs'])==1
    assert run_for_task(b,tmp_path)['status']=='exported-unqualified'
    assert TaskBudget(tmp_path).status()['sessions']==1


def test_task_bound_asset_enforced_refused_before_launch(tmp_path):
    root,c,o,b=setup(tmp_path);plan(c,o,b);task_policy(tmp_path,'enforced')
    with pytest.raises(ValueError,match='enforced budget'):run_for_task(b,tmp_path)
    assert not (b/'execution.json').exists()
    assert TaskBudget(tmp_path).status()['sessions']==0


def test_concurrent_task_build_leaves_status_and_budget_untouched(tmp_path):
    import fcntl
    from datafarm.simulation.usdcraft import BuildLocked
    root,c,o,b=setup(tmp_path);plan(c,o,b);task_policy(tmp_path,'advisory')
    save(tmp_path/'state.json',{'status':'working'})
    with (b/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BuildLocked):run_for_task(b,tmp_path)
    assert 'external_jobs' not in json.loads((tmp_path/'state.json').read_text())
    assert not (b/'job-status.json').exists()
    assert TaskBudget(tmp_path).status()['sessions']==0
