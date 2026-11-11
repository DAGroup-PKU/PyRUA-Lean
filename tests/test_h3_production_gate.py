"""h3_gate.py: two prefixes, then the fixed canary, then the rest; verdicts only from the blind reviews."""
import json
import subprocess
import sys

import pytest
from test_h3_control_hybrid import HYBRID, clean_env, import_hybrid

CANARY = [f'c{k}' for k in range(8)]


def final_package(run, episode):
    d = run / 'pkgs/exposure-lock-r1v-vlm/task-a' / f'{episode}-sim2real'
    d.mkdir(parents=True)
    (d / 'package.json').write_text(json.dumps({'task': 'task-a', 'episode': episode}))
    (d / 'assembled.json').write_text('{}')
    return d


def review(run, package, verdict, kinds=()):
    """What h3_review_prep.py and an independent reviewer leave behind."""
    mapping = run / 'reviews/.mapping.json'
    m = json.loads(mapping.read_text()) if mapping.exists() else {}
    rid = f'r{len(m):08x}'
    m[rid] = {'subject': str(package.resolve())}
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text(json.dumps(m))
    (run / 'reviews' / rid / 'review').mkdir(parents=True)
    if verdict:
        (run / 'reviews' / rid / 'review/review.json').write_text(json.dumps(
            {'schema': 'h3.review/1', 'verdict': verdict, 'scores': {'realism': 4},
             'blocking': [{'kind': k, 'view': 'head'} for k in kinds]}))


@pytest.fixture
def gate(tmp_path):
    g = import_hybrid('h3_gate')
    run = tmp_path / 'run'
    g.init(run, 'g1', 'r1', 'task-a', ['p0', 'p1'], CANARY, ['x0', 'x1'])
    return g, run


def done(g, run, episode, verdict, kinds=()):
    pkg = final_package(run, episode)
    g.record(run, 'g1', episode, pkg, None)
    review(run, pkg, verdict, kinds)


def test_gate_releases_prefix_then_canary_then_rest(gate):
    g, run = gate
    assert g.status(run, 'g1')['allowed'] == ['p0', 'p1']
    done(g, run, 'p0', 'accept')
    done(g, run, 'p1', None)               # reviewed job exists, review not written yet
    st = g.status(run, 'g1')
    assert st['stage'] == 'prefix' and st['episodes']['p1']['verdict'] == 'pending'
    (run / 'reviews' / st['episodes']['p1']['review_id'] / 'review/review.json').write_text(json.dumps(
        {'verdict': 'accept', 'scores': {}}))
    assert g.status(run, 'g1')['allowed'] == ['p0', 'p1', *CANARY]
    for k, ep in enumerate(CANARY[:6]):
        done(g, run, ep, 'accept')
    done(g, run, 'c6', 'reject', ['extra object'])
    assert g.status(run, 'g1')['stage'] == 'canary'
    g.record(run, 'g1', 'c7', None, 'provider error after 3 tries')
    st = g.status(run, 'g1')
    assert st['stage'] == 'open' and st['canary_accepted'] == 6 and st['canary_needed'] == 6
    assert st['allowed'][-2:] == ['x0', 'x1'] and st['episodes']['c7']['verdict'] == 'failed'


def test_gate_stops_on_a_rejected_prefix(gate):
    g, run = gate
    done(g, run, 'p0', 'reject', ['white arm'])
    st = g.status(run, 'g1')
    assert st['stage'] == 'stopped' and st['allowed'] == [] and 'prefix' in st['reason']


def test_gate_stops_when_the_canary_can_no_longer_pass(gate):
    g, run = gate
    done(g, run, 'p0', 'accept')
    done(g, run, 'p1', 'accept')
    done(g, run, 'c0', 'reject')
    done(g, run, 'c1', 'reject')
    assert g.status(run, 'g1')['stage'] == 'canary'   # 6 of the 6 still open can reach 6/8
    done(g, run, 'c2', 'reject')
    st = g.status(run, 'g1')
    assert st['stage'] == 'stopped' and 'fewer than 6/8' in st['reason']


def test_gate_stops_on_a_common_blocking_kind(gate):
    g, run = gate
    done(g, run, 'p0', 'accept')
    done(g, run, 'p1', 'accept')
    for ep in CANARY[:4]:
        done(g, run, ep, 'accept', ['cable drawn as stick'])   # accepted, but the reviewer still names a blocker
    st = g.status(run, 'g1')
    assert st['stage'] == 'stopped' and st['common_blocking_kinds'] == ['cable drawn as stick']


def test_gate_lists_and_results_are_fixed(gate):
    g, run = gate
    with pytest.raises(SystemExit, match='exists'):
        g.init(run, 'g1', 'r1', 'task-a', ['p0', 'p1'], CANARY, [])
    with pytest.raises(SystemExit, match='at least 8'):
        g.init(run, 'g2', 'r1', 'task-a', ['p0', 'p1'], CANARY[:7], [])
    done(g, run, 'p0', 'reject')
    with pytest.raises(SystemExit, match='already recorded'):
        g.record(run, 'g1', 'p0', None, 'retry')
    other = final_package(run, 'zz')
    with pytest.raises(SystemExit, match='not in gate'):
        g.record(run, 'g1', 'zz', other, None)


def test_enqueue_refuses_episodes_the_gate_has_not_released(tmp_path):
    g = import_hybrid('h3_gate')
    run = tmp_path / 'run'
    (run / 'tasks/task-a').mkdir(parents=True)
    (run / 'tasks/task-a/bundle.json').write_text(json.dumps({'episodes': [{'id': 'p0'}, {'id': 'c0'}]}))
    recipe = run / 'recipes/r1.json'
    recipe.parent.mkdir()
    recipe.write_text(json.dumps({'name': 'r1', 'backend': 'control-hybrid'}))
    g.init(run, 'g1', 'r1', 'task-a', ['p0', 'p1'], CANARY, [])
    g.init(run, 'g-other', 'r0', 'task-a', ['p0', 'p1'], CANARY, [])
    tool = [sys.executable, str(HYBRID / 'h3_enqueue.py'), '--run-dir', str(run), '--recipe', str(recipe),
            '--tasks', 'task-a', '--modes', 'sim2real', '--dry-run']
    r = subprocess.run(tool + ['--gate', 'g1'], capture_output=True, text=True, env=clean_env())
    assert r.returncode != 0 and "not released: ['c0']" in r.stderr
    r = subprocess.run(tool + ['--gate', 'g-other', '--episodes', 'p0'], capture_output=True, text=True, env=clean_env())
    assert r.returncode != 0 and 'recipe r0' in r.stderr
    r = subprocess.run(tool + ['--gate', 'g1', '--episodes', 'p0'], capture_output=True, text=True, env=clean_env())
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ['p0-sim2real', 'task-a']
