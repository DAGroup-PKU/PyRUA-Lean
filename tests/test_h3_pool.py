"""h3_pool.py (resident + elastic workers, scheduler and local mode) and h3_enqueue.py build ordering and slots."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest
from test_h3_control_hybrid import HYBRID, clean_env, import_hybrid

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='pid and /proc checks are Linux')


def write_exe(path, body):
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(0o755)
    return path


def pool_run(tmp_path, cfg):
    run = tmp_path / 'run'
    (run / 'queue/pending/control-turbo').mkdir(parents=True)
    (run / 'queue/running').mkdir()
    (run / 'queue/pool.json').write_text(json.dumps({'schema': 'h3.worker-pool/1', **cfg}))
    for d in ('vx', 'models', 'lora'):
        (tmp_path / d).mkdir()
    return run


@pytest.fixture
def pool(monkeypatch, tmp_path):
    for key in ('H3_RUN', 'GPU_SCHEDULER', 'H3_PYTHON', 'H3_VIDEOX_FUN', 'H3_MODELS', 'H3_TURBO_LORA', 'H3_LOCAL_GPU'):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv('DATAFARM_' + key, raising=False)
    monkeypatch.setenv('H3_VIDEOX_FUN', str(tmp_path / 'vx'))
    monkeypatch.setenv('H3_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('H3_TURBO_LORA', str(tmp_path / 'lora'))
    mod = import_hybrid('h3_pool')
    mod.PROCS.clear()
    yield mod
    for p in mod.PROCS.values():
        if p.poll() is None:
            os.killpg(p.pid, 9)
            p.wait()


def ticket(run, name):
    (run / 'queue/pending/control-turbo' / f'{name}.json').write_text(json.dumps({'profile': 'control-turbo'}))


def test_pool_config_fails_closed(pool, tmp_path):
    run = pool_run(tmp_path, {'resident': 2, 'max_workers': 1})
    with pytest.raises(SystemExit, match='max_workers'):
        pool.config(run)
    (run / 'queue/pool.json').write_text(json.dumps({'schema': 'h3.worker-pool/1', 'local_gpus': [0]}))
    with pytest.raises(SystemExit, match='local_host'):
        pool.config(run)
    (run / 'queue/pool.json').write_text(json.dumps({'resident': 1}))
    with pytest.raises(SystemExit, match='schema'):
        pool.config(run)
    (run / 'queue/pool.json').unlink()
    with pytest.raises(SystemExit, match='missing'):
        pool.config(run)


def test_scheduler_pool_keeps_residents_and_scales_with_the_queue(pool, monkeypatch, tmp_path):
    run = pool_run(tmp_path, {'resident': 1, 'max_workers': 3})
    (run / 'queue/ALLOWED_HOSTS').write_text('node-a\n')
    calls, live, mode = tmp_path / 'calls', tmp_path / 'live', tmp_path / 'mode'
    live.write_text('')
    mode.write_text('ok')
    sched = write_exe(tmp_path / 'sched', f'''
if [ "$1" = leases ]; then [ "$(cat {mode})" = broken ] && exit 1; cat {live}; exit 0; fi
[ "$(cat {mode})" = busy ] && {{ echo "no gpu" >&2; exit 2; }}
n=$(wc -l < {calls}); echo "$*" >> {calls}
id=h3-control-turbo-20261110-00000$n-ab; echo "$id 1 node-a x 10$n" >> {live}
echo id=$id; echo host=node-a; echo log=/tmp/$id.log
''')
    calls.write_text('')
    monkeypatch.setenv('GPU_SCHEDULER', str(sched))
    st, log = {'workers': []}, []

    assert pool.cycle(run, st, log.append) == 0.0           # resident launched
    assert pool.cycle(run, st, log.append) == 30.0          # alive, queue empty: nothing more
    submits = calls.read_text().splitlines()
    assert len(submits) == 1 and '--host node-a' in submits[0] and '--min-free-mb 172000' in submits[0]
    assert '--idle-exit 43200' in submits[0] and st['workers'][0]['resident'] is True
    for k in range(5):
        ticket(run, f't{k}')
    assert pool.cycle(run, st, log.append) == 0.0           # elastic up to max_workers
    assert pool.cycle(run, st, log.append) == 0.0
    assert pool.cycle(run, st, log.append) == 30.0
    submits = calls.read_text().splitlines()
    assert len(submits) == 3 and all('--idle-exit 1200' in s for s in submits[1:])
    assert st['alive'] == 3 and st['want'] == 3

    live.write_text('')                                     # every lease gone; the scheduler has no GPU
    mode.write_text('busy')
    assert pool.cycle(run, st, log.append) == 30.0
    assert st['alive'] == 0 and 'no gpu' in st['last_error'] and log[-1].startswith('launch resident: no GPU free')
    st['backoff_until'] = 0
    mode.write_text('broken')                               # unknown lease table: launch nothing
    assert pool.cycle(run, st, log.append) == 30.0
    assert 'unknown' in st['last_error'] and len(calls.read_text().splitlines()) == 3


def test_local_pool_starts_workers_on_free_listed_gpus_and_requeues_dead_claims(pool, monkeypatch, tmp_path):
    host = pool.HOST
    run = pool_run(tmp_path, {'resident': 1, 'max_workers': 2, 'local_gpus': [0, 1, 2], 'local_host': host,
                              'min_free_mb': 1000})
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    write_exe(bindir / 'nvidia-smi', 'echo "0, 500"\necho "1, 90000"\necho "2, 90000"\n')
    fake_py = write_exe(tmp_path / 'fake-python', 'echo "gpu=$CUDA_VISIBLE_DEVICES args=$*"\nexec sleep 60\n')
    monkeypatch.setenv('PATH', f'{bindir}:{os.environ["PATH"]}')
    monkeypatch.setenv('H3_PYTHON', str(fake_py))
    monkeypatch.setenv('GPU_SCHEDULER', str(tmp_path / 'must-not-be-called'))
    st, log = {'workers': []}, []

    assert pool.cycle(run, st, log.append) == 0.0
    ticket(run, 't0')
    ticket(run, 't1')
    assert pool.cycle(run, st, log.append) == 0.0
    assert pool.cycle(run, st, log.append) == 30.0
    assert [w['gpu'] for w in st['workers']] == [1, 2]       # GPU 0 lacks free memory
    deadline = time.time() + 20
    while time.time() < deadline and 'args=' not in (run / 'queue/logs').joinpath(
            st['workers'][1]['id'] + '.log').read_text():
        time.sleep(0.2)
    out = (run / 'queue/logs' / f"{st['workers'][1]['id']}.log").read_text()
    assert 'gpu=2 ' in out and '--queue' in out and '--idle-exit 1200' in out
    assert pool.PROCS[st['workers'][0]['id']].pid == st['workers'][0]['pid']

    dead = subprocess.Popen(['true'])
    dead.wait()
    claimed = run / 'queue/running' / f'001-x@{host}-{dead.pid}-control-turbo.json'
    claimed.write_text(json.dumps({'profile': 'control-turbo'}))
    other = run / 'queue/running' / f'001-y@another-host-{dead.pid}-control-turbo.json'
    other.write_text('{}')
    os.killpg(st['workers'][0]['pid'], 9)
    pool.PROCS[st['workers'][0]['id']].wait()
    pool.cycle(run, st, log.append)
    assert not claimed.exists() and (run / 'queue/pending/control-turbo/001-x.json').is_file() and other.exists()
    assert st['workers'][0]['alive'] is False and st['workers'][-1]['gpu'] == 1   # resident relaunched on the free GPU

    (run / 'queue/pool.json').write_text(json.dumps({'schema': 'h3.worker-pool/1', 'local_gpus': [0],
                                                     'local_host': 'some-other-node'}))
    with pytest.raises(SystemExit, match='refusing'):
        pool.cycle(run, st, log.append)


def test_pool_is_single_instance_and_stops_on_the_stop_file(pool, tmp_path):
    run = pool_run(tmp_path, {'resident': 0})
    (run / 'queue/STOP').write_text('')
    assert pool.run_pool(run) == 0
    assert 'stopped_at' in json.loads((run / 'queue/pool-state.json').read_text())
    lock = open(run / 'queue/.pool.lock', 'a+')
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    r = subprocess.run([sys.executable, str(HYBRID / 'h3_pool.py'), 'run', '--run-dir', str(run)],
                       capture_output=True, text=True, env=clean_env(), timeout=60)
    assert r.returncode == 0 and 'already running' in r.stdout


def test_build_slots_are_shared_across_holders(tmp_path):
    enqueue = import_hybrid('h3_enqueue')
    got = []

    def second():
        with enqueue.build_slot(tmp_path, 1, poll=0.05):
            got.append(time.time())

    with enqueue.build_slot(tmp_path, 1):
        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.4)
        assert got == []
        released = time.time()
    t.join(5)
    assert got and got[0] >= released


@pytest.mark.skipif(not shutil.which('nice'), reason='needs nice')
def test_enqueue_builds_the_first_seed_first_within_the_build_slots(tmp_path):
    run = tmp_path / 'run'
    (run / 'tasks/task-a').mkdir(parents=True)
    (run / 'tasks/task-a/bundle.json').write_text(json.dumps({'episodes': [{'id': 'e0'}, {'id': 'e1'}]}))
    recipe = run / 'recipes/r1.json'
    recipe.parent.mkdir()
    recipe.write_text(json.dumps({'name': 'r1', 'backend': 'control-hybrid'}))
    events = tmp_path / 'events'
    fake_py = write_exe(tmp_path / 'fake-python', f'''
while [ $# -gt 0 ]; do case "$1" in --seed-offset) s=$2;; --out) out=$2;; esac; shift; done
echo "start $s $(date +%s.%N)" >> {events}; sleep 0.3
mkdir -p "$out"; printf '{{"package": "%s", "profile": "control-turbo"}}' "$out" > "$out/ticket.json"
echo "end $s $(date +%s.%N) $(ps -o ni= -p $$ | tr -d ' ')" >> {events}
''')
    r = subprocess.run([sys.executable, str(HYBRID / 'h3_enqueue.py'), '--run-dir', str(run), '--recipe', str(recipe),
                        '--tasks', 'task-a', '--modes', 'sim2real', '--seed-offset', '0', '1', '2',
                        '--build-slots', '2'], capture_output=True, text=True,
                       env=clean_env(H3_PYTHON=str(fake_py)), timeout=120)
    assert r.returncode == 0, r.stderr
    rows = [line.split() for line in events.read_text().splitlines()]
    starts = {(s, float(t)) for kind, s, t, *_ in rows if kind == 'start'}
    ends = [(s, float(t)) for kind, s, t, *_ in rows if kind == 'end']
    assert len(ends) == 6 and all(int(r[3]) >= 10 for r in rows if r[0] == 'end')
    assert max(t for s, t in ends if s == '0') <= min(t for s, t in starts if s != '0')
    timeline = sorted([(t, 1) for _, t in starts] + [(t, -1) for _, t in ends])
    level = peak = 0
    for _, d in timeline:
        level += d
        peak = max(peak, level)
    assert peak <= 2
    queued = sorted(p.name for p in (run / 'queue/pending/control-turbo').glob('*.json'))
    assert len(queued) == 6 and any(n.endswith('e1-sim2real-s2.json') for n in queued)
