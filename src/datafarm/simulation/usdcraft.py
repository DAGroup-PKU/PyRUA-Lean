"""Optional USDCraft CLI bridge. No upstream imports or credentials are required to plan."""
from __future__ import annotations
import fcntl
import json
import math
import os
from datafarm.common import atomic_write_json, sha256_file
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write_json(path, value, sort_keys=False, mode=0o644, fsync=False)


def evidence(path):
    p = Path(path).expanduser().resolve(strict=True)
    if not p.is_file():
        raise ValueError(f'Not a file: {p}')
    return {'path': str(p), 'sha256': sha256_file(p)}


def plan(config, observation, output):
    cfg, obs = read(config), read(observation)
    root = Path(cfg['checkout']).expanduser().resolve(strict=True)
    if not (root / 'usdcraft/cli.py').is_file():
        raise ValueError('checkout must be an extracted USDCraft source tree')
    model, reasoning = cfg['model'], cfg['reasoning']
    if not all(isinstance(x, str) and x for x in (model, reasoning)):
        raise ValueError('model and reasoning must both be explicit')
    launcher = cfg.get('launcher', ['uv', 'run', '--frozen', 'usdcraft'])
    if not isinstance(launcher, list) or not launcher or not all(isinstance(x, str) and x for x in launcher):
        raise ValueError('launcher must be a nonempty argv array')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', obs['object_id']):
        raise ValueError('object_id must be a lowercase slug')
    if not isinstance(obs.get('description'), str) or not obs['description'].strip():
        raise ValueError('description must state observed identity, function and uncertainties')
    refs = [evidence(obs['image'])]
    mode = 'reconstruction' if obs.get('mesh') else 'generation'
    argv = launcher + [mode + '-codex']
    if mode == 'reconstruction':
        refs.append(evidence(obs['mesh']))
        argv += [refs[-1]['path'], '--prompt', obs['description']]
        for key in ('up_axis', 'front_axis', 'meters_per_unit'):
            if key in obs:
                if key == 'meters_per_unit' and (not math.isfinite(float(obs[key])) or float(obs[key]) <= 0):
                    raise ValueError('meters_per_unit must be finite and positive')
                if key != 'meters_per_unit' and obs[key] not in ('+X','-X','+Y','-Y','+Z','-Z'):
                    raise ValueError('invalid source axis')
                argv += ['--' + key.replace('_', '-') + '=' + str(obs[key])]
    else:
        argv += [obs['description']]
    argv += ['--image', refs[0]['path'], '--model', model, '--reasoning', reasoning,
             '--label', obs['object_id'], '--repo-root', str(root), '--keep-workspace', '--json', '--direct']
    # Hash all additional evidence but do not pretend the single-image upstream CLI consumed it.
    refs += [evidence(p) for p in obs.get('context_files', [])]
    refs += [evidence(config), evidence(observation), evidence(root / 'usdcraft/cli.py'), evidence(root / 'pyproject.toml')]
    result = {'schema':'datafarm.usdcraft-plan/1', 'object_id':obs['object_id'], 'mode':mode,
              'cwd':str(root), 'argv':argv, 'inputs':refs, 'observation':obs,
              'timeout_seconds':cfg.get('timeout_seconds'),
              'qualification':'unverified', 'cost':{'basis':'unknown','amount_usd':None},
              'image_semantics':'inferred-geometry' if mode == 'generation' else 'mesh-and-photo-reference'}
    timeout = result['timeout_seconds']
    if timeout is not None and (not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout_seconds must be null or positive finite seconds')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', result)
    return result


def check_inputs(p):
    for ref in p['inputs']:
        if evidence(ref['path']) != ref:
            raise ValueError('Input changed; create a new versioned plan: ' + ref['path'])


def collect(p, payload):
    """Bind export bytes without executing model.py or claiming simulator qualification."""
    if payload.get('status') != p['mode'] + '_ready' or payload.get('provider') != 'codex-cli':
        raise ValueError('USDCraft did not return a ready native Codex record')
    rid = payload['record_id']
    if not isinstance(rid,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',rid):
        raise ValueError('invalid record ID')
    root = Path(p['cwd']).resolve()
    record = (root/'data/records'/rid).resolve()
    material = (root/'data/cache/record_materialization'/rid).resolve()
    if not record.is_relative_to(root/'data/records') or not material.is_relative_to(root/'data/cache/record_materialization'):
        raise ValueError('record escapes data root')
    model = Path(payload['model_path'])
    # Upstream reports paths relative to its own data root, not to this process.
    model = (model if model.is_absolute() else root / model).resolve(strict=True)
    if not model.is_relative_to(record):
        raise ValueError('model outside canonical record')
    urdf = material/'model.urdf'
    tree = ET.parse(urdf).getroot()
    if tree.tag != 'robot' or not tree.findall('link'):
        raise ValueError('missing URDF robot links')
    artifacts = [evidence(model)]
    # Preserve all export dependencies, not just a URDF with broken relative meshes.
    for base in (record, material):
        for f in sorted(base.rglob('*')):
            if f.is_file():
                if not f.resolve().is_relative_to(base):
                    raise ValueError('external symlink in export closure')
                artifacts.append(evidence(f))
    for mesh in tree.iter('mesh'):
        name = mesh.get('filename','')
        if '://' in name or Path(name).is_absolute():
            raise ValueError('URDF mesh URI needs explicit simulator resolver: ' + name)
        resolved = (material/name).resolve(strict=True)
        if not resolved.is_relative_to(material):
            raise ValueError('URDF mesh outside materialization')
    joints = [{'name':j.get('name'), 'type':j.get('type'),
               'axis':j.find('axis').attrib if j.find('axis') is not None else None,
               'limit':j.find('limit').attrib if j.find('limit') is not None else None}
              for j in tree.findall('joint')]
    return {'schema':'datafarm.object-asset/1','object_id':p['object_id'],'backend':'usdcraft',
            'record_id':rid,'urdf':str(urdf),'artifacts':artifacts,'joints':joints,
            'inputs':p['inputs'],'mode':p['mode'],'qualification':'unverified',
            'required_next_step':'Target simulator import, observed geometry/articulation comparison and task contact probes',
            'cost':{'basis':'unknown','amount_usd':None}}


class BuildLocked(BlockingIOError):
    """Another process holds this asset build; nothing was submitted or settled."""


def build_lock(out):
    lock = (out/'.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise BuildLocked('asset build already running in ' + str(out)) from None
    return lock


def run(directory):
    out = Path(directory).resolve(strict=True)
    with build_lock(out):
        return _run_locked(out)


def _run_locked(out):
    # A dead/unknown worker is not permission to create a duplicate asset or incur a second charge.
    if (out/'execution.json').exists():
        return read(out/'execution.json')
    p = read(out/'plan.json');check_inputs(p)
    state = {'status':'starting','started':time.time(),'cost':{'amount_usd':None,'basis':'unknown'}}
    write(out/'execution.json',state)
    with (out/'stdout.json').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
        proc = None
        try:
            proc = subprocess.Popen(p['argv'], cwd=p['cwd'], stdout=stdout, stderr=stderr, start_new_session=True)
            state.update(status='running',pid=proc.pid,pgid=proc.pid);write(out/'execution.json',state)
            code = proc.wait(timeout=p['timeout_seconds'])
            state.update(status='returned' if code == 0 else 'failed',exit_code=code)
        except BaseException as exc:
            if proc and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=type(exc).__name__)
            write(out/'execution.json',state)
            raise
        finally:
            state['ended']=time.time();write(out/'execution.json',state)
    if state['status']=='returned':
        try:
            asset = collect(p,read(out/'stdout.json'))
            write(out/'asset.json',asset)
            state['status']='exported-unqualified'
        except (ValueError,KeyError,OSError,ET.ParseError) as exc:
            state.update(status='invalid-export',error=str(exc))
        write(out/'execution.json',state)
    return state


def run_for_task(directory, task_work):
    """Attach an asset execution to the existing task budget/job/review lifecycle.

    An exported asset stays unqualified. The author queues simulator qualification
    evidence for an independent review before adopting it in a task scene.
    """
    from datafarm.agent.cost_budget import budget_for_work
    work = Path(task_work).resolve(strict=True)
    out = Path(directory).resolve(strict=True)
    if not out.is_relative_to(work):
        raise ValueError('task asset build must live inside its task workspace')
    # The build lock covers job registration, budget reservation and status writes, so a
    # concurrent second call fails before it can overwrite the status or settle the charge.
    with build_lock(out):
        return _run_for_task_locked(out, work, budget_for_work)


def _run_for_task_locked(out, work, budget_for_work):
    identifier = 'asset-' + sha256_file((out / 'plan.json'))[:24]
    # Existing attempts are observed, never submitted again or charged as a new ID.
    if (out / 'execution.json').exists():
        return read(out / 'execution.json')
    budget = budget_for_work(work)
    if budget and budget.enforced:
        raise ValueError('native USDCraft has unknown descendant costs and cannot meet enforced budget caps')
    state_path = work / 'state.json'
    state = read(state_path)
    jobs = state.setdefault('external_jobs', [])
    prior = [job for job in jobs if job.get('id') == identifier]
    if prior:
        raise ValueError('existing asset job ID without execution receipt; reconcile before resubmission')
    if budget:
        budget.reserve(identifier, 0)
    jobs.append({'id':identifier,'status':'queued','kind':'observed-object',
        'status_file':str(out/'job-status.json'),'plan':str(out/'plan.json'),
        'qualification':'unverified','cost_usd':None})
    write(state_path, state)
    write(out/'job-status.json',{'id':identifier,'status':'running','qualification':'unverified'})
    try:
        result = _run_locked(out)
    except BaseException as exc:
        write(out/'job-status.json',{'id':identifier,'status':'failed','error':type(exc).__name__,
              'qualification':'unverified','cost_usd':None})
        raise
    else:
        write(out/'job-status.json',{'id':identifier,
              'status':result['status'] if result['status']=='exported-unqualified' else 'failed',
              'asset_status':result['status'],
              'exit_code':result.get('exit_code'),'qualification':'unverified','cost_usd':None})
        return result
    finally:
        if budget:
            budget.settle(identifier,{'schema':'datafarm.external-usage/1','cost_usd':None,
                'evidence':str(out/'execution.json'),'includes_descendants':False,
                'reason':'USDCraft native observer/author costs are unknown; no hardcap asserted'})
