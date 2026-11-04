"""Freeze evaluator assets, run paired evaluations, retain failed attempts."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from datafarm.evolution.store import digest, identifier, read, write
from datafarm.evolution.versions import files, verify_source


def run(argv, *, cwd, directory, timeout, payload='', env=None):
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError('command must be an argv array, not shell text')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    directory.mkdir(parents=True, exist_ok=False)
    result = {'argv': argv, 'cwd': str(cwd), 'started': time.time(), 'status': 'starting'}
    write(directory / 'execution.json', result)
    with (directory / 'stdout.log').open('w') as out, (directory / 'stderr.log').open('w') as err:
        process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                   text=True, start_new_session=True, env=env)
        result.update(pid=process.pid, status='running')
        write(directory / 'execution.json', result)
        try:
            process.communicate(payload, timeout=timeout)
            result['status'] = 'returned' if process.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
        finally:
            # Kill descendants even if the top-level driver returned successfully.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    result.update(exit_code=process.returncode, ended=time.time())
    write(directory / 'execution.json', result)
    return result


def freeze_policy(store, path):
    path = Path(path).resolve()
    policy = read(path)
    if policy.get('schema') != 'datafarm.evolution-policy/1':
        raise ValueError('unsupported evolution policy')
    if not policy.get('allowed_paths') or any(not isinstance(x, str) or x.startswith('/') or '..' in Path(x).parts
                                            or not x for x in policy['allowed_paths']):
        raise ValueError('policy requires repository-relative allowed_paths')
    if not isinstance(policy.get('protected_paths', []), list) or any(
            not isinstance(x, str) or not x or x.startswith('/') or '..' in Path(x).parts
            for x in policy.get('protected_paths', [])):
        raise ValueError('protected_paths must be repository-relative paths')
    if not policy.get('checks') or not policy.get('metrics'):
        raise ValueError('policy requires independent checks and improvement metrics')
    ids = [identifier(c['id']) for c in policy['checks']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate evaluation check IDs')
    for check in policy['checks']:
        if not isinstance(check.get('argv'), list) or not check['argv']:
            raise ValueError('evaluation check requires argv')
        if not any('{report}' in a for a in check['argv']) or not any('{repo}' in a for a in check['argv']):
            raise ValueError('evaluation argv must consume {repo} and write {report}')
        if not math.isfinite(check.get('timeout', 0)) or check.get('timeout', 0) <= 0:
            raise ValueError('evaluation check requires timeout')
    if policy.get('tests_lessons') and not any('{experience}' in a for c in policy['checks'] for a in c['argv']):
        raise ValueError('lesson evaluation must explicitly consume {experience}')
    for metric, spec in policy['metrics'].items():
        if '.' not in metric or metric.split('.')[0] not in ids:
            raise ValueError('metrics must be check.metric: min|max')
        metric_rule(spec)
    if policy.get('evidence_level') not in {'software', 'simulator', 'new-task', 'meta-improvement'}:
        raise ValueError('policy requires an explicit evidence_level')
    assets = {}
    for relative in policy.get('assets', []):
        unresolved = path.parent / relative
        source = unresolved.resolve(strict=True)
        if Path(relative).is_absolute() or not source.is_relative_to(path.parent) or unresolved.is_symlink():
            raise ValueError('evaluator asset escapes policy directory')
        assets[relative] = source.read_bytes()
    if not assets:
        raise ValueError('freeze evaluator code/data using policy assets')
    manifest = {k: digest(list(v)) for k, v in assets.items()}
    value = {'policy': policy, 'assets': manifest}
    name = digest(value)
    directory = store.root / 'policies' / name
    directory.mkdir(parents=True, exist_ok=True)
    for relative, raw in assets.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('frozen evaluator asset changed')
        target.write_bytes(raw)
    store.put('policies', name, value)
    return name


def check_policy(store, name):
    value = store.get('policies', name)
    if digest(value) != name:
        raise ValueError('frozen policy changed')
    directory = store.root / 'policies' / name
    for relative, sha in value['assets'].items():
        if digest(list((directory / relative).read_bytes())) != sha:
            raise ValueError('frozen evaluator changed')
    return value['policy'], directory


def evaluate(store, source, policy_id, directory, deadline, *, lesson_ids=()):
    policy, suite = check_policy(store, policy_id)
    directory.mkdir(parents=True, exist_ok=False)
    root = directory / 'source'
    shutil.copytree(verify_source(store, source), root)
    before = files(root)
    experience = directory / 'experience.json'
    write(experience, {'lessons': [store.get('lessons', x) for x in lesson_ids]})
    results, metrics, cases = {}, {}, {}
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for check in policy['checks']:
        name = check['id']
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('evaluation wall budget exhausted')
        report = directory / (name + '.json')
        replacements = {'{repo}': str(root), '{suite}': str(suite), '{report}': str(report),
                        '{experience}': str(experience)}
        argv = list(check['argv'])
        for old, new in replacements.items():
            argv = [a.replace(old, new) for a in argv]
        execution = run(argv, cwd=directory, directory=directory / name,
                        timeout=min(check['timeout'], remaining), env=environment)
        outcome = read(report) if report.exists() else {}
        if not isinstance(outcome, dict) or not isinstance(outcome.get('metrics', {}), dict):
            raise ValueError('evaluation report must contain a metrics object')
        valid = (execution['status'] == 'returned' and type(outcome.get('passed')) is bool)
        results[name] = {'execution': execution, 'outcome': outcome, 'valid': valid}
        for key, score in outcome.get('metrics', {}).items():
            if type(score) not in (float, int) or not math.isfinite(score):
                raise ValueError('evaluation metrics must be finite numbers')
            metrics[f'{name}.{key}'] = score
        case_ids = outcome.get('case_ids', [])
        if (not isinstance(case_ids, list) or not case_ids or any(not isinstance(x, str) or not x for x in case_ids)
                or len(case_ids) != len(set(case_ids))):
            raise ValueError('evaluation requires unique case_ids (fixed denominator)')
        cases[name] = sorted(case_ids)
        check_policy(store, policy_id)
        if files(root) != before:
            raise ValueError('evaluation modified source; write artifacts outside source')
    value = {'source': source, 'policy': policy_id, 'checks': results,
             'metrics': metrics, 'case_ids': cases,
             'valid': all(x['valid'] for x in results.values()),
             'passed': all(x['valid'] and x['outcome']['passed'] for x in results.values())}
    write(directory / 'result.json', value)
    return value


def metric_rule(spec):
    """'max'/'min', or a noise-aware rule {direction, tolerance, relative, min_improvement, decisive}.

    tolerance: largest allowed regression (absolute, or a fraction of |baseline| when relative); default 0.
    min_improvement: smallest gain that counts as an improvement; default any positive gain.
    decisive: whether this metric alone can justify promotion; default True.
    """
    if spec in ('min', 'max'):
        return {'direction': spec, 'tolerance': 0.0, 'relative': False, 'min_improvement': 0.0, 'decisive': True}
    if not isinstance(spec, dict) or spec.get('direction') not in ('min', 'max') or set(spec) - {
            'direction', 'tolerance', 'relative', 'min_improvement', 'decisive'}:
        raise ValueError('metric rule must be min|max or {direction, tolerance, relative, min_improvement, decisive}')
    rule = {'direction': spec['direction'], 'tolerance': spec.get('tolerance', 0.0), 'relative': spec.get('relative', False),
            'min_improvement': spec.get('min_improvement', 0.0), 'decisive': spec.get('decisive', True)}
    for key in ('tolerance', 'min_improvement'):
        if type(rule[key]) not in (int, float) or not math.isfinite(rule[key]) or rule[key] < 0:
            raise ValueError(f'metric {key} must be a finite non-negative number')
    if type(rule['relative']) is not bool or type(rule['decisive']) is not bool:
        raise ValueError('metric relative/decisive must be booleans')
    return rule


def compare(policy, baseline, candidate):
    if not baseline['valid'] or not candidate['valid'] or not candidate['passed']:
        return False, 'invalid evaluation or candidate checks failed'
    if baseline['case_ids'] != candidate['case_ids']:
        return False, 'evaluation denominator changed'
    improvements = []
    for name, spec in policy['metrics'].items():
        if name not in baseline['metrics'] or name not in candidate['metrics']:
            return False, f'missing metric: {name}'
        rule = metric_rule(spec)
        b, c = baseline['metrics'][name], candidate['metrics'][name]
        delta = (c - b) if rule['direction'] == 'max' else (b - c)
        scale = abs(b) if rule['relative'] else 1.0
        if delta < -rule['tolerance'] * scale:
            return False, f'metric regressed: {name}'
        if rule['decisive'] and delta > 0 and delta >= rule['min_improvement'] * scale:
            improvements.append(name)
    return bool(improvements), ('improved: ' + ', '.join(improvements) if improvements else 'no measured improvement')
