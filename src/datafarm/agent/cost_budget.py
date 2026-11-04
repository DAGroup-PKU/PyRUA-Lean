"""Persistent reservations for the whole task, including failed/unknown sessions.

The deployment driver must enforce its reservation across *all* descendant model
calls. This ledger is a scheduler guard, not a provider billing firewall.
"""
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import json
from pathlib import Path
from datafarm.common import atomic_write_json, sha256_text


def micros(value):
    if isinstance(value, bool):
        raise ValueError('cost must be a finite nonnegative USD amount')
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('invalid USD amount') from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError('cost must be a finite nonnegative USD amount')
    return int((amount * 1000000).to_integral_value(rounding=ROUND_CEILING))


def validate_policy(policy):
    if policy.get('schema') != 'datafarm.task-cost-policy/2':
        raise ValueError('invalid task cost policy schema')
    if policy.get('mode') not in {'disabled', 'advisory', 'enforced'}:
        raise ValueError('invalid budget mode')
    if policy.get('on_excess', 'warn') not in {'warn', 'pause'}:
        raise ValueError('on_excess must be warn or pause')
    target, limit = micros(policy['target_usd']), micros(policy['limit_usd'])
    if not 0 < target <= limit:
        raise ValueError('require 0 < target_usd <= limit_usd')
    if policy.get('scope') != 'all-model-work-through-white-data':
        raise ValueError('budget must cover all model work through white data')
    return policy


class TaskBudget:
    def __init__(self, work):
        self.work = Path(work)
        self.policy = validate_policy(json.loads((self.work / 'cost-policy.json').read_text()))
        self.digest = sha256_text(json.dumps(self.policy, sort_keys=True))
        self.limit = micros(self.policy['limit_usd'])
        self.mode = self.policy['mode']
        self.enforced = self.mode == 'enforced'

    @contextmanager
    def transaction(self):
        with (self.work / '.cost.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.work / 'cost-ledger.json'
            if not path.exists() and any(
                    'cost_reservation_id' in json.loads(p.read_text())
                    for p in (self.work / 'sessions').glob('*/execution.json')):
                raise ValueError('cost ledger missing after execution; refusing to reset spend')
            data = json.loads(path.read_text()) if path.exists() else {
                'schema': 'datafarm.task-cost-ledger/1', 'policy_sha256': self.digest,
                'reservations': {}, 'blocked': False}
            if data['policy_sha256'] != self.digest:
                raise ValueError('cost policy changed; cannot reset the existing task budget')
            yield data
            atomic_write_json(path, data, sort_keys=False, allow_nan=False, mode=0o644)

    def _summary(self, data):
        rows = list(data['reservations'].values())
        committed = sum(r['charged_microusd'] for r in rows)
        known = sum(r['actual_microusd'] for r in rows if r['actual_microusd'] is not None)
        unresolved = sum(r['status'] in {'reserved', 'unknown'} for r in rows)
        blocked = data['blocked'] if self.enforced else (
            self.mode == 'advisory' and self.policy.get('on_excess') == 'pause' and known >= self.limit)
        return {'mode': self.mode, 'coverage': 'incomplete' if unresolved else 'complete',
                'unknown_sessions': unresolved, 'total_usd': known / 1e6 if not unresolved else None, 'stop_required': blocked or (self.enforced and committed >= self.limit),
                'warning': self.mode != 'disabled' and (known >= self.limit or unresolved > 0),
                'target_usd': self.policy['target_usd'], 'limit_usd': self.policy['limit_usd'],
                'known_usd': known / 1e6, 'committed_usd': committed / 1e6,
                'remaining_usd': max(0, self.limit - committed) / 1e6,
                'target_reached': committed >= micros(self.policy['target_usd']),
                'unresolved': sum(r['status'] in {'reserved', 'unknown'} for r in rows),
                'blocked': blocked, 'sessions': sum(r.get('kind') != 'carryover' for r in rows),
                'history_entries': sum(r.get('kind') == 'carryover' for r in rows)}

    def status(self):
        with self.transaction() as data:
            return self._summary(data)

    def reserve(self, identifier, usd):
        requested = micros(usd)
        if not identifier or (requested <= 0 and self.enforced):
            raise ValueError('reservation ID and positive amount required')
        with self.transaction() as data:
            # Reusing an ID must not authorize a second invocation.
            if identifier in data['reservations']:
                raise ValueError('reservation ID already used')
            state = self._summary(data)
            if state['blocked'] or (self.enforced and state['unresolved']):
                raise ValueError('unresolved/blocked prior session; reconcile before another model call')
            remaining = self.limit - sum(r['charged_microusd'] for r in data['reservations'].values())
            if self.enforced and remaining <= 0:
                raise ValueError('task cost budget exhausted')
            allowed = min(requested, remaining) if self.enforced else requested
            data['reservations'][identifier] = {
                'status': 'reserved', 'reserved_microusd': allowed,
                'charged_microusd': allowed, 'actual_microusd': None}
            return allowed / 1e6

    def settle(self, identifier, receipt):
        with self.transaction() as data:
            row = data['reservations'][identifier]
            if row['status'] != 'reserved':
                raise ValueError('reservation already settled; cannot count a receipt twice')
            valid = isinstance(receipt, dict) and (
                receipt.get('schema') == 'datafarm.session-cost/1'
                and receipt.get('reservation_id') == identifier
                and receipt.get('includes_descendants') is True
                and receipt.get('complete') is True
                and bool(receipt.get('evidence'))
                and receipt.get('cost_basis') in {'provider', 'api-equivalent'})
            try:
                actual = micros(receipt['cost_usd']) if valid else None
            except (ValueError, KeyError, TypeError):
                actual = None
            if actual is None:
                row['status'] = 'unknown'
                data['blocked'] = self.enforced
            else:
                row.update(status='settled', actual_microusd=actual, charged_microusd=actual,
                           receipt=receipt)
                if self.enforced and actual > row['reserved_microusd']:
                    row['status'] = 'overrun'
                    data['blocked'] = True
            try:
                json.dumps(receipt, allow_nan=False)
                row['usage_evidence'] = receipt
            except (ValueError, TypeError):
                row['usage_evidence'] = {'invalid_receipt': True}
            return self._summary(data)


def budget_for_work(work):
    """Budget of a work directory, or None when it has no cost policy; a budgeted input can never lose its policy."""
    work = Path(work)
    lock = json.loads((work / 'input.json').read_text()) if (work / 'input.json').exists() else {}
    expected = lock.get('cost_policy_sha256')
    if expected or (work / 'cost-policy.json').exists() or (work / 'cost-ledger.json').exists():
        budget = TaskBudget(work)
        if expected and budget.digest != expected:
            raise ValueError('cost policy differs from the prepared input contract')
        return budget
    return None
