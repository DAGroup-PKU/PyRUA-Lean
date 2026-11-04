#!/usr/bin/env python3
"""Record feedback, run bounded self-improvement, inspect evidence, and roll back."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from datafarm.evolution.store import Store, read
from datafarm.evolution.events import record, learn, pending, select, record_change
from datafarm.evolution.versions import initialize, active, rollback
from datafarm.evolution.experiments import freeze_policy
from datafarm.evolution.controller import cycle, propose, assess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=ROOT / '.datafarm/evolution')
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init'); init.add_argument('--source', type=Path, default=ROOT)
    for name in ('record', 'learn'):
        cmd = commands.add_parser(name); cmd.add_argument('--file', type=Path, required=True)
    commands.add_parser('status')
    changed = commands.add_parser('record-change')
    changed.add_argument('--source', type=Path, default=ROOT)
    changed.add_argument('--parent', help='base capability version; defaults to active')
    changed.add_argument('--statement', required=True)
    changed.add_argument('--source-ref', required=True)
    changed.add_argument('--scope', type=Path)
    changed.add_argument('--evidence', type=Path, action='append', default=[])
    ctx = commands.add_parser('context'); ctx.add_argument('--scope', type=Path, required=True)
    prop = commands.add_parser('propose')
    prop.add_argument('--source', type=Path, required=True)
    prop.add_argument('--proposal', type=Path, required=True)
    prop.add_argument('--policy', type=Path, required=True)
    prop.add_argument('--event', action='append', required=True)
    check = commands.add_parser('evaluate')
    check.add_argument('--candidate', required=True)
    check.add_argument('--timeout', type=float, default=600)
    check.add_argument('--shadow', action='store_true')
    run = commands.add_parser('cycle')
    run.add_argument('--driver', type=Path, required=True)
    run.add_argument('--policy', type=Path, required=True)
    run.add_argument('--max-candidates', type=int, default=1)
    run.add_argument('--timeout', type=float, default=600)
    run.add_argument('--event', action='append', help='explicitly retry selected feedback, including previously rejected proposals')
    back = commands.add_parser('rollback'); back.add_argument('--version', required=True)
    args = parser.parse_args(argv)
    try:
        store = Store(args.store)
        if args.command == 'init': result = initialize(store, args.source)
        elif args.command == 'record': result = record(store, read(args.file))
        elif args.command == 'learn': result = learn(store, read(args.file))
        elif args.command == 'record-change':
            result = record_change(store, args.source, args.parent or active(store), statement=args.statement,
                                   source_ref=args.source_ref, selectors=read(args.scope) if args.scope else {},
                                   evidence=args.evidence)
        elif args.command == 'status':
            result = {'active': active(store), 'pending_events': pending(store), 'candidates': store.all('states')}
        elif args.command == 'context': result = select(store, read(args.scope), version=active(store))
        elif args.command == 'propose':
            result = propose(store, active(store), args.source, read(args.proposal), args.event, freeze_policy(store, args.policy))
        elif args.command == 'evaluate': result = assess(store, args.candidate, timeout=args.timeout, promote=not args.shadow)
        elif args.command == 'rollback': result = rollback(store, args.version)
        else:
            result = cycle(store, root=ROOT, driver=read(args.driver), policy_path=args.policy,
                           max_candidates=args.max_candidates, timeout=args.timeout, event_ids=args.event)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failures = {'rejected', 'evaluation-error', 'stale', 'timeout', 'failed', 'proposal-error', 'budget-exhausted'}
        return 75 if result.get('status') in failures or any(r.get('status') in failures for r in result.get('results', [])) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'pipeline_evolve: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
