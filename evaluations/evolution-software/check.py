"""Independent deterministic cases for the feedback/version interface, not robot quality."""
import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / 'src'))
    cases = ['event-idempotency', 'scope-isolation', 'hypothesis-not-authority', 'source-mutation-detected']
    outcomes = {}
    for case in cases:
        try:
            from datafarm.evolution.store import Store
            from datafarm.evolution.events import record, learn, select
            from datafarm.evolution.versions import initialize, resolve
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                store = Store(folder / 'store')
                event = {'kind': 'user-requirement', 'statement': 'Keep the requested reference.',
                         'source': {'actor': 'fixture-user', 'ref': 'fixture-1'},
                         'scope': {'hardware': 'a'}}
                first = record(store, event)
                if case == 'event-idempotency':
                    assert record(store, event) == first and len(store.all('events')) == 1
                elif case == 'scope-isolation':
                    assert len(select(store, {'hardware': 'a'})['requirements']) == 1
                    assert not select(store, {'hardware': 'b'})['requirements']
                elif case == 'hypothesis-not-authority':
                    learn(store, {'events': [first['id']], 'scope': {}, 'problem': 'p', 'action': 'a',
                                  'rationale': 'r', 'limits': 'l', 'status': 'verified'})
                    assert not select(store, {})['lessons']
                else:
                    source = folder / 'source'; source.mkdir()
                    (source / 'code.py').write_text('VALUE = 1\n')
                    version = initialize(store, source)
                    pinned = resolve(store, version['id'])
                    (pinned / 'code.py').write_text('VALUE = 2\n')
                    try:
                        resolve(store, version['id'])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('changed pinned source accepted')
                outcomes[case] = {'passed': True}
        except Exception as exc:
            outcomes[case] = {'passed': False, 'error': str(exc)}
    good = sum(x['passed'] for x in outcomes.values())
    args.output.write_text(json.dumps({'passed': good == len(cases), 'case_ids': cases,
        'metrics': {'correct_cases': good, 'failed_cases': len(cases) - good}, 'cases': outcomes}, indent=2))


if __name__ == '__main__':
    main()
