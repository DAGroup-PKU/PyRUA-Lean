#!/usr/bin/env python3
"""Compile a task-generated render overlay against general policy; inspect actual requests."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from datafarm.render.policy import compile_prompt, check_seedance_payload, check_h3_payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compile')
    c.add_argument('--intent', type=Path, required=True)
    c.add_argument('--overlay', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    for name in ['check-seedance', 'check-h3']:
        c = sub.add_parser(name); c.add_argument('--contract', type=Path, required=True)
        c.add_argument('--payload', type=Path, required=True)
    a = p.parse_args()
    try:
        if a.cmd == 'compile':
            result = compile_prompt(a.intent.resolve().parent, json.loads(a.intent.read_text()),
                                    json.loads(a.overlay.read_text()), assets=ROOT)
            with a.output.open('x') as f: f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        else:
            fn = check_seedance_payload if a.cmd == 'check-seedance' else check_h3_payload
            result = fn(json.loads(a.payload.read_text()), a.contract)
        print(json.dumps({'schema': result['schema'], 'valid': True}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as e:
        print(f'render-policy: {e}', file=sys.stderr); return 2

if __name__ == '__main__':
    raise SystemExit(main())
