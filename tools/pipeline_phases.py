#!/usr/bin/env python3
"""Create a task phase by phase: one fresh agent session per phase, capped in USD by the host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from datafarm.agent.phases import (PhaseError, brief_values, capture, cost_status, load_config,  # noqa: E402
                                   render_brief, run_phase)
from datafarm.evolution.integration import source_for  # noqa: E402


def parse_settings(items):
    settings = {}
    for item in items:
        key, separator, value = item.partition('=')
        if not separator or not key:
            raise PhaseError(f'--set needs KEY=VALUE, got {item!r}')
        settings[key] = value
    return settings


def run(args):
    config = load_config(args.config)
    source = Path(source_for(args.task_dir / 'work', ROOT)).resolve()
    template = args.template or args.phase
    if args.brief:
        brief = args.brief.read_text()
    else:
        quoted = ''
        if args.quote_final:
            quoted = (args.task_dir / 'phases' / args.quote_final / 'final.md').read_text()
        summary = args.summary_file.read_text().strip() if args.summary_file else ''
        values = brief_values(args.task_dir, config, args.phase, template, source=source, quoted=quoted,
                              summary=summary, settings=parse_settings(args.set))
        brief = render_brief(template, values)
    if args.print_brief:
        print(brief)
        return {}
    return run_phase(args.task_dir, config, args.phase, template, brief, source=source,
                     resume_from=args.resume_from)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run one phase as a fresh agent session')
    r.add_argument('--task-dir', type=Path, required=True, help='holds work/ from pipeline_agent.py prepare')
    r.add_argument('--config', type=Path, required=True, help='harness, prices and caps; see examples/runtime/phases-*.json')
    r.add_argument('--phase', required=True, help='label of this session, for example P1 or R2b')
    r.add_argument('--template', help='brief in prompts/phases/ when the label differs, for example R2 for R2b')
    r.add_argument('--quote-final', metavar='PHASE', help='quote the final message of an earlier phase as unverified claims')
    r.add_argument('--summary-file', type=Path, help='host notes for the brief')
    r.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='fill a brief value')
    r.add_argument('--brief', type=Path, help='use this brief instead of rendering a template')
    r.add_argument('--resume-from', metavar='PHASE', help='continue that session after an infrastructure failure')
    r.add_argument('--print-brief', action='store_true', help='print the rendered brief and exit')
    c = sub.add_parser('capture', help='run the frozen capture with no model until the gate stops it')
    c.add_argument('--task-dir', type=Path, required=True)
    c.add_argument('--review', required=True, help='the review folder under work/reviews/ that froze the batch')
    s = sub.add_parser('cost', help='spend so far, per phase and for the running phase')
    s.add_argument('--task-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.task_dir = args.task_dir.resolve()
    pinned = Path(source_for(args.task_dir / 'work', ROOT)).resolve() if (args.task_dir / 'work').exists() else ROOT.resolve()
    if pinned != ROOT.resolve():
        return subprocess.run([sys.executable, str(pinned / 'tools/pipeline_phases.py'), *(argv or sys.argv[1:])]).returncode
    try:
        if args.cmd == 'run':
            result = run(args)
        elif args.cmd == 'capture':
            result = capture(args.task_dir, args.review)
        else:
            result = cost_status(args.task_dir)
    except (PhaseError, OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    if result:
        print(json.dumps(result, indent=2))
    if args.cmd == 'run' and result and result['status'] != 'done':
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
