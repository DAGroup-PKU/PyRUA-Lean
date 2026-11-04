#!/usr/bin/env python3
"""Inventory project text by responsibility; validate the actual cold-context load graph."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from datafarm.agent.context import build_context



def inventory():
    catalog = json.loads((ROOT / 'configs/agent-instructions.json').read_text())
    loaded = set()
    for role in catalog['roles']:
        loaded.update(x['path'] for x in build_context(role, root=ROOT)['receipt']['files'])
    if (ROOT / '.git').exists():
        names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    else:
        # Source archives have no Git metadata. Audit the extracted source using
        # the same generated/runtime directory exclusions as the repository.
        # dist/ and build/ are packaging outputs only at the root (as in .gitignore);
        # vendored sources may live in nested directories of those names.
        excluded = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv',
                    '.datafarm', 'runs', 'artifacts',
                    'secrets', 'environments', 'external', 'data'}
        names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                 if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)
                 and p.relative_to(ROOT).parts[0] not in {'dist', 'build'}
                 and not any(part.endswith('.egg-info') for part in p.relative_to(ROOT).parts)]
    rows, embedded, errors = [], [], []
    for name in sorted(set(names)):
        p = ROOT / name
        if not p.is_file() or p.suffix not in {'.md', '.txt', '.json', '.yaml', '.yml', '.py'}:
            continue
        if name == 'tools/audit_instruction_layers.py':
            continue  # this generator is indexed separately
        raw = p.read_bytes()
        if name in loaded or name.startswith(('.agents/', 'prompts/')) or name == 'configs/agent-instructions.json':
            layer = 'universal-instruction'; reason = 'General cross-task behavior, runtime role/session or policy template.'
        elif name.startswith('schemas/'):
            layer = 'universal-runtime-schema'; reason = 'Generic protocol/schema used to generate or check task outputs.'
        elif name == 'configs/expert-harness/prices.json':
            layer = 'deployment-or-device-profile'; reason = 'Device facts/service pricing; validate for the current deployment, not task intent.'
        elif name.startswith('src/'):
            layer = 'runtime-implementation'; reason = 'Executable core/harness/provider library; local prompts retain API schema and are not universally injected.'
        elif name.startswith('vendor/') and p.suffix != '.py':
            layer = 'task-product-or-upstream-reference'; reason = 'Vendored prior output/upstream documentation; never injected as general task authority.'
        elif name.startswith('configs/') and name not in {'configs/joint-margin-requirements.json'}:
            layer = 'task-product-or-tool-schema'; reason = 'Existing task recipe or tool config/schema, selected only after current-task applicability review.'
        elif name.startswith('examples/'):
            layer = 'task-product-template-or-example'; reason = 'Template/reference for agent-generated artifacts, not required input for unknown tasks.'
        elif name.startswith('tests/'):
            layer = 'test-evidence'; reason = 'Software test/fixture; no real task success claim.'
        else:
            layer = 'implementation-reference'; reason = 'Tool implementation or technical interface documentation, read as needed.'
        rows.append({'path': name, 'layer': layer, 'reason': reason, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'loaded_by_default_role': name in loaded})
        if layer == 'universal-instruction':
            text = raw.decode()
            if re.search(r'/home/|/mnt/|ssh -o ', text):
                errors.append(f'{name}: machine-local authority in universal instruction')
        if p.suffix == '.py':
            try: tree = ast.parse(raw)
            except (SyntaxError, UnicodeDecodeError): continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and re.search('prompt|system_message', target.id, re.I):
                            embedded.append({'path': name, 'line': node.lineno, 'symbol': target.id,
                                             'layer': layer, 'disposition': 'runtime-construction' if name.startswith('src/') else 'task-product-or-tool-local'})
    return {'schema': 'datafarm.instruction-inventory/1', 'files': rows, 'embedded_prompt_assignments': embedded,
            'roles': list(catalog['roles']), 'errors': errors,
            'rule': 'Classify by responsibility, not extension. Global loader never scans task products.'}


def main():
    p = argparse.ArgumentParser(description='Classify files by instruction layer and report violations.')
    p.parse_args()
    result = inventory()
    print(json.dumps({'files': len(result['files']), 'embedded_prompts': len(result['embedded_prompt_assignments']),
                      'roles': len(result['roles']), 'errors': result['errors']}, ensure_ascii=False, indent=2))
    return bool(result['errors'])

if __name__ == '__main__': raise SystemExit(main())
