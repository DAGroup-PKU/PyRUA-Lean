"""Explicit synthetic reference/input contract for transport tests; never live service use."""
import base64
import json
from pathlib import Path

from datafarm.render.policy import compile_prompt
from datafarm.common import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def policy_fixture(root, mode='sim2real'):
    root = Path(root)
    real = root / 'real.png'
    real.write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='))
    source = root / 'input.mp4'; source.write_bytes(b'synthetic input fixture')
    ref = {'path': str(real), 'sha256': sha256_file(real), 'provenance': {
        'kind': 'real-episode-frame', 'source': {'path': str(source), 'sha256': sha256_file(source)},
        'frame': 0, 'description': 'synthetic fixture simulates an extracted real frame'},
        'scope': 'full-view', 'views': ['head', 'left_wrist', 'right_wrist'],
        'view_regions': {view: [0, 0, 1, 1] for view in ['head', 'left_wrist', 'right_wrist']},
        'coverage': ['full_scene', 'robot_body', 'attached_cameras_and_cables', 'task_objects',
                     'table_and_background', 'photographic_character']}
    axes = [{'axis': 'table-material', 'values': ['wood', 'stone']}] if mode == 'ood' else []
    intent = {'schema': 'datafarm.render-intent/1', 'mode': mode, 'run_id': 'new-run', 'task_id': 'unseen-task',
        'mode_confirmation': {'mode': mode, 'run_id': 'new-run', 'task_id': 'unseen-task',
          'quote': 'synthetic test instruction', 'source': 'test only', 'authorized_ood_axes': axes},
        'real_baseline': 'reference scene', 'authorized_ood_axes': axes,
        'appearance_views': ['head', 'left_wrist', 'right_wrist'],
        'appearance_requirements': {
            name: {'required': True, 'description': 'synthetic fixture requirement'}
            for name in ['robot_body', 'attached_cameras_and_cables', 'task_objects',
                         'table_and_background', 'photographic_character']},
        'appearance_invariants': ['robot identity'], 'real_references': [ref]}
    overlay = {'TASK_DESCRIPTION': 'Move the new object', 'SOURCE_VIDEO_ROLE': 'motion carrier',
        'CAMERA_TIMING_CONTRACT': 'preserve camera and frame timing', 'TASK_EVENT_CONTRACT': 'grasp transport release',
        'CONTINUITY_CONTRACT': 'preserve terminal relation'}
    compiled = compile_prompt(root, intent, overlay, assets=ROOT)
    path = root / 'render-contract.json'; path.write_text(json.dumps(compiled))
    body = {'model': 'test-model', 'content': [
        {'type': 'text', 'text': compiled['prompt']},
        {'type': 'video_url', 'video_url': {'url': 'https://example.invalid/source.mp4'}},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(real.read_bytes()).decode()}}]}
    return path, body, intent, overlay
