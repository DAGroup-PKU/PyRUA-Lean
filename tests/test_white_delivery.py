"""Synthetic binding tests; not physical task qualification."""
import hashlib
import json

import pytest
from PIL import Image

from datafarm.common import sha256_file
from datafarm.task.contract import verify_delivery


def fixture(root):
    def put(name, value):
        p = root / name
        p.write_text(json.dumps(value))
        return {'path': name, 'sha256': sha256_file(p)}
    source = put('input-source.json', {'synthetic': True})
    files = [source]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    put('input.json', {'files': files, 'sha256': digest, 'target_count': 1, 'stop_after': 'white'})
    media = put('media.json', {'synthetic_media_placeholder': True})
    state = put('state-data.json', [1, 2])
    action = put('action-data.json', [3, 4])
    artifacts = {
        'trajectory': put('trajectory.json', {'schema': 'datafarm.white-trajectory/1',
            'state': state, 'action': action, 'state_schema': {'units': 'rad'},
            'action_schema': {'units': 'rad'}, 'clock': {'fps': 30}}),
        'video': put('videos.json', {'schema': 'datafarm.white-video-set/1', 'views': {'head': media}}),
        'requirements': put('requirements.json', {'synthetic': True})}
    transcript = put('transcript.json', {'synthetic_reviewer_execution': True})
    execution = put('execution.json', {'role': 'reviewer', 'actor_id': 'review',
        'producer_id': 'producer', 'transcript': transcript})
    Image.new('RGB', (4, 4)).save(root / 'view.png')
    visual = {'path': 'view.png', 'sha256': sha256_file(root / 'view.png')}
    review = {'schema': 'datafarm.white-review/1', 'producer_id': 'producer', 'reviewer_id': 'review',
        'subjects': {k: v['sha256'] for k, v in artifacts.items()}, 'execution': execution,
        'verdicts': {k: 'pass' for k in ['task', 'motion', 'alignment', 'joint_margin', 'visual']},
        'evidence': [transcript], 'visual_evidence': {'head': visual}}
    d = {'schema': 'datafarm.white-delivery/1', 'producer_id': 'producer', 'input_sha256': digest,
         'target_count': 1, 'episodes': [{'id': 's0', 'required_views': ['head'],
            'artifacts': artifacts, 'review': put('review.json', review)}],
         'attempts': put('attempts.json', [{'id': 'failed0', 'outcome': 'rejected'},
                                         {'id': 's0', 'outcome': 'admitted'}]),
         'dataset': put('dataset.json', {'schema': 'datafarm.dataset-files/1',
            'episode_ids': ['s0'], 'files': [media, state, action]}),
         'reading_guide': put('guide.json', {'synthetic': True}),
         'grounding': put('grounding.json', {'schema': 'datafarm.grounding/1', 'physical_quantities': [
             {'name': 'table_height', 'label': 'fitted', 'unit': 'm', 'value': 0.75,
              'evidence': [source]}]})}
    put('delivery.json', d)
    return d, review, put


def test_white_binding_success_is_not_render_or_training_admission(tmp_path):
    fixture(tmp_path)
    result = verify_delivery(tmp_path)
    assert result['episodes'] == 1 and result['attempts'] == 2
    assert result['rendering_certified'] is False and result['training_certified'] is False


@pytest.mark.parametrize('mutation', ['scope', 'count', 'views', 'media', 'self', 'stale', 'visual', 'execution'])
def test_white_rejects_incomplete_or_changed_evidence(tmp_path, mutation):
    d, review, put = fixture(tmp_path)
    if mutation == 'scope':
        lock = json.loads((tmp_path / 'input.json').read_text()); lock.pop('stop_after')
        put('input.json', lock)
    elif mutation == 'count': d['target_count'] = 2
    elif mutation == 'views': d['episodes'][0]['required_views'].append('wrist')
    elif mutation == 'media': (tmp_path / 'media.json').write_text('changed')
    elif mutation == 'self': review['reviewer_id'] = 'producer'
    elif mutation == 'stale': review['subjects']['trajectory'] = '0' * 64
    elif mutation == 'visual': review['visual_evidence'] = {}
    elif mutation == 'execution': (tmp_path / 'transcript.json').unlink()
    d['episodes'][0]['review'] = put('review.json', review)
    put('delivery.json', d)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_delivery(tmp_path)
