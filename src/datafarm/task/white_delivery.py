"""Source-bound white-stage delivery, distinct from rendered/training admission."""
import json
from pathlib import Path

from datafarm.common import sha256_file
from datafarm.core.file_binding import bound_file, require
from datafarm.task.contract import _is_visual_evidence, verify_grounding, verify_review_independence


def verify_white_delivery(root: Path, lock: dict, delivery: dict) -> dict:
    require(lock.get('stop_after') == 'white', 'white delivery not requested in input lock')
    require(delivery.get('input_sha256') == lock['sha256'], 'delivery input mismatch')
    count = delivery.get('target_count')
    require(type(count) is int and count > 0 and count == lock.get('target_count'),
            'requested target count changed or missing')
    producer = delivery.get('producer_id')
    require(isinstance(producer, str) and bool(producer), 'producer missing')
    episodes = delivery.get('episodes')
    require(isinstance(episodes, list) and len(episodes) == count, 'target count not delivered')
    ids = [e['id'] for e in episodes]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == count,
            'duplicate/invalid episode IDs')
    quantities = verify_grounding(root, delivery)
    attempts = json.loads(bound_file(root, delivery['attempts']).read_text())
    require(isinstance(attempts, list) and len(attempts) >= count, 'attempt denominator missing')
    require(len({a['id'] for a in attempts}) == len(attempts), 'duplicate attempts')
    require(all(a.get('outcome') in {'admitted', 'rejected', 'infrastructure_failure'} for a in attempts),
            'invalid attempt outcome')
    require({a['id'] for a in attempts if a['outcome'] == 'admitted'} == set(ids),
            'attempt/admission mismatch')
    dataset = json.loads(bound_file(root, delivery['dataset']).read_text())
    require(dataset.get('schema') == 'datafarm.dataset-files/1', 'invalid dataset manifest')
    require(dataset.get('episode_ids') == ids, 'dataset membership mismatch')
    require(bool(dataset.get('files')), 'dataset has no files')
    dataset_paths = {bound_file(root, f, external=True) for f in dataset['files']}
    bound_file(root, delivery['reading_guide'])
    independence = set()
    for episode in episodes:
        artifacts = episode['artifacts']
        require(set(artifacts) == {'trajectory', 'video', 'requirements'}, 'white subjects missing')
        for ref in artifacts.values():
            bound_file(root, ref)
        videos = json.loads(bound_file(root, artifacts['video']).read_text())
        require(videos.get('schema') == 'datafarm.white-video-set/1', 'white video manifest missing')
        require(isinstance(videos.get('views'), dict) and bool(videos['views']), 'views missing')
        require(set(videos['views']) == set(episode['required_views']), 'declared views missing')
        for ref in videos['views'].values():
            require(bound_file(root, ref, external=True) in dataset_paths, 'video absent from dataset')
        trajectory = json.loads(bound_file(root, artifacts['trajectory']).read_text())
        require(trajectory.get('schema') == 'datafarm.white-trajectory/1', 'trajectory manifest missing')
        require(all(trajectory.get(k) for k in ['state_schema', 'action_schema', 'clock']),
                'state/action/clock schema missing')
        for name in ['state', 'action']:
            require(bound_file(root, trajectory[name], external=True) in dataset_paths,
                    'trajectory absent from dataset')
        review = json.loads(bound_file(root, episode['review']).read_text())
        require(review.get('schema') == 'datafarm.white-review/1', 'white review missing')
        require(review.get('producer_id') == producer, 'review producer mismatch')
        require(review.get('subjects') == {k: v['sha256'] for k, v in artifacts.items()},
                'stale review subjects')
        require(review.get('verdicts') == {k: 'pass' for k in
                ['task', 'motion', 'alignment', 'joint_margin', 'visual']}, 'review not fully passed')
        independence.add(verify_review_independence(root, review, episode['review'], producer,
                                                     {v['sha256'] for v in artifacts.values()}))
        require(bool(review.get('evidence')), 'review evidence missing')
        for ref in review['evidence']:
            bound_file(root, ref)
        visual = review.get('visual_evidence', {})
        require(set(visual) == set(episode['required_views']), 'visual coverage missing')
        for ref in visual.values():
            require(_is_visual_evidence(bound_file(root, ref)), 'visual evidence must decode')
    return {'schema': 'datafarm.white-delivery-verification/1', 'valid': True,
            'episodes': count, 'attempts': len(attempts),
            'delivery_sha256': sha256_file(root / 'delivery.json'),
            'review_independence': 'runtime-verified' if independence == {'runtime-verified'}
            else 'declared-execution-record', 'physical_quantities': quantities,
            'scope': 'white-stage content bindings; semantics require independent execution evidence',
            'rendering_certified': False, 'training_certified': False}
