import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_pi05_dataset import admitted_reviews
from datafarm.training.cohort import cohort_contract


def args(tmp_path, count=2, scope='exploratory_diagnostic'):
    parent = tmp_path / 'parent.json'
    parent.write_text(json.dumps({'rows': [{'source_sample_id': s} for s in ['a', 'b', 'c']]}))
    return SimpleNamespace(expected_episodes=count, training_scope=scope,
                           parent_execution_plan=parent, semantic_allowlist=tmp_path / 'allowlist.json')


def test_subset_keeps_parent_gaps_and_digest(tmp_path):
    a = args(tmp_path)
    result = cohort_contract(a, [{'sample_id': 'c'}, {'sample_id': 'a'}])
    assert result['missing_ids'] == ['b']
    assert result['parent_count'] == 3 and result['selected_count'] == 2
    assert result['parent_complete'] is False
    first_hash = result['parent_execution_plan_sha256']
    a.parent_execution_plan.write_text(a.parent_execution_plan.read_text() + '\n')
    assert cohort_contract(a, [{'sample_id': 'a'}, {'sample_id': 'c'}])['parent_execution_plan_sha256'] != first_hash


def test_production_cannot_silently_shrink(tmp_path):
    with pytest.raises(ValueError, match='expected episode count'):
        cohort_contract(args(tmp_path, count=3, scope='production_policy'), [{'sample_id': 'a'}, {'sample_id': 'b'}])


def test_production_accepts_any_declared_count(tmp_path):
    assert cohort_contract(args(tmp_path, scope='production_policy'), [{'sample_id': 'a'}, {'sample_id': 'b'}]) is None


@pytest.mark.parametrize('ids', [['a', 'a'], ['a', 'unknown'], ['a']])
def test_duplicate_foreign_or_wrong_count_rejected(tmp_path, ids):
    with pytest.raises(ValueError):
        cohort_contract(args(tmp_path), [{'sample_id': s} for s in ids])


def test_parent_required(tmp_path):
    a = args(tmp_path); a.parent_execution_plan = None
    with pytest.raises(ValueError, match='parent'):
        cohort_contract(a, [{'sample_id': 'a'}, {'sample_id': 'b'}])


@pytest.mark.parametrize('fault', ['duplicate', 'reject', 'hash', 'waiver', 'failed_check'])
def test_subset_semantic_evidence_stays_strict(tmp_path, fault):
    a = args(tmp_path)
    rows = [{'sample_id': s, 'verdict': 'accept', 'waivers': [], 'checks': {'real_check': True},
             'generated_video_sha256': s + '_hash'} for s in ['a', 'b']]
    if fault == 'duplicate': rows[1] = rows[0].copy()
    elif fault == 'reject': rows[0]['verdict'] = 'reject'
    elif fault == 'hash': rows[0]['generated_video_sha256'] = 'different'
    elif fault == 'waiver': rows[0]['waivers'] = ['allowed']
    elif fault == 'failed_check': rows[0]['checks']['real_check'] = False
    a.semantic_allowlist.write_text(json.dumps({'schema_version': 'datafarm.pi05-semantic-allowlist/1.1.0',
        'waiver_policy': {'allowed_failed_checks': []}, 'reviews': rows}))
    with pytest.raises(ValueError):
        admitted_reviews(a, {s: {'provider_video_sha256': s + '_hash'} for s in ['a', 'b']})
