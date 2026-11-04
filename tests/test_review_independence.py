"""Delivery reviews under the agent runtime must cite a host-verified runtime review.

validate_completion itself (context, transcript, identity and author-session checks) is
covered by test_configurable_runtime; here it is replaced by a stub so the delivery-side
binding rules are tested on their own.
"""
import json

import pytest

import datafarm.task.contract as task_contract
from datafarm.task.contract import verify_delivery
from test_white_delivery import fixture


def runtime_review(root, d, review, put, *, verdict='pass', cover_all=True, bind_review=True):
    artifacts = d['episodes'][0]['artifacts']
    subjects = [{'path': v['path'], 'sha256': v['sha256']} for v in artifacts.values()]
    if not cover_all:
        subjects = subjects[:1]
    review = {**review, 'runtime_review': 'episodes-001'}
    review.pop('execution')
    d['episodes'][0]['review'] = review_ref = put('review.json', review)
    folder = root / '.runtime/reviews/episodes-001'
    folder.mkdir(parents=True)
    (folder / 'request.json').write_text(json.dumps({'id': 'episodes-001', 'subjects': subjects,
                                                     'instructions': 'review'}))
    evidence = [review_ref] if bind_review else [subjects[0]]
    (folder / 'report.json').write_text(json.dumps({'verdict': verdict, 'subjects': subjects,
                                                    'evidence': evidence}))
    (folder / 'completion.json').write_text(json.dumps({'verdict': verdict}))
    put('delivery.json', d)
    return folder


@pytest.fixture
def stub_completion(monkeypatch):
    calls = []

    def validate(work, folder):
        calls.append(folder.name)
        return json.loads((folder / 'completion.json').read_text())
    monkeypatch.setattr(task_contract, 'validate_completion', validate)
    return calls


def test_runtime_review_is_reported_as_verified(tmp_path, stub_completion):
    d, review, put = fixture(tmp_path)
    runtime_review(tmp_path, d, review, put)
    result = verify_delivery(tmp_path)
    assert result['review_independence'] == 'runtime-verified'
    assert stub_completion == ['episodes-001']


def test_declared_execution_record_outside_runtime_is_labelled_declared(tmp_path):
    fixture(tmp_path)
    assert verify_delivery(tmp_path)['review_independence'] == 'declared-execution-record'


def test_runtime_workspace_refuses_hand_written_execution_record(tmp_path):
    fixture(tmp_path)
    (tmp_path / '.runtime').mkdir()
    (tmp_path / '.runtime/author.json').write_text('{}')
    with pytest.raises(ValueError, match='must cite the runtime review'):
        verify_delivery(tmp_path)


@pytest.mark.parametrize('kwargs, message', [
    ({'verdict': 'reject'}, 'did not pass'),
    ({'cover_all': False}, 'did not cover every delivered artifact'),
    ({'bind_review': False}, 'does not bind this episode review'),
])
def test_runtime_review_must_pass_cover_and_bind(tmp_path, stub_completion, kwargs, message):
    d, review, put = fixture(tmp_path)
    runtime_review(tmp_path, d, review, put, **kwargs)
    with pytest.raises(ValueError, match=message):
        verify_delivery(tmp_path)


def test_cited_runtime_review_must_exist_and_have_a_safe_id(tmp_path, stub_completion):
    d, review, put = fixture(tmp_path)
    folder = runtime_review(tmp_path, d, review, put)
    (folder / 'completion.json').unlink()
    with pytest.raises(ValueError, match='has not completed'):
        verify_delivery(tmp_path)
    review = json.loads((tmp_path / 'review.json').read_text())
    review['runtime_review'] = '../escape'
    d['episodes'][0]['review'] = put('review.json', review)
    put('delivery.json', d)
    with pytest.raises(ValueError, match='invalid runtime review ID'):
        verify_delivery(tmp_path)
