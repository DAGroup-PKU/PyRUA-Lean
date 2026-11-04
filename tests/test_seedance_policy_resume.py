"""A persisted paid task must resume without uploading or submitting a second request."""
import json

import pytest
from render_policy_fixtures import policy_fixture

from datafarm.render.seedance import batch
from datafarm.render.seedance import client
from datafarm.render.seedance.common import sha, write

CONFIG = {'clock': {'transport_fps': 24, 'canonical_fps': 30, 'max_transport_frames': 697}}


def prepared(tmp_path, monkeypatch):
    policy, body, _intent, _overlay = policy_fixture(tmp_path)
    source = tmp_path / 'input.mp4'
    ref = {'path': str(source), 'sha256': sha(source)}
    row = {key: dict(ref) for key in ('white', 'carrier', 'action', 'source_manifest', 'structure')}
    row.update(frames=1, transport_frames=1, transport_fps=24, canonical_fps=30)
    job = {'id': 'resume-one', 'category': 'test', 'source': row}
    expected = {'recipe_sha256': 'recipe', 'render_contract': str(policy)}
    monkeypatch.setattr(batch, 'input_contract', lambda *args: dict(expected))
    monkeypatch.setenv('ARK_API_KEY', 'test-only')
    task = tmp_path / 'episodes' / job['id'] / 'task.json'

    class Response:
        status_code = 200

        def json(self):
            return {'id': 'existing-provider-id'}

    client.create_once(task, body, expected, post=lambda *args, **kwargs: Response())
    # The process previously stopped after paid submission and before download/finalize.
    def forbid(*args, **kwargs):
        pytest.fail('resume attempted a new upload or paid submission')

    monkeypatch.setattr(client, 'upload', forbid)
    monkeypatch.setattr(client, 'create_once', forbid)
    polls = []

    def poll(path, raw):
        polls.append(json.loads(path.read_text())['task_id'])
        raw.write_bytes(b'downloaded existing task')
        return True

    monkeypatch.setattr(client, 'poll_download', poll)
    monkeypatch.setattr(batch, 'finalize', lambda *args: {'outputs': {}})
    return job, task, polls


def test_interrupted_paid_task_resumes_original_id(tmp_path, monkeypatch):
    job, _task, polls = prepared(tmp_path, monkeypatch)
    result = batch.run_one(tmp_path, job, CONFIG)
    assert polls == ['existing-provider-id']
    assert result['task_id'] == 'existing-provider-id'
    assert result['status'] == 'technical_complete_pending_visual_review'


@pytest.mark.parametrize('change', ['recipe', 'missing-policy', 'prompt-policy', 'real-images'])
def test_resume_rejects_changed_or_unverified_submission(tmp_path, monkeypatch, change):
    job, task, polls = prepared(tmp_path, monkeypatch)
    record = json.loads(task.read_text())
    if change == 'recipe':
        record['contract']['recipe_sha256'] = 'changed'
    elif change == 'missing-policy':
        record['contract'].pop('render_policy')
    elif change == 'prompt-policy':
        record['contract']['render_policy']['prompt_sha256'] = 'changed'
    else:
        record['contract']['render_policy']['actual_image_sha256s'] = []
    write(task, record)
    with pytest.raises(RuntimeError, match='submitted'):
        batch.run_one(tmp_path, job, CONFIG)
    assert not polls


def test_contract_is_checked_before_any_media_is_published(tmp_path, monkeypatch):
    job, _task, _polls = prepared(tmp_path, monkeypatch)
    job = {**job, 'id': 'fresh-one'}
    (tmp_path / 'other').mkdir()
    _policy, body, _intent, _overlay = policy_fixture(tmp_path / 'other')
    wrong = {**body, 'content': [{'type': 'text', 'text': 'a prompt the contract never compiled'},
                                 *[x for x in body['content'] if x.get('type') != 'text']]}
    monkeypatch.setattr(client, 'payload', lambda *args: wrong)
    with pytest.raises(ValueError, match='prompt differs'):
        batch.run_one(tmp_path, job, {**CONFIG, 'model': 'm', 'upload_interval_seconds': 0, 'create_interval_seconds': 0})
