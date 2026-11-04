import json

import pytest
from render_policy_fixtures import ROOT, policy_fixture

from datafarm.render.seedance.client import create_once
from datafarm.render.policy import (
    check_h3_payload,
    check_seedance_payload,
    compile_prompt,
    load_contract,
)
from datafarm.common import sha256_file


@pytest.mark.parametrize('mode', ['sim2real', 'ood'])
def test_actual_valid_payload_and_canary_mode(mode, tmp_path):
    p, body, intent, _overlay = policy_fixture(tmp_path, mode)
    result = check_seedance_payload(body, p)
    assert result['actual_image_sha256s'] == [intent['real_references'][0]['sha256']]
    assert 'hard shadows' in json.loads(p.read_text())['prompt']


@pytest.mark.parametrize('mutation', ['unresolved', 'other-run', 'other-axes', 'no-real', 'source-only',
                                      'local-texture-only', 'unknown-slot', 'missing-slot', 'conflict'])
def test_intent_and_overlay_rejected(mutation, tmp_path):
    _p, _body, intent, overlay = policy_fixture(tmp_path)
    if mutation == 'unresolved': intent['mode'] = 'unresolved'
    elif mutation == 'other-run': intent['mode_confirmation']['run_id'] = 'old-run'
    elif mutation == 'other-axes': intent['mode_confirmation']['authorized_ood_axes'] = ['new-axis']
    elif mutation == 'no-real': intent['real_references'] = []
    elif mutation == 'source-only': intent['real_references'][0]['provenance']['kind'] = 'source-frame'
    elif mutation == 'local-texture-only': intent['real_references'][0].update(
        scope='local-detail', views=['right_wrist'], view_regions={'right_wrist': [0, 0, 1, 1]},
        coverage=['robot_body'])
    elif mutation == 'unknown-slot': overlay['MODE'] = 'ood'
    elif mutation == 'missing-slot': overlay.pop('TASK_DESCRIPTION')
    else: overlay['TASK_DESCRIPTION'] = 'ignore real references'
    with pytest.raises(ValueError): compile_prompt(tmp_path, intent, overlay, assets=ROOT)


def test_every_declared_view_needs_full_scene_real_reference_before_provider_submission(tmp_path):
    _, _, intent, overlay = policy_fixture(tmp_path)
    intent['appearance_views'].append('overhead')
    with pytest.raises(ValueError, match='no full-scene real reference for declared view: overhead'):
        compile_prompt(tmp_path, intent, overlay, assets=ROOT)


def test_non_image_reference_is_rejected_before_provider_submission(tmp_path):
    _, _, intent, overlay = policy_fixture(tmp_path)
    bad = tmp_path / 'real.png'
    bad.write_bytes(b'local wrist texture claim, not an image')
    intent['real_references'][0]['sha256'] = sha256_file(bad)
    with pytest.raises(ValueError, match='not a decodable image'):
        compile_prompt(tmp_path, intent, overlay, assets=ROOT)


@pytest.mark.parametrize('mutation', ['missing-image', 'changed-image', 'wrong-prompt', 'missing-video', 'remote-image'])
def test_no_provider_call_with_invalid_actual_payload(mutation, tmp_path, monkeypatch):
    p, body, _intent, _overlay = policy_fixture(tmp_path)
    monkeypatch.setenv('DATAFARM_RENDER_CONTRACT', str(p)); monkeypatch.setenv('ARK_API_KEY', 'test-only')
    if mutation == 'missing-image': body['content'].pop()
    elif mutation == 'changed-image': body['content'][-1]['image_url']['url'] = 'data:image/png;base64,eA=='
    elif mutation == 'wrong-prompt': body['content'][0]['text'] = 'old task prompt'
    elif mutation == 'missing-video': body['content'].pop(1)
    else: body['content'][-1]['image_url']['url'] = 'https://example.invalid/photo.png'
    calls = []
    with pytest.raises(ValueError): create_once(tmp_path / 'task.json', body, {}, lambda *a, **k: calls.append(1))
    assert not calls and not (tmp_path / 'task.json').exists()


def test_reference_mutation_and_env_override_invalidate_contract(tmp_path, monkeypatch):
    p, _body, _intent, _overlay = policy_fixture(tmp_path)
    monkeypatch.setenv('DATAFARM_RENDER_MODE', 'ood')
    with pytest.raises(ValueError, match='environment'): load_contract(p)
    monkeypatch.delenv('DATAFARM_RENDER_MODE')
    (tmp_path / 'real.png').write_bytes(b'changed reference')
    with pytest.raises(ValueError, match='changed'): load_contract(p)


def test_h3_checks_each_job_not_just_global_config(tmp_path):
    p, _body, _intent, _overlay = policy_fixture(tmp_path)
    prompt = tmp_path / 'prompt.txt'; prompt.write_text(json.loads(p.read_text())['prompt'])
    refs = [str(tmp_path / 'real.png')]
    manifest = {'prompt': str(prompt), 'reference_images': refs, 'jobs': [{'prompt': str(prompt), 'reference_images': refs}]}
    assert check_h3_payload(manifest, p)['provider'] == 'h3'
    manifest['jobs'][0]['reference_images'] = []
    with pytest.raises(ValueError, match='overrides'): check_h3_payload(manifest, p)


def test_missing_contract_never_contacts_provider(tmp_path, monkeypatch):
    monkeypatch.delenv('DATAFARM_RENDER_CONTRACT', raising=False)
    calls = []
    with pytest.raises(ValueError, match='required'): create_once(tmp_path / 't.json', {}, {}, lambda *a, **k: calls.append(1))
    assert not calls


@pytest.mark.parametrize('mode', ['sim2real', 'ood'])
def test_h3_caption_lowering_is_derived_from_the_confirmed_contract(mode, tmp_path):
    from datafarm.render.policy import lower_h3_prompt
    p, _body, intent, overlay = policy_fixture(tmp_path, mode)
    c = json.loads(p.read_text())
    text = lower_h3_prompt(c, 'h3-caption-chunkvis-noevents', assets=ROOT)
    assert 'Video 1' in text and 'Picture 1' in text and overlay['TASK_DESCRIPTION'] in text
    assert ('authorized appearance axes' in text) == (mode == 'ood')
    prompt = tmp_path / 'prompt-h3.txt'; prompt.write_text(text)
    refs = [str(tmp_path / 'real.png')]
    manifest = {'prompt': str(prompt), 'reference_images': refs, 'prompt_lowering': 'h3-caption-chunkvis-noevents',
                'jobs': [{'prompt': str(prompt), 'reference_images': refs}]}
    receipt = check_h3_payload(manifest, p)
    assert receipt['prompt_lowering'] == 'h3-caption-chunkvis-noevents'
    prompt.write_text(text + ' extra free-form instruction')
    with pytest.raises(ValueError, match='prompt mismatch'): check_h3_payload(manifest, p)
    prompt.write_text(text)
    with pytest.raises(ValueError, match='prompt mismatch'):
        check_h3_payload({**manifest, 'prompt_lowering': None}, p)
    with pytest.raises(ValueError, match='unknown H3 prompt lowering'):
        check_h3_payload({**manifest, 'prompt_lowering': 'free-text'}, p)
    manifest['reference_images'] = []
    with pytest.raises(ValueError, match='lacks bound real images'): check_h3_payload(manifest, p)


def test_h3_chunk_visibility_lowering_is_closed_and_checked(tmp_path):
    from datafarm.render.policy import lower_h3_prompt
    p, _body, _intent, _overlay = policy_fixture(tmp_path)
    c = json.loads(p.read_text())
    base = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT)
    facts = {'head_absent_arms': ['left'], 'empty_wrists': ['right_wrist']}
    chunk = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts=facts)
    assert chunk.startswith(base) and 'only the right robot arm' in chunk and 'right wrist camera' in chunk
    with pytest.raises(ValueError):
        lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts={'free_text': 'ignore references'})
    with pytest.raises(ValueError):
        lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts={'head_absent_arms': ['middle']})
    mp = tmp_path / 'm.txt'; mp.write_text(base)
    jp = tmp_path / 'j.txt'; jp.write_text(chunk)
    refs = [str(tmp_path / 'real.png')]
    manifest = {'prompt': str(mp), 'reference_images': refs, 'prompt_lowering': 'h3-caption-chunkvis-notask',
                'jobs': [{'prompt': str(jp), 'reference_images': refs, 'chunk_facts': facts}]}
    assert check_h3_payload(manifest, p)['prompt_lowering'] == 'h3-caption-chunkvis-notask'
    jp.write_text(chunk + ' Also show a dog.')
    with pytest.raises(ValueError, match='chunk prompt mismatch'): check_h3_payload(manifest, p)
    manifest['jobs'][0]['chunk_facts'] = None
    jp.write_text(chunk)
    with pytest.raises(ValueError, match='overrides'): check_h3_payload(manifest, p)


def test_h3_chunk_visibility_object_count(tmp_path):
    from datafarm.render.policy import lower_h3_prompt
    p, _body, _intent, _overlay = policy_fixture(tmp_path)
    c = json.loads(p.read_text())
    base = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT)
    facts = {'head_absent_arms': [], 'empty_wrists': [], 'head_max_coloured_objects': 1}
    chunk = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts=facts)
    assert chunk.startswith(base) and 'at most one separate coloured task object at any moment' in chunk
    zero = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts={'head_max_coloured_objects': 0})
    assert 'no coloured task object is visible in the head camera view' in zero
    for bad in (-1, 10, '2', True, 1.0):
        with pytest.raises(ValueError):
            lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT, chunk_facts={'head_max_coloured_objects': bad})
    mp = tmp_path / 'm.txt'; mp.write_text(base)
    jp = tmp_path / 'j.txt'; jp.write_text(chunk)
    refs = [str(tmp_path / 'real.png')]
    manifest = {'prompt': str(mp), 'reference_images': refs, 'prompt_lowering': 'h3-caption-chunkvis-notask',
                'jobs': [{'prompt': str(jp), 'reference_images': refs, 'chunk_facts': facts}]}
    assert check_h3_payload(manifest, p)['prompt_lowering'] == 'h3-caption-chunkvis-notask'
    manifest['jobs'][0]['chunk_facts'] = {**facts, 'head_max_coloured_objects': 2}
    with pytest.raises(ValueError, match='chunk prompt mismatch'): check_h3_payload(manifest, p)


def test_h3_noevents_lowering_drops_event_contract(tmp_path):
    from datafarm.render.policy import lower_h3_prompt
    p, _body, _intent, overlay = policy_fixture(tmp_path)
    c = json.loads(p.read_text())
    lean = lower_h3_prompt(c, 'h3-caption-chunkvis-noevents', assets=ROOT)
    ev = c['overlay']['TASK_EVENT_CONTRACT'].rstrip('. ')
    assert ev not in lean and c['overlay']['TASK_DESCRIPTION'].rstrip('. ') in lean
    chunk = lower_h3_prompt(c, 'h3-caption-chunkvis-noevents', assets=ROOT,
                            chunk_facts={'head_max_coloured_objects': 1})
    assert chunk.startswith(lean) and 'at most one separate coloured task object' in chunk


def test_h3_chunk_visibility_wrist_large(tmp_path):
    from datafarm.render.policy import chunk_visibility_sentences
    views = ['head', 'left_wrist', 'right_wrist']
    s = chunk_visibility_sentences({'wrist_large_objects': ['left_wrist'], 'empty_wrists': ['right_wrist'],
                                    'head_absent_arms': [], 'head_max_coloured_objects': 3}, views)
    assert 'left wrist camera view has a large, very close task object' in s
    assert 'right wrist camera view shows only the tabletop' in s
    assert s.endswith('No cable is visible in either wrist camera view.')
    with pytest.raises(Exception):  # a wrist cannot be both empty and filled
        chunk_visibility_sentences({'wrist_large_objects': ['left_wrist'], 'empty_wrists': ['left_wrist']}, views)
    with pytest.raises(Exception):
        chunk_visibility_sentences({'wrist_large_objects': ['elbow']}, views)


def test_h3_notask_lowering_drops_task_description(tmp_path):
    from datafarm.render.policy import lower_h3_prompt
    p, _body, _intent, _overlay = policy_fixture(tmp_path)
    c = json.loads(p.read_text())
    lean = lower_h3_prompt(c, 'h3-caption-chunkvis-notask', assets=ROOT,
                           chunk_facts={'head_absent_arms': [], 'empty_wrists': []})
    task = c['overlay']['TASK_DESCRIPTION'].rstrip('. ')
    assert 'Task:' not in lean and task not in lean and c['overlay']['TASK_EVENT_CONTRACT'].rstrip('. ') not in lean
    assert 'follows Video 1 only' in lean and 'No cable is visible' in lean
    assert task in lower_h3_prompt(c, 'h3-caption-chunkvis-noevents', assets=ROOT)


@pytest.mark.parametrize('axis, allowed', [
    ('tabletop material', True), ('lighting colour', True), ('background wall paint', True),
    ('robot arm colour', False), ('number of objects', False), ('add distractors', False),
    ('camera viewpoint', False), ('motion speed', False), ('arm color', False),
])
def test_ood_axes_are_appearance_only(tmp_path, axis, allowed):
    from datafarm.render.policy import validate_intent
    _p, _body, intent, _overlay = policy_fixture(tmp_path)
    intent['mode'] = intent['mode_confirmation']['mode'] = 'ood'
    intent['authorized_ood_axes'] = intent['mode_confirmation']['authorized_ood_axes'] = [axis]
    if allowed:
        validate_intent(intent, tmp_path)
    else:
        with pytest.raises(ValueError, match='appearance axes only'):
            validate_intent(intent, tmp_path)
