"""Ark edit transport with immutable create attempts and resumable downloads."""
import base64
import fcntl
import os
import time
from pathlib import Path
import requests
from datafarm.render.seedance.common import read, write, now, sha
from datafarm.render.seedance.references import images
from datafarm.render.seedance.upload_transport import exchange, UploadError
from datafarm.render.policy import check_seedance_payload, submission_contract_path

ENDPOINT = 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks'
UPLOAD_HOST_ENV = 'DATAFARM_SEEDANCE_UPLOAD_HOST'
SUCCESS = {'success', 'succeeded', 'completed', 'done'}
FAILED = {'failed', 'cancelled', 'canceled', 'expired'}


def upload_host(configured=None):
    """Base URL of the carrier upload host: explicit opt-in, no default.

    Seedance reads the white carrier video from a public URL, so uploading publishes
    real-scene media to that host for its expiry window.
    """
    host = (configured or os.environ.get(UPLOAD_HOST_ENV) or '').strip().rstrip('/')
    if not host:
        raise RuntimeError(f'no carrier upload host: set {UPLOAD_HOST_ENV} or the batch config upload_host to a '
                           'tempfile.org-compatible host you accept (e.g. https://tempfile.org, an anonymous public '
                           'file host). Seedance fetches the carrier from that public URL, so real-scene media '
                           'becomes downloadable by anyone with the link until it expires.')
    if not host.startswith('https://'):
        raise RuntimeError(f'{UPLOAD_HOST_ENV} must be an https URL')
    return host


def session(environment=False):
    client = requests.Session(); client.trust_env = environment
    proxy = os.environ.get('SEEDANCE_API_PROXY')
    if proxy:
        client.proxies.update(http=proxy, https=proxy)
    return client


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str) and (value.startswith(('http:', 'https:', 'data:')) or 'Bearer ' in value):
        return '<runtime-url-or-secret-redacted>'
    return value


def find(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for v in value.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(value, list):
        for v in value:
            r = find(v, key)
            if r is not None:
                return r


def reserve(run, name, interval):
    with (run / f'{name}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = run / f'{name}.json'
        previous = read(state)['epoch'] if state.exists() else 0
        time.sleep(max(0, previous + interval - time.time()))
        write(state, {'epoch': time.time(), 'updated_at': now()})


def upload(path, run, jobdir, interval, host=None):
    host = upload_host(host)  # fail before any rate slot or attempt is spent
    # Hold lock through upload and remote hash verification: no burst uploads.
    with (run / 'upload.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reserve(run, 'upload-rate', interval)
        expected = sha(path)
        for attempt in range(6):
            try:
                url,size = exchange(path, expected, jobdir, host, timeout=90)
                write(jobdir / 'upload.json', {'sha256': expected, 'remote_sha256': expected,
                                              'bytes': size, 'expiry_hours': 24, 'uploaded_at': now(),
                                              'transport':'curl-wall-clock-bounded','attempt_deadline_seconds':90})
                return url
            except (requests.RequestException, ValueError, KeyError, IndexError, AssertionError, UploadError) as error:
                write(jobdir / 'upload-error.json', {'type': type(error).__name__, 'attempt': attempt + 1, 'at': now()})
                if attempt == 5:
                    raise RuntimeError('bounded upload attempts exhausted') from None
                time.sleep(5 * (attempt + 1))


def inline(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')


def payload(model, job, carrier_url):
    return {'model': model, 'content': [
        {'type': 'text', 'text': job['video_prompt']},
        {'type': 'video_url', 'video_url': {'url': carrier_url}, 'role': 'reference_video'},
        *[{'type': 'image_url', 'image_url': {'url': inline(p)}, 'role': 'reference_image'}
          for p in (r['path'] for r in images(job))]],
        'generate_audio': False, 'resolution': '1080p', 'ratio': 'adaptive', 'duration': -1,
        'watermark': False, 'seed': job['seed'], 'inner_generation_options': {'pe_classification': 'edit'}}


def create_once(task_path, body, contract, post=None):
    """Persist intent before sending. Any existing no-ID record blocks another POST."""
    policy_path = Path(contract['render_contract']) if contract.get('render_contract') else submission_contract_path()
    policy = check_seedance_payload(body, policy_path)
    contract = {**contract, 'render_policy': policy}
    if task_path.exists():
        old = read(task_path)
        if old['contract'] != contract:
            raise RuntimeError('immutable attempt contract changed')
        return old
    record = {'status': 'submitting', 'created_at': now(), 'updated_at': now(), 'contract': contract}
    write(task_path, record)
    try:
        response = (post or session().post)(ENDPOINT, json=body,
                    headers={'Authorization': 'Bearer ' + os.environ['ARK_API_KEY']}, timeout=(45, 120))
        try:
            result = response.json()
        except ValueError:
            result = {'error': 'non-json response'}
        record.update(http_status=response.status_code, response=scrub(result))
        if response.status_code >= 400:
            record['status'] = 'create_rejected'
        else:
            identifier = find(result, 'task_id') or find(result, 'id') or find(result, 'taskId')
            record.update(task_id=identifier, status='submitted' if identifier else 'ambiguous_no_id')
    except requests.RequestException as error:
        record.update(status='ambiguous_transport', error_type=type(error).__name__)
    record['updated_at'] = now(); write(task_path, record)
    return record


def poll_download(task_path, raw):
    record = read(task_path); identifier = record.get('task_id')
    if not identifier:
        raise RuntimeError('no task ID; will not resubmit this source')
    if raw.exists():
        return True
    client = session()
    response = client.get(f'{ENDPOINT}/{identifier}', headers={'Authorization': 'Bearer ' + os.environ['ARK_API_KEY']}, timeout=(45, 90))
    response.raise_for_status(); body = response.json()
    state = str(find(body, 'status') or 'unknown').lower()
    record.update(status=state, updated_at=now(), last_response=scrub(body))
    write(task_path, record)
    if state in FAILED:
        raise RuntimeError('terminal provider failure: ' + state)
    if state not in SUCCESS:
        return False
    url = find(body, 'video_url') or find(body, 'videoUrl') or find(body, 'url')
    if isinstance(url, dict):
        url = url.get('url')
    if not isinstance(url, str) or not url.startswith('http'):
        raise RuntimeError('provider success lacks video URL')
    remote = client.get(url, stream=True, timeout=(30, 240)); remote.raise_for_status()
    partial = raw.with_suffix('.partial')
    with partial.open('wb') as output:
        for block in remote.iter_content(1 << 20):
            if block:
                output.write(block)
    partial.replace(raw)
    record['download'] = {'sha256': sha(raw), 'bytes': raw.stat().st_size, 'at': now()}
    write(task_path, record)
    return True
