"""Explicit reference roles for render jobs."""
import hashlib
import json

from datafarm.render.seedance.common import verify_file

# Device-state appearance references (e.g. a display unlit / lit).  They show how a
# state and product look, never where the device is. Local product shape comes from
# the real references; placement, motion/contact and state timing come from the
# carrier video. Order is fixed: off, on, then the source frame zero.
STATE_ROLES = ('device_state_off', 'device_state_on')
REAL_SCENE_ROLE = 'real_scene_reference'


def _same_authority(reference, authority):
    if not isinstance(reference, dict) or not isinstance(authority, dict):
        raise TypeError('reference authority must be an object')
    if any(key not in reference or key not in authority for key in ('path', 'sha256')):
        raise ValueError('reference authority requires path and sha256')
    if any(reference[key] != authority[key] for key in ('path', 'sha256')):
        raise ValueError('reference differs from declared authority')


def _continuity_authority(job):
    continuity = job['continuity_reference']
    authority = continuity['image']
    verify_file(continuity['previous_render'])
    lo, hi = continuity['previous_global_range']
    start = job['source']['global_source_range'][0]
    if (continuity['global_frame'] != start or not lo <= start < hi or
            continuity['previous_local_frame'] != start - lo):
        raise ValueError('continuity reference must depict current source frame zero')
    return authority


def _full_scene_images(job, refs):
    authorities = job['real_references']
    if not isinstance(authorities, list) or not authorities:
        raise ValueError('real_references must be a non-empty ordered list')
    count = len(authorities)
    roles = [ref.get('role') for ref in refs]
    suffix = roles[count:]
    if roles[:count] != [REAL_SCENE_ROLE] * count or suffix not in (
            [], ['source_frame_zero'], ['continuity_same_state'],
            ['source_frame_zero', 'continuity_same_state']):
        raise ValueError('unsupported full-scene reference roles or order')
    for reference, authority in zip(refs[:count], authorities):
        _same_authority(reference, authority)
    for reference in refs[count:]:
        if reference['role'] == 'source_frame_zero':
            authority = job['source']['structure']
        else:
            authority = _continuity_authority(job)
        _same_authority(reference, authority)
    return refs


def images(job):
    if 'references' not in job:
        raise ValueError('a render job needs an explicit ordered references list')
    refs = job['references']
    if 'real_references' in job:
        return _full_scene_images(job, refs)
    roles = [r['role'] for r in refs]
    if roles not in ([], ['source_frame_zero'], ['object_appearance', 'source_frame_zero'],
                     ['object_appearance', 'source_frame_zero', 'continuity_same_state'],
                     ['object_appearance', 'continuity_same_state'],
                     ['continuity_same_state'],
                     list(STATE_ROLES), list(STATE_ROLES) + ['source_frame_zero']):
        raise ValueError('unsupported text-material reference roles')
    for ref in refs:
        if ref['role'] == 'object_appearance':
            authority = job['objects']
        elif ref['role'] in STATE_ROLES:
            authority = job['state_references'][ref['role']]
        elif ref['role'] == 'continuity_same_state':
            authority = _continuity_authority(job)
        else:
            authority = job['source']['structure']
        _same_authority(ref, authority)
    return refs


def contract(job, config):
    row = job['source']
    base = {'recipe_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
            'source_sha256': row['carrier']['sha256'], 'frames': row['frames'],
            'transport_frames': row['transport_frames'], 'model': config['model']}
    if job.get('render_contract'):
        base['render_contract'] = job['render_contract']
    refs = images(job)
    for ref in refs:
        verify_file(ref)
    base.update(reference_schema=('full-scene-v1' if 'real_references' in job else 'text-material-v1'),
                references=[{'role': r['role'], 'sha256': r['sha256']} for r in refs],
                image_roles=[r['role'] for r in refs])
    return base
