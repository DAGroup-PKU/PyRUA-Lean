import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datafarm.render.seedance.client import create_once, payload
from datafarm.render.seedance.common import sha
from datafarm.render.seedance.references import contract, images
from datafarm.render.policy import check_seedance_payload, compile_prompt

ROOT = Path(__file__).resolve().parents[1]


class FullSceneReferenceTests(unittest.TestCase):
    def fixture(self, root):
        assets = {}
        for index, name in enumerate(('real-head', 'real-left-wrist', 'real-right-wrist',
                                      'source', 'continuity')):
            path = root / f'{name}.png'
            Image.new('RGB', (8, 8), (index * 30, 40, 80)).save(path)
            assets[name] = {'path': str(path), 'sha256': sha(path)}
        previous = root / 'previous.mp4'
        previous.write_bytes(b'previous-render')
        job = {
            'id': 'generic-task-segment-01',
            'seed': 42,
            'video_prompt': 'full-scene prompt',
            'source': {
                'structure': assets['source'],
                'carrier': {'path': 'carrier.mp4', 'sha256': 'carrier-hash'},
                'frames': 575,
                'transport_frames': 577,
                'global_source_range': [529, 1104],
            },
            'real_references': [
                assets['real-head'], assets['real-left-wrist'],
                assets['real-right-wrist'],
            ],
            'continuity_reference': {
                'image': assets['continuity'],
                'previous_render': {'path': str(previous), 'sha256': sha(previous)},
                'previous_global_range': [0, 625],
                'global_frame': 529,
                'previous_local_frame': 529,
            },
        }
        job['references'] = [
            {'role': 'real_scene_reference', **authority}
            for authority in job['real_references']
        ] + [{'role': 'continuity_same_state', **assets['continuity']}]
        return job, assets

    def test_three_real_references_and_continuity_reach_contract_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, _ = self.fixture(root)
            selected = images(job)
            self.assertEqual(
                [reference['role'] for reference in selected],
                ['real_scene_reference'] * 3 + ['continuity_same_state'],
            )

            frozen = contract(job, {'model': 'model'})
            self.assertEqual(frozen['reference_schema'], 'full-scene-v1')
            self.assertEqual(
                [reference['sha256'] for reference in frozen['references'][:3]],
                [reference['sha256'] for reference in job['real_references']],
            )

            body = payload('model', job, 'https://carrier.example')
            encoded = [part['image_url']['url'] for part in body['content']
                       if part['type'] == 'image_url']
            hashes = [hashlib.sha256(base64.b64decode(value.split(',', 1)[1])).hexdigest()
                      for value in encoded]
            self.assertEqual(
                hashes[:3],
                [reference['sha256'] for reference in job['real_references']],
            )
            self.assertEqual(len(hashes), 4)

    def test_three_real_references_pass_policy_while_reorder_or_omission_blocks_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, assets = self.fixture(root)
            episode = root / 'episode.mp4'
            episode.write_bytes(b'synthetic episode source')
            views = ['head', 'left_wrist', 'right_wrist']
            real_references = []
            for view, authority in zip(views, job['real_references']):
                real_references.append({**authority, 'provenance': {
                    'kind': 'real-episode-frame',
                    'source': {'path': str(episode), 'sha256': sha(episode)},
                    'frame': 0, 'description': f'synthetic {view} fixture'},
                    'scope': 'full-view', 'views': [view],
                    'view_regions': {view: [0, 0, 8, 8]},
                    'coverage': ['full_scene', 'robot_body', 'attached_cameras_and_cables',
                                 'task_objects', 'table_and_background', 'photographic_character']})
            intent = {'schema': 'datafarm.render-intent/1', 'mode': 'sim2real',
                      'run_id': 'generic-run', 'task_id': 'generic-task',
                      'mode_confirmation': {'mode': 'sim2real', 'run_id': 'generic-run',
                                            'task_id': 'generic-task', 'quote': 'test only',
                                            'source': 'fixture', 'authorized_ood_axes': []},
                      'real_baseline': 'synthetic three-view baseline', 'authorized_ood_axes': [],
                      'appearance_views': views,
                      'appearance_requirements': {
                          area: {'required': True, 'description': 'fixture requirement'}
                          for area in ['robot_body', 'attached_cameras_and_cables', 'task_objects',
                                       'table_and_background', 'photographic_character']},
                      'appearance_invariants': ['robot identity'],
                      'real_references': real_references}
            overlay = {'TASK_DESCRIPTION': 'generic manipulation',
                       'SOURCE_VIDEO_ROLE': 'motion carrier',
                       'CAMERA_TIMING_CONTRACT': 'preserve declared views and timing',
                       'TASK_EVENT_CONTRACT': 'preserve source events',
                       'CONTINUITY_CONTRACT': 'preserve source state'}
            compiled = compile_prompt(root, intent, overlay, assets=ROOT)
            policy = root / 'render-contract.json'
            policy.write_text(json.dumps(compiled))
            job.update(video_prompt=compiled['prompt'], render_contract=str(policy),
                       real_references=real_references)
            job['references'] = [
                {'role': 'real_scene_reference', **reference} for reference in real_references
            ] + [{'role': 'continuity_same_state', **assets['continuity']}]
            body = payload('model', job, 'https://carrier.example')
            proof = check_seedance_payload(body, policy)
            self.assertEqual(proof['actual_image_sha256s'][:3],
                             [reference['sha256'] for reference in real_references])

            for mutation in ('reordered', 'missing'):
                changed = copy.deepcopy(body)
                if mutation == 'reordered':
                    changed['content'][2], changed['content'][3] = changed['content'][3], changed['content'][2]
                else:
                    changed['content'].pop(4)
                calls = []
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    create_once(root / f'{mutation}.json', changed,
                                {'render_contract': str(policy)},
                                lambda *args, calls=calls, **kwargs: calls.append(1))
                self.assertEqual(calls, [])

    def test_optional_source_and_continuity_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, assets = self.fixture(root)
            real = job['references'][:3]
            for suffix in (
                    [],
                    [{'role': 'source_frame_zero', **assets['source']}],
                    [{'role': 'continuity_same_state', **assets['continuity']}],
                    [{'role': 'source_frame_zero', **assets['source']},
                     {'role': 'continuity_same_state', **assets['continuity']}],
            ):
                job['references'] = real + suffix
                self.assertEqual(images(job), job['references'])

    def test_rejects_reordered_missing_or_replaced_real_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, assets = self.fixture(root)
            original = list(job['references'])

            job['references'] = [original[1], original[0], *original[2:]]
            with self.assertRaises(ValueError):
                images(job)

            job['references'] = [original[0], original[1], original[3]]
            with self.assertRaises(ValueError):
                images(job)

            job['references'] = list(original)
            job['references'][1] = {
                'role': 'real_scene_reference', **assets['source'],
            }
            with self.assertRaises(ValueError):
                images(job)

    def test_rejects_continuity_for_wrong_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, _ = self.fixture(root)
            job['continuity_reference']['previous_local_frame'] = 528
            with self.assertRaises(ValueError):
                images(job)

    def test_real_authorities_require_explicit_references(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _ = self.fixture(Path(directory))
            del job['references']
            with self.assertRaises(ValueError):
                images(job)


if __name__ == '__main__':
    unittest.main()
