"""CPU walk through the H3 tools around the GPU worker, with chunk renders replaced by copies of the white clips.

Build -> per-chunk tickets -> seed screen composites and merge -> selection, assembly, exposure lock -> score and pack
-> OOD restyle, plus orphan requeue and clean references.  The evaluator and the HED tool are stubbed (they need models).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_h3_control_hybrid import HYBRID, clean_env, import_hybrid, make_carrier, no_bindings, recipe_config, synthetic_bundle  # noqa: F401

pytestmark = pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='needs ffmpeg/ffprobe')
TOTAL = 220


def built_packages(no_bindings, tmp_path):  # noqa: F811
    package = import_hybrid('h3_package')
    run = tmp_path / 'run'
    for d in ('run/recipes', 'run/tasks/unseen-task/refs', 'run/queue', 'models', 'vx'):
        (tmp_path / d).mkdir(parents=True)
    carrier = tmp_path / 'carrier.mp4'
    make_carrier(carrier, TOTAL)
    bundle = run / 'tasks/unseen-task/bundle.json'
    bundle.write_text(json.dumps(synthetic_bundle(tmp_path, carrier)))
    recipe = run / 'recipes/test-hybrid.json'
    recipe.write_text(json.dumps({**recipe_config(), 'name': 'test-hybrid'}))
    no_bindings.setenv('H3_MODELS', str(tmp_path / 'models'))
    no_bindings.setenv('H3_VIDEOX_FUN', str(tmp_path / 'vx'))
    no_bindings.setenv('H3_RUN', str(run))
    no_bindings.setattr(package.h3_paths, '_RUN_DIR', None)

    def fake_hed(job, out_dir, ctl, threads, device):
        target = out_dir / f"{job['chunk_id']}-hed-soft.mp4"
        shutil.copy(job['white'], target)
        return {'path': str(target), 'sha256': package.sha256(target), 'min_edge_intensity': 0,
                'mean_active_fraction_gt20': 0.0, 'mean_normalized_intensity': 0.0, 'device': 'stub',
                'tool_sha256': 'stub'}

    no_bindings.setattr(package, 'make_hed', fake_hed)
    packages = []
    for offset, tag in ((0, 'ep0-sim2real'), (1, 'ep0-sim2real-s1')):
        out = run / 'pkgs/test-hybrid/unseen-task' / tag
        assert package.cmd_build(SimpleNamespace(
            bundle=str(bundle), recipe=str(recipe), episode='ep0', mode='sim2real', variant=None, seed_offset=offset,
            out=str(out), force=False, hed_jobs=1, hed_threads=1, hed_device='cpu')) == 0
        packages.append(out)
    return run, packages


def fake_render(pkg):
    manifest = json.loads((pkg / 'job-manifest.json').read_text())
    for job in manifest['jobs']:
        chunk = pkg / 'render/chunks' / job['chunk_id']
        chunk.mkdir(parents=True)
        shutil.copy(job['white'], chunk / 'chunk-render.mp4')
        (chunk / 'segment-manifest.json').write_text('{}')
    return manifest


def test_build_finalize_score_and_ood_on_cpu(no_bindings, tmp_path):  # noqa: F811
    run, packages = built_packages(no_bindings, tmp_path)
    ticket = json.loads((packages[0] / 'ticket.json').read_text())
    assert ticket['profile'] == 'control-turbo' and ticket['package'] == str(packages[0])

    enqueue = import_hybrid('h3_enqueue')
    tickets = enqueue.chunk_tickets(ticket)
    assert [suffix for suffix, _ in tickets] == ['-c00', '-c01']
    one_job = json.loads(Path(tickets[1][1]['manifest']).read_text())
    assert [j['chunk_index'] for j in one_job['jobs']] == [1]

    manifests = [fake_render(p) for p in packages]
    assert enqueue.chunk_tickets(ticket) == []  # rendered chunks get no ticket
    chunks = len(manifests[0]['jobs'])

    screen = import_hybrid('h3_screen_prep')
    screen_dir = run / 'screen/unseen-task/ep0'
    screen.composites(SimpleNamespace(cands=[str(p) for p in packages], out=str(screen_dir)))
    assert (screen_dir / 'c01-cand1.jpg').is_file()
    (screen_dir / 'part-0.json').write_text(json.dumps({'chunks': [
        {'chunk': k, 'choice': k % 2, 'scores': [0, 1]} for k in range(chunks)]}))
    screen.merge(SimpleNamespace(screen_dir=str(screen_dir), chunks=chunks, screener='test-model'))
    assert json.loads((screen_dir / 'choice.json').read_text())['schema'] == 'h3.seed-screen/1'

    finalize = import_hybrid('h3_finalize')
    scored = []

    def fake_score(generated, carrier, out, task, run_dir):
        scored.append(Path(out).parent.parent.parent.name)
        Path(out).write_text('{}')

    no_bindings.setattr(finalize, 'score', fake_score)
    final = finalize.select(SimpleNamespace(
        render_recipe='test-hybrid', task='unseen-task', episode_tag='ep0-sim2real', name='test-vlm',
        choice_file=Path('screen/unseen-task/ep0/choice.json'), seeds=['s0', 's1']), run)
    assert final == run / 'pkgs/exposure-lock-test-vlm/unseen-task/ep0-sim2real'
    assert scored == ['test-vlm', 'exposure-lock-test-vlm']
    media = import_hybrid('h3_media')
    assembled = json.loads((final / 'assembled.json').read_text())
    assert media.probe(assembled['final'])[:3] == (1280, 1080, TOTAL)
    assert media.probe(assembled['carrier_clip']).frames == TOTAL
    lock_record = json.loads((final / 'exposure-lock.json').read_text())
    assert lock_record['schema_version'] == 'h3.chunk-photometric-lock/1' and lock_record['geometry_changed'] is False
    selected = json.loads((run / 'pkgs/test-vlm/unseen-task/ep0-sim2real/package.json').read_text())
    assert selected['selection']['choice'] == [k % 2 for k in range(chunks)]

    pack = import_hybrid('h3_score_and_pack')
    calls = []

    def fake_eval(*args):
        calls.append(args[0])
        out = Path(args[args.index('--out') + 1])
        if args[0] == 'score':
            out.write_text('{}')
        else:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'manifest.json').write_text('{}')

    no_bindings.setattr(pack, 'run_eval', fake_eval)
    no_bindings.setenv('H3_EVAL_DINO', str(tmp_path))
    assert pack.process(final, run).startswith('ok')
    assert calls == ['score', 'pack'] and pack.process(final, run).startswith('skip')

    out = tmp_path / 'ood'
    r = subprocess.run([sys.executable, str(HYBRID / 'h3_ood_restyle.py'), '--src', str(final), '--out', str(out),
                        '--material-bgr', '104,98,92', '--light', 'cool-6500k'],
                       capture_output=True, text=True, env=clean_env(), timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    restyled = json.loads((out / 'assembled.json').read_text())
    assert restyled['frames'] == TOTAL and media.probe(restyled['final'])[:3] == (1280, 1080, TOTAL)
    assert json.loads((out / 'restyle.json').read_text())['schema'] == 'h3.ood-restyle/1'


def draw_red_box(video):
    """Re-encode a chunk render with a saturated red box in the head view: an object the carrier does not have."""
    tmp = video.with_suffix('.tmp.mp4')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video), '-vf',
                    'drawbox=x=300:y=100:w=140:h=140:color=red:t=fill', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p', str(tmp)], check=True)
    tmp.replace(video)


def test_unusable_seed_screen_falls_back_to_the_occupancy_screen(no_bindings, tmp_path):  # noqa: F811
    """A failed VLM screen (no or partial choice.json) must not stop the episode: the occupancy screen picks, per chunk,
    the candidate without invented content, and the package records the fallback and its reason."""
    run, packages = built_packages(no_bindings, tmp_path)
    manifests = [fake_render(p) for p in packages]
    chunks = len(manifests[0]['jobs'])
    assert chunks == 2
    draw_red_box(packages[1] / 'render/chunks' / manifests[1]['jobs'][0]['chunk_id'] / 'chunk-render.mp4')
    draw_red_box(packages[0] / 'render/chunks' / manifests[0]['jobs'][1]['chunk_id'] / 'chunk-render.mp4')

    finalize = import_hybrid('h3_finalize')
    no_bindings.setattr(finalize, 'score', lambda generated, carrier, out, task, run_dir: Path(out).write_text('{}'))
    partial = run / 'screen/unseen-task/ep0/choice.json'
    partial.parent.mkdir(parents=True)
    partial.write_text(json.dumps({'schema': 'h3.seed-screen/1', 'screener': 'test-model',
                                   'chunks': [{'chunk': 0, 'choice': 0, 'scores': [0, 1]}]}))
    for choice_file, reason in ((Path('screen/unseen-task/ep0/choice.json'), 'expected 0..1'),
                                (None, 'no choice file given')):
        final = finalize.select(SimpleNamespace(
            render_recipe='test-hybrid', task='unseen-task', episode_tag='ep0-sim2real', name='test-vlm',
            choice_file=choice_file, seeds=['s0', 's1']), run)
        assert final == run / 'pkgs/exposure-lock-test-vlm/unseen-task/ep0-sim2real'
        selection = json.loads((run / 'pkgs/test-vlm/unseen-task/ep0-sim2real/package.json').read_text())['selection']
        assert selection['mode'] == 'occupancy' and selection['choice'] == [0, 1]
        assert reason in selection['fallback_reason']
        assert selection['penalties'][0][1] > selection['penalties'][0][0]
        recipe = json.loads((run / 'recipes/test-vlm.json').read_text())
        assert recipe['selection']['mode'] == 'occupancy' and 'fallback' in recipe['single_variable']

    select = import_hybrid('h3_select')
    assert select.choice_problem({'schema': 'h3.seed-screen/1', 'chunks': [
        {'chunk': 0, 'choice': 2, 'scores': [0, 1]}]}, 1, 2) == 'chunk 0 chooses candidate 2 of 2'
    assert 'schema' in select.choice_problem({'chunks': []}, 0, 2)
    assert select.choice_problem({'schema': 'h3.seed-screen/1', 'chunks': 'x'}, 1, 2).startswith('malformed')


def test_rejected_build_leaves_no_ticket(no_bindings, tmp_path):  # noqa: F811
    package = import_hybrid('h3_package')
    no_bindings.setattr(package, 'audit', lambda pkg: (_ for _ in ()).throw(ValueError('rejected')))
    with pytest.raises(ValueError, match='rejected'):
        built_packages(no_bindings, tmp_path)
    out = tmp_path / 'run/pkgs/test-hybrid/unseen-task/ep0-sim2real'
    assert not (out / 'ticket.json').exists() and not (out / 'package.json').exists()


def test_requeue_orphans_fails_closed_on_an_unknown_lease_table(tmp_path):
    run = tmp_path / 'run'
    (run / 'queue/running').mkdir(parents=True)
    claimed = run / 'queue/running/001-a@node-1-777-control-turbo.json'
    live = run / 'queue/running/001-b@node-1-888-control-turbo.json'
    for path in (claimed, live):
        path.write_text(json.dumps({'profile': 'control-turbo'}))
    sched = tmp_path / 'sched'
    sched.write_text('#!/bin/sh\necho "lease directory unavailable" >&2\nexit 1\n')
    sched.chmod(0o755)
    tool = [sys.executable, str(HYBRID / 'h3_requeue_orphans.py'), '--run-dir', str(run)]
    env = clean_env(GPU_SCHEDULER=str(sched))
    r = subprocess.run(tool, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and claimed.exists() and live.exists()
    sched.write_text('#!/bin/sh\necho "h3-control-turbo-20261103-000000-ab 1 node-1 x 888"\n')
    r = subprocess.run(tool, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert not claimed.exists() and live.exists()
    assert (run / 'queue/pending/control-turbo/001-a.json').is_file()


def test_clean_references_add_no_object_sheet_by_default(no_bindings, tmp_path):  # noqa: F811
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    run = tmp_path / 'run'
    tdir = run / 'tasks/unseen-task'
    (tdir / 'refs').mkdir(parents=True)
    no_bindings.setattr(import_hybrid('h3_paths'), '_RUN_DIR', None)
    source = tmp_path / 'source.mp4'
    make_carrier(source, 12)
    robot = tmp_path / 'robot-sheet.png'
    objects = tmp_path / 'object-sheet.png'
    for path in (robot, objects):
        cv2.imwrite(str(path), np.full((720, 1280, 3), 128, np.uint8))
    provenance = {'description': 'real', 'source': str(source)}
    base = {'real_references': [
        {'path': str(source), 'scope': 'full-view', 'coverage': ['full_scene', 'task_objects', 'robot_body'],
         'provenance': provenance},
        {'path': str(robot), 'scope': 'local-detail', 'coverage': ['robot_body'], 'provenance': provenance},
        {'path': str(objects), 'scope': 'local-detail', 'coverage': ['task_objects'], 'provenance': provenance}]}
    (tdir / 'bundle.json').write_text(json.dumps(base))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'sources': {v: str(source) for v in ('head', 'left_wrist', 'right_wrist')},
                                'head_plate': {'mode': 'frame', 'frame': 3},
                                'wrist_frames': {'left_wrist': 1, 'right_wrist': 2}}))
    r = subprocess.run([sys.executable, str(HYBRID / 'h3_clean_references.py'), 'unseen-task', '--base', 'bundle.json',
                        '--spec', str(spec), '--run-dir', str(run)], capture_output=True, text=True, env=clean_env())
    assert r.returncode == 0, r.stderr[-2000:]
    made = json.loads((tdir / 'bundle-clean.json').read_text())
    assert [Path(x['path']).name for x in made['real_references']] == ['clean-scene-3view.png', 'clean-robot-sheet.png']
    assert not (tdir / 'refs/clean-object-sheet.png').exists()
