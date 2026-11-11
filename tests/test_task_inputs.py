"""Inputs for a new embodiment and initial placements: robot package checks, LeRobot episode import, placement plans."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from datafarm.simulation import placement
from datafarm.simulation.robot_package import PackageError, check_package, safe_extract
from datafarm.task.lerobot_episode import EpisodeImportError, import_episode

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}

URDF = '''<robot name="armbot">
  <link name="base"><visual><geometry><mesh filename="package://armbot_pkg/meshes/base.stl"/></geometry></visual></link>
  <link name="l1"><visual><geometry><mesh filename="../meshes/l1.stl"/></geometry></visual></link>
  <link name="l2"><visual><geometry><mesh filename="package://armbot_pkg/meshes/missing.stl"/></geometry></visual></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><limit lower="-1" upper="1"/></joint>
  <joint name="j2" type="prismatic"><parent link="l1"/><child link="l2"/></joint>
</robot>'''


def robot_folder(tmp_path):
    pkg = tmp_path / 'pkg' / 'armbot_pkg'
    (pkg / 'meshes').mkdir(parents=True)
    (pkg / 'urdf').mkdir()
    for name in ('base.stl', 'l1.stl'):
        (pkg / 'meshes' / name).write_text('solid')
    (pkg / 'urdf' / 'armbot.urdf').write_text(URDF)
    (pkg / 'hand_eye_calibration.json').write_text('{}')
    return tmp_path / 'pkg'


def test_robot_package_report_names_what_blocks_grounding(tmp_path):
    report = check_package(robot_folder(tmp_path))
    main = report['urdfs'][0]
    assert report['main'] == 'armbot_pkg/urdf/armbot.urdf' and report['ok'] is False
    assert main['root_link'] == 'base' and main['actuated_names'] == ['j1', 'j2']
    assert main['joints_without_limits'] == ['j2']
    assert main['missing_meshes'] == ['package://armbot_pkg/meshes/missing.stl']
    assert report['calibration_files'] == ['armbot_pkg/hand_eye_calibration.json']
    (tmp_path / 'pkg/armbot_pkg/meshes/missing.stl').write_text('solid')
    assert check_package(tmp_path / 'pkg')['ok'] is True
    r = subprocess.run([sys.executable, str(ROOT / 'tools/check_robot_package.py'), str(tmp_path / 'pkg')],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0 and json.loads(r.stdout)['main'] == 'armbot_pkg/urdf/armbot.urdf'


def test_robot_package_tree_errors(tmp_path):
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / 'r.urdf').write_text('<robot name="x"><link name="a"/><link name="b"/>'
                                '<joint name="j" type="fixed"><parent link="a"/><child link="c"/></joint></robot>')
    problems = check_package(bad)['urdfs'][0]['problems']
    assert any("child link 'c'" in p for p in problems) and any('2 root links' in p for p in problems)
    (bad / 'r.urdf').unlink()
    (bad / 'r.xacro').write_text('<robot/>')
    with pytest.raises(PackageError, match='only .xacro'):
        check_package(bad)


def test_zip_extraction_refuses_unsafe_paths_and_skips_links(tmp_path):
    good = tmp_path / 'good.zip'
    with zipfile.ZipFile(good, 'w') as zf:
        zf.writestr('armbot/urdf/a.urdf', '<robot name="a"><link name="base"/></robot>')
        zf.writestr('__MACOSX/._a.urdf', 'junk')
        link = zipfile.ZipInfo('armbot/escape')
        link.external_attr = (0o120777 << 16)
        zf.writestr(link, '/etc/passwd')
    assert safe_extract(good, tmp_path / 'out') == ['armbot/urdf/a.urdf']
    assert not (tmp_path / 'out/armbot/escape').exists() and not (tmp_path / 'out/__MACOSX').exists()
    assert check_package(good, tmp_path / 'out2')['ok'] is True
    with pytest.raises(PackageError, match='--extract-to'):
        check_package(good)
    for name in ('../evil.txt', '/abs.txt'):
        evil = tmp_path / 'evil.zip'
        with zipfile.ZipFile(evil, 'w') as zf:
            zf.writestr(name, 'x')
        with pytest.raises(PackageError, match='unsafe path'):
            safe_extract(evil, tmp_path / 'out3')
    assert not (tmp_path / 'evil.txt').exists()


def lerobot_dataset(tmp_path):
    root = tmp_path / 'ds'
    info = {'codebase_version': 'v2.1', 'fps': 30, 'chunks_size': 1000, 'robot_type': 'armbot',
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': {'observation.images.head': {'dtype': 'video'},
                         'observation.state': {'dtype': 'float32', 'shape': [14], 'names': ['j%d' % i for i in range(14)]},
                         'action': {'dtype': 'float32', 'shape': [14]}}}
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/episodes.jsonl').write_text(
        json.dumps({'episode_index': 0, 'tasks': ['other'], 'length': 10}) + '\n'
        + json.dumps({'episode_index': 3, 'tasks': ['scoop the cereal'], 'length': 120}) + '\n')
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000/episode_000003.parquet').write_bytes(b'not really parquet')
    (root / 'videos/chunk-000/observation.images.head').mkdir(parents=True)
    (root / 'videos/chunk-000/observation.images.head/episode_000003.mp4').write_bytes(b'head video')
    return root


def test_lerobot_episode_becomes_a_self_contained_input(tmp_path):
    ds = lerobot_dataset(tmp_path)
    wrist = tmp_path / 'left.mp4'
    wrist.write_bytes(b'left video')
    out = tmp_path / 'episode'
    rec = import_episode(ds, 3, out, {'cam_high': 'observation.images.head', 'cam_left_wrist': str(wrist)},
                         robot_folder(tmp_path))
    assert rec['tasks'] == ['scoop the cereal'] and rec['fps'] == 30 and rec['length'] == 120
    assert rec['cameras']['cam_high'] == {'video_key': 'observation.images.head', 'file': 'videos/cam_high.mp4'}
    assert (out / 'videos/cam_left_wrist.mp4').read_bytes() == b'left video'
    assert (out / 'robot/armbot_pkg/urdf/armbot.urdf').is_file() and rec['robot_package'] == 'robot'
    assert rec['features']['observation.state']['shape'] == [14] and 'observation.images.head' not in rec['features']
    assert set(rec['files']) >= {'videos/cam_high.mp4', 'data/episode_000003.parquet', 'meta/info.json'}
    assert json.loads((out / 'meta/episode.jsonl').read_text())['episode_index'] == 3
    assert json.loads((out / 'episode.json').read_text())['schema'] == 'datafarm.lerobot-episode/1'
    with pytest.raises(EpisodeImportError, match='not empty'):
        import_episode(ds, 3, out)
    with pytest.raises(EpisodeImportError, match='does not exist'):
        import_episode(ds, 4, tmp_path / 'e4')
    info = json.loads((ds / 'meta/info.json').read_text())
    (ds / 'meta/info.json').write_text(json.dumps({**info, 'codebase_version': 'v3.0'}))
    with pytest.raises(EpisodeImportError, match='v3.0'):
        import_episode(ds, 3, tmp_path / 'e5')


def test_lerobot_import_cli_feeds_prepare(tmp_path):
    ds = lerobot_dataset(tmp_path)
    out = tmp_path / 'episode'
    r = subprocess.run([sys.executable, str(ROOT / 'tools/import_lerobot_episode.py'), '--dataset', str(ds),
                        '--episode', '3', '--out', str(out)], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['cameras']['head']['file'] == 'videos/head.mp4'
    request = tmp_path / 'request.txt'
    request.write_text('make data')
    r = subprocess.run([sys.executable, str(ROOT / 'tools/pipeline_agent.py'), 'prepare', '--episode', str(out),
                        '--request-file', str(request), '--target-count', '2', '--stop-after', 'white',
                        '--output', str(tmp_path / 'task/work')], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    files = json.loads((tmp_path / 'task/work/input.json').read_text())['files']
    assert any(str(f.get('path', f)).endswith('episode.json') for f in files)


SCENE = {'objects_initial': {'cup_a': {'xy': [0.10, 0.00]}, 'cup_b': {'x': 0.25, 'y': 0.05}, 'bowl': [0.0, 0.2, 0.75]}}
REACH = {'both_arms': {'x': [-0.2, 0.4], 'y': [-0.1, 0.35]}}


def test_local_plan_jitters_around_the_nominals_and_keeps_spacing():
    plan = placement.generate_plan(SCENE, 'local_robustness', 50, 7, jitter_m=0.03)
    assert plan == placement.generate_plan(SCENE, 'local_robustness', 50, 7, jitter_m=0.03)
    assert plan['digest'] != placement.generate_plan(SCENE, 'local_robustness', 50, 8)['digest']
    for ep in plan['episodes']:
        for name, (x, y) in ep['object_xy'].items():
            nx, ny = plan['nominals'][name]
            assert abs(x - nx) <= 0.03 and abs(y - ny) <= 0.03
    assert all(c['x_span_m'] <= 0.06 for c in plan['coverage'].values())


def test_broad_plan_covers_the_reachable_table():
    plan = placement.generate_plan(SCENE, 'broad_table', 100, 7, reachable=REACH, min_dist_m=0.08)
    for ep in plan['episodes']:
        pts = list(ep['object_xy'].values())
        assert all(-0.2 <= x <= 0.4 and -0.1 <= y <= 0.35 for x, y in pts)
        assert min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5
                   for i, a in enumerate(pts) for b in pts[i + 1:]) >= 0.08
    assert all(c['x_span_m'] > 0.4 for c in plan['coverage'].values())
    with pytest.raises(ValueError, match='reachable_tabletop_xy'):
        placement.generate_plan(SCENE, 'broad_table', 5, 1)
    with pytest.raises(ValueError, match='closer than'):
        placement.generate_plan({'objects_initial': {'a': [0, 0], 'b': [0.01, 0]}}, 'local_robustness', 5, 1)


def test_default_mode_follows_the_goal():
    assert placement.default_mode(5) == 'local_robustness'
    assert placement.default_mode(100) == 'broad_table'
    assert placement.default_mode(20, 'data for policy training') == 'broad_table'
    assert placement.default_mode(100, 'a validation run') == 'local_robustness'


def test_placement_cli(tmp_path):
    scene, ground = tmp_path / 'scene.json', tmp_path / 'grounding.json'
    scene.write_text(json.dumps(SCENE))
    ground.write_text(json.dumps({'reachable_tabletop_xy': REACH}))
    out = tmp_path / 'plan.json'
    tool = [sys.executable, str(ROOT / 'tools/prepare_object_placement_plan.py'), '--scene', str(scene),
            '--grounding', str(ground), '--count', '60', '--seed', '3', '--output', str(out)]
    r = subprocess.run(tool, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['mode'] == 'broad_table' and json.loads(out.read_text())['episode_count'] == 60
    r = subprocess.run(tool, capture_output=True, text=True, env=ENV)
    assert r.returncode == 2 and 'exists' in r.stderr
