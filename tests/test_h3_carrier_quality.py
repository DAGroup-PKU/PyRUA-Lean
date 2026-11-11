"""CPU checks of the H3 carrier and reference preparation (white replay, carrier, episode references, keep regions,
wide-wrist crop back, tail padding).  No simulator, GPU, model or network: the SAPIEN replay itself is not run; its
spec, entry resolution, identity check and mesh generators are; media steps use synthetic videos and ffmpeg."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / 'src/datafarm/render/h3'
FFMPEG = bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason='needs ffmpeg/ffprobe')


def hybrid(name):
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    if str(HYBRID) not in sys.path:
        sys.path.insert(0, str(HYBRID))
    import importlib
    return importlib.import_module(name)


def write_video(path, frames, fps=24):
    """Encode BGR uint8 frames (list of HxWx3) losslessly enough for colour checks."""
    import numpy as np
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                          '-crf', '4', '-pix_fmt', 'yuv444p', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


# ------------------------------------------------------------------ white replay (no simulator)
from datafarm.simulation import white_replay as wr  # noqa: E402


def good_spec():
    return {'schema': wr.SPEC_SCHEMA, 'wrist_fov_factor': 0.65, 'image_stride': 2,
            'colors': {'@robot': [0.1, 0.1, 0.1, 1], '@robot_links': {'*gripper*': [0.05, 0.05, 0.05]},
                       'drum': {'body': [0.8, 0.1, 0.1, 1], 'top': [0.9, 0.85, 0.7, 1]},
                       '@states': {'scale': [{'when': [0.2, 0.2, 0.2], 'to': [1, 1, 1, 1]}]}},
            'shapes': {'apple': [{'kind': 'apple', 'radii': [0.04, 0.04, 0.035], 'flat_z': -0.03,
                                  'color': [0.7, 0.1, 0.1]}, {'kind': 'stem', 'top_z': 0.03, 'color': [0.4, 0.3, 0.1]}],
                       'proxy': [{'kind': 'hide'}]},
            'extras': [{'name': 'wall', 'kind': 'box', 'pose': [0, 1, 0.5, 1, 0, 0, 0], 'half_size': [1, 0.01, 0.5],
                        'color': [0.9, 0.9, 0.88], 'emission': [0.35, 0.35, 0.35]}]}


def test_replay_spec_accepts_every_documented_change_and_rejects_malformed_ones(tmp_path):
    assert wr.validate_spec(good_spec()) == []
    bad = good_spec()
    bad['wrist_fov_factor'] = 1.5
    bad['colors']['@robot'] = [2, 0, 0]
    bad['colors']['drum'] = {'body': [1, 0, 0]}
    bad['shapes']['apple'][0].pop('radii')
    bad['shapes']['proxy'] = [{'kind': 'sphere'}]
    bad['extras'].append({'name': 'wall', 'kind': 'box', 'pose': [0, 0, 0], 'half_size': [1, 1], 'color': [1, 1, 1]})
    bad['typo'] = 1
    problems = '\n'.join(wr.validate_spec(bad))
    for needle in ('wrist_fov_factor', "colors[@robot]", 'exactly body and top', 'needs radii', 'kind must be one of',
                   'pose must be', 'half_size', 'duplicate name', "unknown key 'typo'"):
        assert needle in problems, needle
    p = tmp_path / 'spec.json'
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match='invalid white-replay spec'):
        wr.load_spec(p)
    assert wr.main(['--check-spec', str(p)]) == 1


def test_wrist_fov_scaling_keeps_the_principal_point():
    assert wr.scale_k([[600, 0, 640], [0, 600, 360], [0, 0, 1]], 0.65) == [[390, 0, 640], [0, 390, 360], [0, 0, 1]]


def test_entry_resolves_only_the_frozen_run_seed(tmp_path):
    work = tmp_path / 'W'
    att = work / 'capture/attempts/a01_s7'
    att.mkdir(parents=True)
    script = work / 'task/expert_v2/run_seed.py'
    script.parent.mkdir(parents=True)
    script.write_text('IMAGE_STRIDE = 5\n')
    (att / 'attempt.json').write_text(json.dumps({'seed': 7}))
    (att / 'expert.json').write_text(json.dumps({'source_sha256': {'run_seed.py': wr.sha256(script)}}))
    assert wr.resolve_entry(att) == (script, 7, 'recorded seed + frozen source')
    script.write_text('IMAGE_STRIDE = 6\n')   # edited after capture: no longer the frozen entry
    with pytest.raises(SystemExit, match='no frozen run_seed.py'):
        wr.resolve_entry(att)
    (att / 'launch.json').write_text(json.dumps({'argv': ['python', '/x/run_seed.py', '7', str(att), '--image']}))
    assert wr.resolve_entry(att) == (Path('/x/run_seed.py'), 7, 'launch.json')


def test_entry_resolves_a_receipt_argv_and_a_source_snapshot(tmp_path):
    work = tmp_path / 'W'
    att = work / 'capture/attempts/a01_s9'
    att.mkdir(parents=True)
    script = work / 'task/expert/run_seed.py'
    script.parent.mkdir(parents=True)
    script.write_text('def main():\n    pass\n')
    (att / 'seed.json').write_text(json.dumps({'seed': 9}))
    (att / 'source-snapshot.json').write_text(json.dumps({'expert/run_seed.py': script.read_text()}))
    assert wr.resolve_entry(att) == (script, 9, 'recorded seed + frozen source')
    (att / 'receipt.json').write_text(json.dumps({'argv': [str(script), '9', str(att)]}))
    assert wr.resolve_entry(att) == (script, 9, 'receipt.json')
    script.write_text('def main():\n    return 1\n')                  # edited after the attempt ran
    with pytest.raises(SystemExit, match='changed since attempt'):
        wr.resolve_entry(att)


def test_identity_check_ignores_contact_order_and_catches_any_state_drift(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    rows = [{'t': i, 'q': [0.1 * i, 0.2], 'contacts': [[1, 2], [3, 4]]} for i in range(5)]
    (a / 'states.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    shuffled = [dict(r, contacts=list(reversed(r['contacts']))) for r in rows]
    (b / 'states.jsonl').write_text('\n'.join(json.dumps(r) for r in shuffled))
    rep = wr.compare_logs(a, b)
    assert rep['state_log_identical'] and not rep['byte_identical']
    shuffled[3]['q'][0] += 1e-9
    (b / 'states.jsonl').write_text('\n'.join(json.dumps(r) for r in shuffled))
    rep = wr.compare_logs(a, b)
    assert not rep['state_log_identical'] and rep['first_state_diff_line'] == 3
    np = pytest.importorskip('numpy')
    (a / 'states.jsonl').unlink()
    np.savez(a / 'log.npz', q=np.arange(6.0))
    np.savez(b / 'log.npz', q=np.arange(6.0))
    assert wr.compare_logs(a, b)['state_log_identical']
    np.savez(b / 'log.npz', q=np.arange(6.0) + 1e-12)
    assert wr.compare_logs(a, b)['differing_arrays'] == ['q']


def cams(f_left=0.65, f_right=0.65):
    K = [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]
    return {'cam_high': {'K_original': K, 'K_used': K},
            'cam_left_wrist': {'K_original': K, 'K_used': wr.scale_k(K, f_left)},
            'cam_right_wrist': {'K_original': K, 'K_used': wr.scale_k(K, f_right)}}


def test_replay_is_ok_only_when_every_requested_change_was_applied():
    spec = good_spec()
    rep = {'state_log_identical': True, 'camera_intrinsics': cams(),
           'recoloured_entities': {'link1': 3}, 'image_stride': [48, 2], 'shaped': {'apple': {}, 'proxy': {}},
           'extras_added': {'wall': 'box'}}
    assert wr.replay_ok(rep, spec)
    for key, value in (('state_log_identical', False), ('camera_intrinsics', {}), ('recoloured_entities', {}),
                       ('image_stride', None), ('shaped', {'apple': {}}), ('extras_added', {}),
                       ('camera_intrinsics', cams(f_right=1.0)),        # one wrist camera rendered unscaled
                       ('camera_intrinsics', {'cam_high': cams()['cam_high']}),   # no wrist camera at all
                       ('recolour_leaks', ['table'])):   # a robot colour that also reached the table
        assert not wr.replay_ok({**rep, key: value}, spec), key
    assert wr.replay_ok({**rep, 'camera_intrinsics': {}}, {**spec, 'wrist_fov_factor': 1.0})


def test_recolouring_copies_a_shared_material_with_all_its_values():
    import types

    class Mat:
        def __init__(self):
            self.base_color, self.emission, self.roughness, self.metallic = [1, 1, 1, 1], [0, 0, 0, 0], 1.0, 0.0
            self.specular, self.transmission, self.ior, self.transmission_roughness = 0.0, 0.0, 1.45, 0.0
            self.base_color_texture = self.emission_texture = None
            self.metallic_texture = self.normal_texture = self.roughness_texture = self.transmission_texture = None
    fake = types.SimpleNamespace(render=types.SimpleNamespace(RenderMaterial=Mat))
    white = Mat()
    white.base_color, white.roughness, white.normal_texture = [.85, .85, .85, 1], .8, 'tex'
    a, b = wr.copy_material(fake, white), wr.copy_material(fake, white)
    assert a is not white and b is not a
    assert (a.base_color, a.roughness, a.normal_texture, a.ior) == ([.85, .85, .85, 1], .8, 'tex', 1.45)
    a.base_color = [.08, .09, .08, 1]                   # the robot's copy; the table keeps white
    assert white.base_color == [.85, .85, .85, 1] and b.base_color == [.85, .85, .85, 1]


def test_render_mesh_generators_build_valid_closed_meshes():
    np = pytest.importorskip('numpy')
    for V, F in (wr.revolve_x(np.linspace(-0.1, 0.1, 9), np.full(9, 0.05)),
                 wr.apple_mesh([0.04, 0.04, 0.035], -0.03),
                 wr.revolve_z([(0.0, 0.0), (0.04, 0.0), (0.045, 0.09), (0.0, 0.09)]),
                 wr.handle_mesh(90, 0.04, 0.02, 0.07, 0.03, 0.005)):
        assert V.ndim == 2 and V.shape[1] == 3 and F.shape[1] == 3
        assert F.min() >= 0 and F.max() < len(V) and np.isfinite(V).all()
    V, _ = wr.apple_mesh([0.04, 0.04, 0.035], -0.03)
    assert V[:, 2].min() >= -0.03 - 1e-12                 # flat foot


# ------------------------------------------------------------------ carrier
def room(wall, table, robot=False):
    import numpy as np
    f = np.full((720, 1280, 3), wall, np.uint8)
    f[300:] = table
    if robot:
        f[100:500, 200:300] = 30
    return f


def test_room_seam_check_flags_a_flat_white_scene_and_passes_a_shaded_one():
    h3_carrier = hybrid('h3_carrier')
    assert h3_carrier.room_seams(room(200, 204, robot=True)) == []         # flat whites: no seam above tau 25
    seams = h3_carrier.room_seams(room(186, 237, robot=True))              # shaded white: wall/table seam kept
    assert seams and seams[0]['length'] > 0.5 and seams[0]['tilt_deg'] < 5 and seams[0]['step'] > 25


@needs_ffmpeg
def test_carrier_compose_refuses_a_low_rate_white_and_keeps_every_frame(tmp_path):
    h3_carrier = hybrid('h3_carrier')
    np = pytest.importorskip('numpy')
    views = {}
    for side, size in (('head', (720, 1280)), ('left', (360, 640)), ('right', (360, 640))):
        views[side] = write_video(tmp_path / f'{side}-5.mp4', [np.full((*size, 3), 40 * i, np.uint8) for i in range(6)], 5)
    with pytest.raises(SystemExit, match='re-render it densely'):
        h3_carrier.compose(views, tmp_path / 'low.mp4')
    frames = [room(186, 237)] * 30
    views = {'head': write_video(tmp_path / 'h.mp4', frames),
             'left': write_video(tmp_path / 'l.mp4', [f[::2, ::2] for f in frames]),
             'right': write_video(tmp_path / 'r.mp4', [f[::2, ::2] for f in frames])}
    meta = h3_carrier.compose(views, tmp_path / 't3.mp4')
    info = h3_carrier.probe(tmp_path / 't3.mp4')
    assert (info.width, info.height, info.frames, meta['fps']) == (1280, 1080, 30, 24)
    rep = h3_carrier.check(tmp_path / 't3.mp4')
    assert rep['frames_with_room_seam'] == rep['sampled_frames'] and rep['warnings'] == []
    flat = [room(200, 204)] * 30
    flat_views = {'head': write_video(tmp_path / 'fh.mp4', flat),
                  'left': write_video(tmp_path / 'fl.mp4', [f[::2, ::2] for f in flat]),
                  'right': write_video(tmp_path / 'fr.mp4', [f[::2, ::2] for f in flat])}
    h3_carrier.compose(flat_views, tmp_path / 'flat.mp4')
    assert any('invent a wall' in w for w in h3_carrier.check(tmp_path / 'flat.mp4')['warnings'])


@needs_ffmpeg
def test_replay_png_frames_become_a_view_at_the_replay_image_rate(tmp_path):
    h3_carrier = hybrid('h3_carrier')
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    att, rep = tmp_path / 'att', tmp_path / 'rep'
    att.mkdir(); rep.mkdir()
    for cam, size in (('cam_high', (720, 1280)), ('cam_left_wrist', (360, 640)), ('cam_right_wrist', (360, 640))):
        lines = []
        for i in range(12):
            cv2.imwrite(str(rep / f'{cam}-{i:03d}.png'), np.full((*size, 3), 10 * i, np.uint8))
            lines.append(f"file '{att}/{cam}-{i:03d}.png'")
        name = f'{cam}-frames.txt' if cam == 'cam_high' else f'{cam}.concat'   # both list names
        (att / name).write_text('\n'.join(lines) + '\n')
    (rep / 'image-manifest.json').write_text(json.dumps({'physics_hz': 240, 'stride': 10}))
    views = h3_carrier.replay_views(rep, att)
    for v in views.values():
        assert h3_carrier.probe(v).frames == 12 and h3_carrier.fps_of(v) == 24


# ------------------------------------------------------------------ episode references
@pytest.fixture
def real_episode(tmp_path):
    """Synthetic real episode: shaded room, two dark arms that sweep in, a red and a blue object; wrists see the red one."""
    if not FFMPEG:
        pytest.skip('needs ffmpeg/ffprobe')
    np = pytest.importorskip('numpy')
    head, left, right = [], [], []
    for i in range(60):
        f = room(186, 230)
        x = 100 + 4 * i
        f[150:520, x:x + 70] = 25                              # left arm
        f[150:520, 1180 - x - 70:1180 - x] = 28                # right arm
        f[420:520, 520:600] = (40, 40, 200)                    # red object
        f[430:510, 700:770] = (200, 60, 40)                    # blue object
        head.append(f)
        w = np.full((360, 640, 3), 220, np.uint8)
        w[300:, 200:440] = 20                                  # gripper at the bottom
        if 20 <= i <= 30:
            w[80:220, 250:390] = (40, 40, 200)
        left.append(w)
        right.append(w.copy())
    real = {'head': write_video(tmp_path / 'cam_high.mp4', head), 'left': write_video(tmp_path / 'cam_left.mp4', left),
            'right': write_video(tmp_path / 'cam_right.mp4', right)}
    return {k: str(v) for k, v in real.items()}


def test_episode_reference_set_respects_the_memory_budget(tmp_path, real_episode):
    refs = hybrid('h3_episode_references')
    cv2 = pytest.importorskip('cv2')
    spec = {'schema': refs.SPEC_SCHEMA, 'out_dir': str(tmp_path / 'refs'), 'real': real_episode}
    sp = tmp_path / 'spec.json'
    sp.write_text(json.dumps(spec))
    rec = refs.build(refs.load_spec(sp))
    names = [Path(r['path']).name for r in rec['references']]
    assert names[0] == 'real-3view-f0.png'
    assert {'robot-sheet.png', 'object-sheet.png', 'wrist-sheet.png'} <= set(names) | set(map(lambda p: Path(p).name,
                                                                                               rec['dropped']))
    assert rec['mpix'] <= 25.0 + 1e-6
    for r in rec['references']:   # bundle REFERENCE fields only, real provenance
        assert set(r) == {'path', 'provenance', 'scope', 'views', 'view_regions', 'coverage'}
        assert r['provenance']['kind'] == 'real-episode-frame'
    wrist = cv2.imread(str(tmp_path / 'refs/wrist-sheet.png'))
    assert wrist.shape[:2] == (720, 640)
    sheet = cv2.imread(str(tmp_path / 'refs/robot-sheet.png'))
    assert (abs(sheet.astype(int) - 128).max(axis=2) > 6).mean() > 0.02     # real robot pixels kept
    col = refs.robot_color(tmp_path / 'refs/robot-sheet.png')
    assert col[3] == 1.0 and max(col[:3]) < 0.2                             # dark links
    assert json.loads(json.dumps(refs.robot_colors(spec)))['@robot'] == col
    # a tighter budget drops full-view frames after f0 first, then sheets from the end
    kept, dropped = refs.budget([{'path': str(tmp_path / 'refs/real-3view-f0.png')}],
                                [{'path': str(tmp_path / 'refs/robot-sheet.png')},
                                 {'path': str(tmp_path / 'refs/wrist-sheet.png')}], limit=13.0)
    assert [Path(r['path']).name for r in kept] == ['real-3view-f0.png', 'robot-sheet.png']
    assert [Path(r['path']).name for r in dropped] == ['wrist-sheet.png']


def test_reference_megapixels_follow_the_2048_short_edge_rescale(tmp_path):
    refs = hybrid('h3_episode_references')
    from PIL import Image
    for (w, h), mp in (((1280, 1080), 4.97), ((1280, 720), 7.46), ((1280, 360), 14.91), ((640, 720), 4.72)):
        Image.new('RGB', (w, h)).save(tmp_path / f'{w}x{h}.png')
        assert abs(refs.ref_mpix(tmp_path / f'{w}x{h}.png') - mp) < 0.01, (w, h)


def test_projected_masks_select_the_robot_and_give_per_link_colours(tmp_path, real_episode):
    refs = hybrid('h3_episode_references')
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    root = tmp_path / 'masks'
    (root / 'cam_high').mkdir(parents=True)
    for i in range(0, 60, 5):
        lab = np.zeros((720, 1280), np.uint8)
        x = 100 + 4 * i
        lab[150:520, x:x + 70] = 1
        lab[150:520, 1180 - x - 70:1180 - x] = 2
        cv2.imwrite(str(root / 'cam_high' / f'{i:06d}.png'), lab)
    (root / 'links.json').write_text(json.dumps({'links': ['left_link', 'right_link']}))
    with pytest.raises(SystemExit, match='incomplete'):
        refs.RobotMasks(root)
    (root / 'done.json').write_text('{}')
    masks = refs.RobotMasks(root)
    assert masks.frames() == list(range(0, 60, 5))
    colors = refs.robot_link_colors(real_episode['head'], masks)
    assert set(colors) == {'left_link', 'right_link'} and all(max(c[:3]) < 0.2 for c in colors.values())
    frame = refs.read_frame(real_episode['head'], 10)
    px = masks.pixels(frame, 10)
    assert px[300, 145] and not px[470, 560]                 # arm in, red object out


def test_robot_mask_projection_draws_nearer_links_over_farther_ones():
    masks_mod = hybrid('h3_robot_masks')
    np = pytest.importorskip('numpy')
    square = (np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]], float),
              np.array([[0, 1, 2], [0, 2, 3]], np.int32))
    meshes = {'far': [square], 'near': [square]}
    poses = {'far': np.eye(4), 'near': np.eye(4)}
    poses['far'][:3, 3] = [0, 0, 1.0]
    poses['near'][:3, 3] = [0.06, 0, 0.6]          # overlaps the far square's right part
    K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    lab = masks_mod.project_labels(['far', 'near'], meshes, poses, np.eye(4), K, np.zeros(5), 640, 480)
    assert lab[240, 335] == 2 and lab[240, 300] == 1 and lab[0, 0] == 0   # nearer link wins the overlap
    T = masks_mod.origin_matrix(type('E', (), {'get': lambda self, k: {'xyz': '1 2 3', 'rpy': '0 0 1.5707963'}[k]})())
    assert np.allclose(T[:3, 3], [1, 2, 3]) and np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-6)


# ------------------------------------------------------------------ keep regions
@needs_ffmpeg
def test_keep_region_keeps_real_pixels_exactly_where_the_white_shows_the_state(tmp_path):
    keep = hybrid('h3_keep_regions')
    h3_media = hybrid('h3_media')
    np = pytest.importorskip('numpy')
    real_frames = []
    for i in range(10):
        f = np.full((720, 1280, 3), 120, np.uint8)
        f[300:330, 600:680] = (230, 160, 60) if i >= 5 else (40, 40, 40)   # a backlit blue display, lit from frame 5
        real_frames.append(f)
    real = write_video(tmp_path / 'real.mp4', real_frames)
    white, inpaint = [], []
    for i in range(8):
        w = np.full((768, 1024, 3), 180, np.uint8)
        if i >= 3:
            w[400:420, 500:550] = 252                                          # replayed state: a bright rectangle
        white.append(w)
    wv = write_video(tmp_path / 'white.mp4', white)
    mv = write_video(tmp_path / 'mask.mp4', [np.full((768, 1024, 3), 255, np.uint8)] * 8)
    spec = {'schema': keep.SPEC_SCHEMA, 'states': {'scale_display': [{'when': [0.1, 0.1, 0.1], 'to': [1, 1, 1, 1]}]},
            'regions': [{'name': 'display', 'white': {'area': [200, 4000]},
                         'texture': {'source': str(real), 'source_sha256': h3_media.sha256(real), 'frame': 7,
                                     'box': [590, 290, 690, 340], 'rule': {'b_minus_r': 30, 'b_min': 150}}}]}
    sp = tmp_path / 'keep.json'
    sp.write_text(json.dumps(spec))
    job = {'chunk_id': 'c00', 'white': str(wv), 'white_sha256': h3_media.sha256(wv), 'generated_frames': 8,
           'inpaint_video': str(wv), 'inpaint_video_sha256': h3_media.sha256(wv), 'inpaint_mask': str(mv)}
    rec = keep.apply([job], sp, tmp_path / 'cache')
    assert rec['frames'] == {'display': 5}
    assert job['keep_regions']['frames'] == {'display': 5} and job['inpaint_video'] != str(wv)
    inp, mask = h3_media.read_all(job['inpaint_video']), h3_media.read_all(job['inpaint_mask'])
    assert mask[5, 410, 525].max() < 30 and mask[1, 410, 525].min() > 200 and mask[5, 100, 100].min() > 200
    b, g, r = (int(x) for x in inp[5, 410, 525])
    assert b - r > 60                                                          # real blue display pixels kept
    assert 0 < job['inpaint_regenerate_fraction'] < 1
    again = dict(job, inpaint_video=str(wv), inpaint_mask=str(mv))
    keep.apply([again], sp, tmp_path / 'cache')                               # cached: same patched inputs
    assert again['inpaint_video'] == job['inpaint_video']
    spec['regions'][0]['texture']['source_sha256'] = '0' * 64
    (tmp_path / 'cache').rename(tmp_path / 'cache-old')
    sp.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match='sha256 mismatch'):
        keep.apply([dict(job, inpaint_video=str(wv), inpaint_mask=str(mv))], sp, tmp_path / 'cache')
    assert any('needs box' in p for p in keep.validate({'schema': keep.SPEC_SCHEMA, 'regions': [
        {'name': 'd', 'texture': {'source': 'x', 'source_sha256': 'y', 'frame': 1}}]}))


# ------------------------------------------------------------------ wide-wrist crop back
def test_wrist_crop_window_is_the_original_camera_view_inside_the_wide_one():
    crop = hybrid('h3_wrist_crop')
    x0, y0, w, h = crop.crop_window(0.65, (640, 360))
    assert (round(x0, 3), round(y0, 3), round(w, 3), round(h, 3)) == (112.0, 63.0, 416.0, 234.0)
    x0, y0, w, h = crop.crop_window(0.65, (1270, 700))           # off-centre principal point: clamped in the panel
    assert x0 + w <= 640 + 1e-9 and y0 + h <= 360 + 1e-9 and x0 > 112
    with pytest.raises(ValueError):
        crop.crop_window(1.0, (640, 360))
    fg = crop.filter_graph(0.65, {'left': (640, 360), 'right': (600, 380)})
    assert fg.count('scale=640:360') == 2 and fg.endswith('vstack=inputs=2[o]')


# ------------------------------------------------------------------ tail padding through assembly and the exposure lock
@needs_ffmpeg
def test_tail_padded_package_is_cut_back_to_the_episode_length(tmp_path, monkeypatch):
    """Build (HED stubbed) with the default tail_pad, fake the worker by copying each white clip as the chunk render,
    then assemble and exposure-lock: both outputs have exactly the episode's frames."""
    hcp = hybrid('h3_package')
    asm = hybrid('h3_assemble')
    lock = hybrid('h3_exposure_lock')
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_h3_control_hybrid import make_carrier, recipe_config, synthetic_bundle
    run = tmp_path / 'run'
    for d in ('run/recipes', 'run/tasks/unseen-task', 'models', 'vx'):
        (tmp_path / d).mkdir(parents=True)
    carrier = tmp_path / 'carrier.mp4'
    make_carrier(carrier, 200)
    bundle = synthetic_bundle(tmp_path, carrier)
    bundle['episodes'][0]['wide_wrist'] = {'factor': 0.65, 'principal_points': {'left': [640, 360], 'right': [640, 360]}}
    (run / 'tasks/unseen-task/bundle.json').write_text(json.dumps(bundle))
    recipe = run / 'recipes/test-hybrid.json'
    recipe.write_text(json.dumps({**recipe_config(), 'name': 'test-hybrid'}))
    monkeypatch.setenv('H3_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('H3_VIDEOX_FUN', str(tmp_path / 'vx'))
    monkeypatch.setenv('H3_RUN', str(run))
    monkeypatch.setattr(hcp.h3_paths, '_RUN_DIR', None)

    def fake_hed(job, out_dir, ctl, threads, device):
        target = out_dir / f"{job['chunk_id']}-hed-soft.mp4"
        shutil.copy(job['white'], target)
        return {'path': str(target), 'sha256': hcp.sha256(target), 'min_edge_intensity': 0,
                'mean_active_fraction_gt20': 0.0, 'mean_normalized_intensity': 0.0, 'device': 'stub',
                'tool_sha256': 'stub'}
    monkeypatch.setattr(hcp, 'make_hed', fake_hed)
    out = run / 'pkgs/test-hybrid/unseen-task/ep0-sim2real'
    args = type('A', (), dict(bundle=str(run / 'tasks/unseen-task/bundle.json'), recipe=str(recipe), episode='ep0',
                              mode='sim2real', variant=None, seed_offset=0, out=str(out), force=False, hed_jobs=1,
                              hed_threads=1, hed_device='cpu', run_dir=None))()
    assert hcp.cmd_build(args) == 0
    pkg = json.loads((out / 'package.json').read_text())
    assert (pkg['frames'], pkg['tail_pad'], pkg['render_frames']) == (200, 48, 248)
    assert pkg['wide_wrist']['factor'] == 0.65
    manifest = json.loads((out / 'job-manifest.json').read_text())
    assert manifest['jobs'][-1]['keep_global_range'][1] == 248
    for job in manifest['jobs']:          # fake worker: the white clip is the chunk "render"
        d = out / 'render/chunks' / job['chunk_id']
        d.mkdir(parents=True)
        shutil.copy(job['white'], d / 'chunk-render.mp4')
        (d / 'segment-manifest.json').write_text('{}')
    record = asm.assemble(out)
    assert record['frames'] == 200 and hcp.probe(record['final']).frames == 200
    assert hcp.probe(record['carrier_clip']).frames == 200
    locked = lock.lock_package(out, {'name': 'exposure-lock-test'}, run)
    assert json.loads((locked / 'assembled.json').read_text())['frames'] == 200
    assert hcp.probe(locked / 'final-t3-24fps.mp4').frames == 200
    crop = hybrid('h3_wrist_crop')
    cropped = crop.crop_back(locked, run)
    assert cropped.parts[-3] == 'exposure-lock-test-wristcrop'
    info = hcp.probe(cropped / 'final-t3-24fps.mp4')
    assert (info.width, info.height, info.frames) == (1280, 1080, 200)
    assert json.loads((cropped / 'package.json').read_text())['wrist_crop_back']['factor'] == 0.65
    assert (cropped / 'render/chunks').is_dir()


@needs_ffmpeg
def test_bundle_check_fails_on_a_low_rate_carrier_and_malformed_new_fields(tmp_path, monkeypatch, capsys):
    h3_bundle = hybrid('h3_bundle')
    np = pytest.importorskip('numpy')
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_h3_control_hybrid import synthetic_bundle
    low = write_video(tmp_path / 'low.mp4', [np.full((1080, 1280, 3), 200, np.uint8)] * 8, fps=5)
    bundle = synthetic_bundle(tmp_path, low)
    bundle['episodes'][0]['wide_wrist'] = {'factor': 1.2, 'principal_points': {'left': [1, 2]}}
    bundle['episodes'][0]['keep_regions'] = str(tmp_path / 'missing-keep.json')
    bp = tmp_path / 'bundle.json'
    bp.write_text(json.dumps(bundle))
    (tmp_path / 'run').mkdir()
    monkeypatch.setenv('H3_RUN', str(tmp_path / 'run'))
    monkeypatch.setattr(h3_bundle.h3_paths, '_RUN_DIR', None)
    rc = h3_bundle.cmd_check(type('A', (), {'bundle': str(bp)})())
    report = json.loads(capsys.readouterr().out)
    problems = '\n'.join(report['problems'])
    assert rc == 1
    for needle in ('re-render the white densely', 'wide_wrist.factor', 'principal_points', 'keep_regions spec missing'):
        assert needle in problems, needle
