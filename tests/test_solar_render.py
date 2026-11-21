"""CPU checks of the SolarRenderer chained bidirectional route: window plan, stitch schedule, canvas geometry, background
anchor, state hand-over, display fill, bindings, the recipe and the chain's plan-only CLI. No model, GPU or network."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLAR = ROOT / 'src/datafarm/render/solar'
FFMPEG = bool(shutil.which('ffmpeg'))
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason='needs ffmpeg')


def solar(name):
    if str(SOLAR) not in sys.path:
        sys.path.insert(0, str(SOLAR))
    import importlib
    return importlib.import_module(name)


def canvas():
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    return solar('solar_canvas')


def legacy_plan(total, win, k, wins=(), start=0, windows=0):
    """The window loop of the original chain tool, kept verbatim as the reference."""
    out, w, s = [], 0, start
    while True:
        remaining = total - s
        if remaining < k + 2 or (windows and w >= windows):
            break
        n = min(wins[w] if w < len(wins) else win, remaining)
        out.append((s, n))
        if s + n >= total:
            break
        s = s + n - 1 - k
        w += 1
    return out


def test_legal_lengths():
    sc = canvas()
    assert [n for n in range(1, 200) if sc.legal_length(n)] == [22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192]
    for n in range(1, 700):
        m = sc.pad_to_legal(n)
        assert sc.legal_length(m) and m >= n and m - max(n, 22) < 17


@pytest.mark.parametrize('total,win,wins', [(622, 447, ()), (622, 447, (141,)), (622, 447, (209,)), (4348, 532, ()),
                                            (300, 532, ()), (548, 532, ()), (549, 532, ()), (2000, 447, (141, 300))])
def test_plan_matches_reference_and_covers_the_episode(total, win, wins):
    sc = canvas()
    plan = sc.plan_windows(total, win, 16, wins)
    assert plan == legacy_plan(total, win, 16, wins)
    assert plan[0][0] == 0 and plan[-1][0] + plan[-1][1] == total
    for (s0, n0), (s1, _n1) in zip(plan, plan[1:]):
        assert s1 == s0 + n0 - 1 - 16
    assert all(n >= 18 for _, n in plan)


def test_weigh_apple_handover_frame():
    sc = canvas()
    assert sc.plan_windows(622, 447, 16, (141,)) == [(0, 141), (124, 447), (554, 68)]
    assert sc.plan_windows(622, 447, 16, (209,)) == [(0, 209), (192, 430)]


def test_plan_rejects_bad_lengths():
    sc = canvas()
    with pytest.raises(ValueError):
        sc.plan_windows(600, 500, 16)
    with pytest.raises(ValueError):
        sc.plan_windows(600, 447, 16, (17,))


@pytest.mark.parametrize('total,win,wins', [(622, 447, (141,)), (4348, 532, ()), (100, 39, ()), (300, 532, ())])
def test_stitch_schedule_writes_every_frame_once(total, win, wins):
    sc = canvas()
    k = 16
    plan = sc.plan_windows(total, win, k, wins)
    sched = sc.stitch_schedule(plan, k)
    assert [cf for cf, _ in sched] == list(range(total))
    for cf, parts in sched:
        assert abs(sum(w for *_x, w in parts) - 1) < 1e-9
        for wi, local, _w in parts:
            s0, n = plan[wi]
            assert s0 + local == cf and 0 <= local < n
    blends = [parts for _cf, parts in sched if len(parts) == 2]
    assert len(blends) == k * (len(plan) - 1)
    if blends:
        ramp = [parts[1][2] for parts in blends[:k]]
        assert ramp == sorted(ramp) and 0 < ramp[0] < ramp[-1] < 1


def test_canvas_geometry_roundtrip():
    sc = canvas()
    import numpy as np
    t3 = np.full((1080, 1280, 3), 200, np.uint8)
    c = sc.t3_to_canvas(t3)
    assert c.shape == (768, 1344, 3)
    assert c[:, :217].max() == 0 and c[:, 217 + 910:].max() == 0 and c[:, 217:1127].min() == 200
    back = sc.canvas_to_t3(c)
    assert back.shape == (1080, 1280, 3) and abs(int(back.mean()) - 200) <= 1
    z = sc.head_zone()
    assert z[:512, 217:1127].all() and not z[512:].any() and not z[:, :217].any()


def test_anchor_replaces_only_static_head_background():
    sc = canvas()
    import numpy as np
    car0 = np.full((768, 1344, 3), 128, np.uint8)
    car = car0.copy()
    car[100:300, 500:800] = 255                       # the arm is here at the hand-over frame
    frame = np.full_like(car0, 50)
    real = np.full_like(car0, 220)
    counter = sc.MovedCounter(car0)
    for _ in range(10):
        counter.add(car0)
    moving = car0.copy()
    moving[400:500, 300:400] = 0                       # the task moves this area often
    for _ in range(5):
        counter.add(moving)
    no_paste = counter.mask() | ~sc.head_zone()
    out, frac = sc.anchor_blend(frame, car, car0, real, no_paste)
    px = lambda y, x: int(out[y, x].astype(int).mean())  # noqa: E731
    assert abs(px(50, 300) - 220) <= 1                    # static head background -> real frame 0
    assert abs(px(200, 650) - 50) <= 1                    # where the carrier changed: rendered frame kept
    assert abs(px(450, 350) - 50) <= 1                    # area the task moves: kept
    assert abs(px(700, 600) - 50) <= 1                    # wrist views: never anchored
    assert out[:, :217].max() == 0 and 0 < frac < 1


def lit_scale(np, lit=True):
    car = np.full((1080, 1280, 3), 180, np.uint8)
    car[300:330, 600:680] = (255, 255, 255) if lit else (0, 0, 0)    # wide display
    car[800:860, 100:160] = 255                                         # white square: not a display
    car[900:1000, 400:900] = 255                                        # too large
    return car


def test_lit_display_mask_and_handover_paste():
    sc = canvas()
    import numpy as np
    m = sc.lit_display_mask(lit_scale(np))
    ys, xs = np.nonzero(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (300, 329, 600, 679)
    assert not sc.lit_display_mask(lit_scale(np, lit=False)).any()
    frame = np.full((768, 1344, 3), 40, np.uint8)
    src = np.full((1080, 1280, 3), 230, np.uint8)
    out, npx = sc.handover_paste(frame, src, m)
    assert npx > 30 * 80
    cy, cx = int(315 * 768 / 1080), 217 + int(640 * 910 / 1280)
    assert out[cy, cx].min() > 200                                     # display centre from the source render
    assert out[50, 300].tolist() == [40, 40, 40]                       # elsewhere untouched
    assert (out != frame).any(axis=2).sum() < 0.01 * out.shape[0] * out.shape[1]


def test_fill_lit_displays():
    sc = canvas()
    import numpy as np
    out, k = sc.fill_lit_displays(lit_scale(np), (255, 178, 92))
    assert k == 1 and out[315, 640].tolist() == [255, 178, 92] and out[830, 130].tolist() == [255, 255, 255]
    same, k0 = sc.fill_lit_displays(lit_scale(np, lit=False))
    assert k0 == 0


def test_bindings_fail_closed(monkeypatch, tmp_path):
    paths = solar('solar_paths')
    for key in paths.BINDING_KEYS:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv('DATAFARM_' + key, raising=False)
    monkeypatch.setenv('DATAFARM_BINDINGS', str(tmp_path / 'none.json'))
    monkeypatch.setattr(paths, '_bindings_file', lambda: {})
    with pytest.raises(paths.MissingBinding):
        paths.solar_repo()
    with pytest.raises(paths.MissingBinding):
        paths.binding('NOT_A_KEY')
    monkeypatch.setenv('SOLAR_REPO', str(tmp_path))
    with pytest.raises(paths.MissingBinding, match='scripts/render.py'):
        paths.solar_repo()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/render.py').write_text('')
    assert paths.solar_repo() == tmp_path
    monkeypatch.setenv('SOLAR_LORA', str(tmp_path))
    with pytest.raises(paths.MissingBinding, match='solarrenderer_control_lora'):
        paths.lora_pair('bidir')
    for name in paths.LORA_PAIRS['bidir']:
        (tmp_path / f'{name}.safetensors').write_bytes(b'')
    assert [p.name for p in paths.lora_pair('bidir')] == ['solarrenderer_control_lora.safetensors',
                                                          'solarrenderer_4step_lora.safetensors']
    monkeypatch.setenv('SOLAR_H3', '${SOLAR_ROOT}/weights')
    with pytest.raises(paths.MissingBinding, match='unexpanded'):
        paths.h3_weights()


def test_prompt_condition_is_state_neutral_and_generic_is_english():
    prompt = solar('solar_prompt')
    what, kept = prompt.condition('a red rounded apple, and a pale grey kitchen scale.')
    assert '(a red rounded apple, and a pale grey kitchen scale)' in what
    assert 'open or closed and every object is placed exactly as {v} shows' in kept
    free = prompt.generic_free('dual-arm AgileX Piper robot')
    assert set(free) == {'summary', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'}
    text = ' '.join(free.values())
    assert 'dual-arm AgileX Piper robot' in text and '{robot}' not in text and text.isascii()
    for narrated in ('puts ', 'places the', 'then the'):
        assert narrated not in free['detailed_description']


def test_recipe_is_consistent_with_the_code():
    sc = canvas()
    recipe = json.loads((ROOT / 'configs/render/solar-chain-bidir-v1.json').read_text())
    assert recipe['schema'] == 'datafarm.solar-render-recipe/1' and recipe['engine'] == 'solarrenderer'
    chain = recipe['chain']
    assert sc.legal_length(chain['window_frames']) and sc.legal_length(chain['window_frames_141gb'])
    cv = recipe['canvas']
    assert (cv['width'], cv['height'], cv['active_x0'], cv['active_width']) == (sc.CANVAS_W, sc.CANVAS_H, sc.ACTIVE_X0,
                                                                               sc.ACTIVE_W)
    paths = solar('solar_paths')
    assert tuple(recipe['lora_pair']) == paths.LORA_PAIRS['bidir']
    for rel in ('docs/reference/rendering.md', 'src/datafarm/render/solar/README.md', 'vendor/solarrenderer/README.md',
                'vendor/solarrenderer/solarrenderer-attention-env.patch'):
        assert (ROOT / rel).is_file(), rel
    assert 'solar-chain-bidir-v1.json' in (ROOT / 'docs/reference/rendering.md').read_text()


def test_external_assets_pin_solarrenderer():
    assets = json.loads((ROOT / 'configs/external-assets.json').read_text())
    repo = [r for r in assets['repositories'] if r.get('binding') == 'SOLAR_REPO']
    assert len(repo) == 1 and len(repo[0]['upstream_commit']) == 40
    files = [f for f in assets['files'] if 'solar' in f['profiles']]
    assert {Path(f['path']).name for f in files} >= {'solarrenderer_control_lora.safetensors',
                                                     'solarrenderer_4step_lora.safetensors'}
    for f in files:
        assert len(f['sha256']) == 64 and f['bytes'] > 0 and f['path'].startswith('${SOLAR_')


def write_carrier(path, n, lit_from=None):
    np = pytest.importorskip('numpy')
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x1080',
                          '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '4', '-pix_fmt',
                          'yuv444p', str(path)], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(lit_scale(np, lit=lit_from is not None and i >= lit_from).tobytes())
    p.stdin.close()
    assert p.wait() == 0


@needs_ffmpeg
def test_chain_plan_only_and_handover_check(tmp_path):
    canvas()
    car = tmp_path / 'carrier.mp4'
    write_carrier(car, 140)
    tool = str(SOLAR / 'solar_chain.py')
    r = subprocess.run([sys.executable, tool, '--carrier', str(car), '--plan-only', '--win', '73', '--wins', '56'],
                       capture_output=True, text=True, check=True)
    plan = json.loads(r.stdout)
    assert plan['carrier_frames'] == 140 and plan['windows'] == [[0, 56], [39, 73], [95, 45]]
    assert plan['handover_frames'] == [39, 95]
    r = subprocess.run([sys.executable, tool, '--carrier', str(car), '--win', '73', '--wins', '56', '--handover',
                        '40:x.png', '--first', 'f', '--real', 'r', '--prompt', 'p', '--root', str(tmp_path / 'o')],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'not hand-over frames' in r.stderr


@needs_ffmpeg
def test_clip_pads_to_legal_length(tmp_path):
    canvas()
    car = tmp_path / 'carrier.mp4'
    write_carrier(car, 50)
    clip = solar('solar_clip')
    meta = clip.write_clip(str(car), 10, 30, tmp_path / 'clip', pad17=True)
    assert meta['frames'] == 30 and meta['padded_frames'] == 9
    import cv2
    cap = cv2.VideoCapture(str(tmp_path / 'clip/structure.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 39
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1344, 768)


@needs_ffmpeg
def test_carrier_display_fill_reports_first_lit_frame(tmp_path):
    canvas()
    car = tmp_path / 'carrier.mp4'
    write_carrier(car, 12, lit_from=5)
    r = subprocess.run([sys.executable, str(SOLAR / 'solar_carrier_displays.py'), str(car), str(tmp_path / 'out.mp4')],
                       capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    assert info['frames'] == 12 and info['first_lit_frame'] == 5 and info['frames_with_lit_display'] == 7
