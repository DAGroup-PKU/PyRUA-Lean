"""CPU-only checks of the vendored H3 control-hybrid backend (src/datafarm/render/h3).

No GPU, model, provider or network: modules compile/import, CLIs answer --help without torch or bindings,
deployment paths fail closed, the default recipes are valid for the vendored builder, the builder's CPU part
produces a manifest that passes the render-policy gate, and the evaluator prompts carry no machine paths.
"""
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from render_policy_fixtures import policy_fixture

MACHINE_PATH = re.compile(r'/home/|/mnt/|ssh -o ')
ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / 'src/datafarm/render/h3'
EVAL = ROOT / 'evaluations/render-h3'
CONFIGS = ROOT / 'configs/render'
PY_FILES = sorted(p.relative_to(HYBRID).as_posix() for p in HYBRID.rglob('*.py'))
LIBRARY_ONLY = ('h3_spatial.py', 'h3_lora.py', 'h3_media.py')
CLIS = [f for f in PY_FILES if f not in LIBRARY_ONLY]
SHELL = sorted(p.name for p in HYBRID.glob('*.sh'))


def load_paths():
    spec = importlib.util.spec_from_file_location('h3_paths_under_test', HYBRID / 'h3_paths.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATHS = load_paths()
ENV_KEYS = ('H3_RUN', *PATHS.BINDING_KEYS)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ENV_KEYS and not k.startswith('DATAFARM_')}
    env['DATAFARM_BINDINGS'] = str(ROOT / 'tests' / 'no-such-bindings.json')  # ignore a deployment bindings.json
    env.update(extra)
    return env


@pytest.fixture
def no_bindings(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv('DATAFARM_' + key, raising=False)
    monkeypatch.setenv('DATAFARM_BINDINGS', str(ROOT / 'tests' / 'no-such-bindings.json'))
    monkeypatch.delenv('DATAFARM_WORKSPACE', raising=False)
    monkeypatch.setattr(PATHS, '_RUN_DIR', None)
    return monkeypatch


def test_every_vendored_python_file_byte_compiles(tmp_path):
    assert {'h3_control_worker.py', 'h3_package.py', 'h3_paths.py',
            'h3_sam2_table_masks.py'} <= set(PY_FILES)
    for rel in PY_FILES:
        py_compile.compile(str(HYBRID / rel), cfile=str(tmp_path / (rel.replace('/', '_') + 'c')), doraise=True)


def test_no_machine_paths_or_implicit_run_dir_in_render_code():
    for rel in PY_FILES + SHELL:
        text = (HYBRID / rel).read_text()
        assert not MACHINE_PATH.search(text), rel
        assert not re.search(r'^RUN\s*=\s*Path\(', text, re.M), rel


def test_light_modules_import_without_bindings():
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    modules = [Path(f).stem for f in PY_FILES
               if f not in {'h3_control_worker.py', 'h3_lora.py'}]
    code = ('import importlib, sys; sys.path.insert(0, %r)\n'
            'for m in %r: importlib.import_module(m)\nprint("ok", len(%r))') % (str(HYBRID), modules, modules)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=clean_env(), cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.startswith('ok')


@pytest.mark.parametrize('rel', CLIS)
def test_cli_help_needs_no_gpu_torch_or_bindings(rel):
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    r = subprocess.run([sys.executable, str(HYBRID / rel), '--help'], capture_output=True, text=True,
                       env=clean_env(), cwd=ROOT, timeout=120)
    assert r.returncode == 0, (rel, r.stderr[-2000:])
    assert ('usage' in r.stdout.lower()) or len(r.stdout.strip()) > 20, rel


def test_worker_parses_arguments_before_runtime_imports():
    text = (HYBRID / 'h3_control_worker.py').read_text()
    assert text.index('ARGS = build_parser().parse_args()') < text.index('import torch')
    assert text.index('ARGS = build_parser().parse_args()') < text.index('VIDEOX_FUN_REPO = ')
    for flag in ('--queue', '--profile', '--turbo-lora', '--base-model', '--control-checkpoint',
                 '--control-config', '--memory-mode', '--idle-exit', '--max-tickets'):
        assert f'"{flag}"' in text, flag


@pytest.mark.parametrize('name', SHELL)
def test_shell_scripts_parse_and_fail_closed_without_run_dir(name, tmp_path):
    assert subprocess.run(['bash', '-n', str(HYBRID / name)]).returncode == 0
    args = {'launch_control_worker.sh': ['full']}[name]
    r = subprocess.run(['bash', str(HYBRID / name), *args], capture_output=True, text=True,
                       env=clean_env(), cwd=tmp_path, timeout=60)
    assert r.returncode != 0, name
    assert 'H3_RUN' in r.stderr or 'run directory' in r.stderr, (name, r.stderr)


def test_scheduler_launch_fails_closed_without_scheduler_binding(tmp_path):
    run = tmp_path / 'run'; (run / 'queue').mkdir(parents=True)
    r = subprocess.run(['bash', str(HYBRID / 'launch_control_worker.sh'), 'full'], capture_output=True,
                       text=True, env=clean_env(H3_RUN=str(run)), cwd=tmp_path, timeout=60)
    assert r.returncode != 0 and 'GPU_SCHEDULER' in r.stderr
    sched = tmp_path / 'sched'; sched.write_text('#!/bin/sh\necho CALLED >&2\n'); sched.chmod(0o755)
    env = clean_env(H3_RUN=str(run), GPU_SCHEDULER=str(sched), H3_VIDEOX_FUN=str(tmp_path), H3_MODELS=str(tmp_path),
                    H3_TURBO_LORA=str(tmp_path))
    r = subprocess.run(['bash', str(HYBRID / 'launch_control_worker.sh'), 'full'], capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=60)
    assert r.returncode != 0 and 'ALLOWED_HOSTS' in r.stderr and 'CALLED' not in r.stderr


def test_run_dir_and_bindings_fail_closed(no_bindings, tmp_path):
    with pytest.raises(SystemExit, match='H3_RUN'):
        PATHS.run_dir()
    with pytest.raises(SystemExit, match='not an existing directory'):
        PATHS.set_run_dir(tmp_path / 'missing')
    for key in ('H3_MODELS', 'H3_VIDEOX_FUN', 'H3_TURBO_LORA', 'GPU_SCHEDULER', 'H3_EVAL_DINO', 'H3_SAM2'):
        with pytest.raises(SystemExit, match=key):
            PATHS.binding(key)
        with pytest.raises(SystemExit, match=key):
            PATHS.require(key)
    with pytest.raises(SystemExit, match='unknown'):
        PATHS.binding('MODELS_ROOT')
    with pytest.raises(SystemExit, match='H3_MODELS'):
        PATHS.default_model_path('base_model')
    with pytest.raises(SystemExit, match='H3_VIDEOX_FUN'):
        PATHS.expand({'control_config': '${H3_VIDEOX_FUN}/config/x.yaml'})
    assert PATHS.python_exe() == sys.executable
    assert PATHS.binding('H3_HED_ANNOTATOR_REPO', required=False) is None
    no_bindings.setenv('H3_MODELS', str(tmp_path / 'missing'))
    with pytest.raises(SystemExit, match='does not exist'):
        PATHS.models_root()


def test_bindings_resolve_from_env_prefix_and_explicit_file(no_bindings, tmp_path):
    models, vx, run = tmp_path / 'models', tmp_path / 'vx', tmp_path / 'run'
    for d in (models, vx, run):
        d.mkdir()
    no_bindings.setenv('H3_RUN', str(run))
    assert PATHS.run_dir() == run.resolve()
    no_bindings.setenv('DATAFARM_H3_MODELS', str(models))
    bindings = tmp_path / 'bindings.json'; bindings.write_text(json.dumps({'H3_VIDEOX_FUN': str(vx)}))
    no_bindings.setenv('DATAFARM_BINDINGS', str(bindings))
    assert PATHS.default_model_path('base_model') == str(models / 'MiniMax-H3' / 'Ref2VA')
    assert PATHS.default_model_path('hed_tool') == str(vx / 'tools' / 'prepare_hed_control.py')
    assert PATHS.expand(['${H3_MODELS}/a', {'k': '${H3_VIDEOX_FUN}'}]) == [f'{models}/a', {'k': str(vx)}]
    no_bindings.setenv('H3_MODELS', str(vx))  # plain env wins over DATAFARM_ and the file
    assert PATHS.models_root() == vx


def test_runtime_defaults_cover_every_config_and_asset_binding(tmp_path):
    from datafarm.runtime import defaults, expand
    known = defaults(tmp_path)
    for key in PATHS.BINDING_KEYS:
        assert key in known, key
    for cfg in CONFIGS.glob('*.json'):
        expand(json.loads(cfg.read_text()), known)  # datafarm init materializes configs with these defaults
        assert set(PATHS.TOKEN.findall(cfg.read_text())) <= set(PATHS.BINDING_KEYS), cfg.name
    assets = json.loads((ROOT / 'configs/external-assets.json').read_text())
    hybrid = [e for e in assets['files'] if {'h3-control', 'h3-ood'} & set(e['profiles'])]
    assert len(hybrid) >= 10
    for entry in hybrid:
        expand(entry['path'], known)
        assert re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) and entry['bytes'] > 0, entry['name']
    names = {e['name'] for e in hybrid}
    for needed in ('MiniMax-H3-Fun-Controlnet-Union.safetensors', 'minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors',
                   'ControlNetHED.pth'):
        assert any(n.endswith(needed) for n in names), needed
    repo = assets['repositories'][0]
    assert repo['binding'] == 'H3_VIDEOX_FUN' and repo['upstream_commit'].startswith('6f3fb60')


def test_vendored_videox_fun_patch_matches_its_recorded_hash():
    patch = ROOT / 'vendor/videox_fun/videox-fun-datafarm.patch'
    assets = json.loads((ROOT / 'configs/external-assets.json').read_text())
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == assets['repositories'][0]['patch_sha256']
    assert not MACHINE_PATH.search(patch.read_text())


def recipe_config():
    return json.loads((CONFIGS / 'h3-control-hybrid-v1.json').read_text())


def test_default_recipe_has_the_validated_control_hybrid_settings():
    c = recipe_config()
    assert c['backend'] == 'control-hybrid' and c['canvas'] == 't3-guttered-1024x768'
    assert c['worker_profile'] == 'control-turbo' and c['denoise_forwards'] == 4
    assert c['chunk'] == {'frames': 192, 'advance': 120, 'keep_head': 72}
    ctl = c['control']
    assert ctl['type'] == 'soft-hed' and ctl['hed_min_intensity'] == 0
    assert ctl['edge_gate'] == {'type': 'luma-step', 'tau': 25, 'window': 7, 'dilate': 5}
    assert (ctl['context_scale'], ctl['inpaint_source'], ctl['mask'], ctl['guidance_scale']) == \
        (1.0, 'white', 'active-repaint', 1.0)
    assert ctl['white_video_reference'] == 39
    for key in ('base_model', 'control_checkpoint', 'control_config', 'videox_fun_repo', 'hed_tool', 'hed_checkpoint'):
        assert ctl[key].startswith('${H3_'), key
    assert c['prompt_lowering'] == 'h3-caption-chunkvis-notask'
    assert c['assembly'] == {'seam': 'min-diff', 'min_lead': 56}
    assert c['seeds']['count'] == 8 and c['seeds']['seed_offsets'] == list(range(8))
    assert c['postprocess']['exposure_lock']['anchor'] == 'first chunk'
    assert {'real_scene', 'robot', 'objects'} <= set(c['references_policy'])
    assert c['worker']['turbo_lora'] == '${H3_TURBO_LORA}'
    from datafarm.render.policy import H3_LOWERINGS
    assert c['prompt_lowering'] in H3_LOWERINGS


def test_recipe_configs_reference_existing_repository_files():
    for cfg in CONFIGS.glob('h3-*.json'):
        text = cfg.read_text()
        assert not MACHINE_PATH.search(text), cfg.name
        for rel in set(re.findall(r'((?:vendor|evaluations|configs|docs)/[\w./-]+\.(?:py|sh|txt|md|json))', text)):
            assert (ROOT / rel).is_file(), (cfg.name, rel)
    ood = json.loads((CONFIGS / 'h3-ood-restyle-v1.json').read_text())
    assert ood['authorization']['supported_axes'] == ['tabletop material/colour', 'light colour temperature/tint']
    assert 'grants no OOD authorization' in ood['authorization']['required']
    assert [Path(s['tool']).name for s in ood['steps']] == ['h3_sam2_table_masks.py', 'h3_ood_restyle.py']


def import_hybrid(name):
    pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    if str(HYBRID) not in sys.path:
        sys.path.insert(0, str(HYBRID))
    import importlib
    return importlib.import_module(name)


def test_default_recipe_is_valid_for_the_vendored_builder(no_bindings, tmp_path):
    hcp = import_hybrid('h3_package')
    with pytest.raises(SystemExit, match='H3_'):
        hcp.control_settings(recipe_config())
    models, vx = tmp_path / 'models', tmp_path / 'vx'
    models.mkdir(); vx.mkdir()
    no_bindings.setenv('H3_MODELS', str(models)); no_bindings.setenv('H3_VIDEOX_FUN', str(vx))
    ctl = hcp.control_settings(recipe_config())
    assert ctl['base_model'] == str(models / 'MiniMax-H3' / 'Ref2VA')
    assert ctl['hed_tool'] == str(vx / 'tools' / 'prepare_hed_control.py')
    assert recipe_config()['worker_profile'] in hcp.PROFILES
    ch = recipe_config()['chunk']
    for total in (124, 941, 1044, 1500):
        chunks = hcp.plan_chunks(total, ch['frames'], ch['advance'], ch['keep_head'])
        assert chunks[0]['keep_global'][0] == 0 and chunks[-1]['keep_global'][1] == total
        assert all(c['frames'] in hcp.LEGAL for c in chunks)
        assert all(a['keep_global'][1] == b['keep_global'][0] for a, b in zip(chunks, chunks[1:]))
    with pytest.raises(SystemExit, match='depth-dav2'):
        hcp.control_settings({**recipe_config(), 'control': {**recipe_config()['control'], 'type': 'depth-dav2'}})
    with pytest.raises(SystemExit):
        hcp.h3_paths.run_dir()


def synthetic_bundle(tmp_path, carrier=None, frames=None):
    _p, _body, intent, overlay = policy_fixture(tmp_path)
    ref = dict(intent['real_references'][0])
    ref.pop('sha256')
    ref['provenance'] = {**ref['provenance'], 'source': ref['provenance']['source']['path']}
    return {'schema': 'h3.task-bundle/1', 'task': 'unseen-task', 'run_id': 'new-run', 'layout': 't3',
            'carrier_fps': 24, 'mode_confirmations': {
                'sim2real': {'quote': 'synthetic test instruction', 'source': 'test only'}},
            'appearance_views': intent['appearance_views'], 'appearance_requirements': intent['appearance_requirements'],
            'appearance_invariants': intent['appearance_invariants'], 'real_baseline': intent['real_baseline'],
            'real_references': [ref], 'overlay': overlay, 'realpool': {'sources': []},
            'episodes': [{'id': 'ep0', 'carrier': str(carrier or tmp_path / 'none.mp4'), 'frame_range': None}]}


def test_bundle_mode_comes_only_from_the_bundle_confirmation(tmp_path):
    h3_bundle = import_hybrid('h3_bundle')
    bundle = synthetic_bundle(tmp_path)
    intent = h3_bundle.build_intent(bundle, 'sim2real', None)
    assert intent['mode_confirmation']['quote'] == 'synthetic test instruction'
    assert intent['run_id'] == 'new-run' and intent['authorized_ood_axes'] == []
    with pytest.raises(SystemExit, match='mode_confirmations.ood'):
        h3_bundle.build_intent(bundle, 'ood', None)
    bundle['mode_confirmations']['ood'] = {'quote': 'q', 'source': 's'}
    with pytest.raises(SystemExit, match='authorized_ood_axes'):
        h3_bundle.build_intent(bundle, 'ood', None)
    with pytest.raises(SystemExit, match='run_id'):
        h3_bundle.build_intent({**bundle, 'run_id': ''}, 'sim2real', None)
    contract = h3_bundle.compile_contract(bundle, 'sim2real', None, None, tmp_path / 'contract')
    assert contract['intent']['mode'] == 'sim2real' and (tmp_path / 'contract/prompt.txt').is_file()


def make_carrier(path, frames):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'color=c=white:s=320x270:r=24:d={frames / 24}', '-vf',
                    'drawbox=x=40:y=60:w=60:h=80:color=0x202020:t=fill,format=yuv420p', '-frames:v', str(frames),
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(path)], check=True)


@pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='needs ffmpeg/ffprobe')
def test_builder_cpu_path_produces_a_gate_checked_control_package(no_bindings, tmp_path):
    """Default recipe through h3_package build (HED stubbed: it needs torch): chunk clips, repaint mask,
    luma-step gate, measured chunk facts, chunkvis prompts, ticket and the render-policy audit."""
    hcp = import_hybrid('h3_package')
    run = tmp_path / 'run'
    for d in ('run/recipes', 'run/tasks/unseen-task', 'models', 'vx'):
        (tmp_path / d).mkdir(parents=True)
    carrier = tmp_path / 'carrier.mp4'
    make_carrier(carrier, 130)
    (run / 'tasks/unseen-task/bundle.json').write_text(json.dumps(synthetic_bundle(tmp_path, carrier)))
    recipe = run / 'recipes/test-hybrid.json'
    recipe.write_text(json.dumps({**recipe_config(), 'name': 'test-hybrid'}))
    no_bindings.setenv('H3_MODELS', str(tmp_path / 'models')); no_bindings.setenv('H3_VIDEOX_FUN', str(tmp_path / 'vx'))

    def fake_hed(job, out_dir, ctl, threads, device):
        target = out_dir / f"{job['chunk_id']}-hed-soft.mp4"
        shutil.copy(job['white'], target)
        return {'path': str(target), 'sha256': hcp.sha256(target), 'min_edge_intensity': 0,
                'mean_active_fraction_gt20': 0.0, 'mean_normalized_intensity': 0.0, 'device': 'stub',
                'tool_sha256': 'stub'}

    no_bindings.setattr(hcp, 'make_hed', fake_hed)
    out = run / 'pkgs/test-hybrid/unseen-task/ep0-sim2real'
    args = type('A', (), dict(bundle=str(run / 'tasks/unseen-task/bundle.json'), recipe=str(recipe), episode='ep0',
                              mode='sim2real', variant=None, seed_offset=0, out=str(out), force=False, hed_jobs=1,
                              hed_threads=1, hed_device='cpu', run_dir=None))()
    with pytest.raises(SystemExit, match='H3_RUN'):
        hcp.cmd_build(args)
    PATHS_IN_MODULE = hcp.h3_paths
    no_bindings.setattr(PATHS_IN_MODULE, '_RUN_DIR', None)
    no_bindings.setenv('H3_RUN', str(run))
    assert hcp.cmd_build(args) == 0
    manifest = json.loads((out / 'job-manifest.json').read_text())
    ticket = json.loads((out / 'ticket.json').read_text())
    assert manifest['backend'] == 'control-hybrid' and manifest['prompt_lowering'] == 'h3-caption-chunkvis-notask'
    assert ticket['profile'] == 'control-turbo'
    job = manifest['jobs'][0]
    assert job['generated_frames'] == 141 and job['control_meta']['device'] == 'stub'
    assert manifest['control']['edge_gate']['tau'] == 25 and job['control_video'].endswith('-gate25.mp4')
    assert set(job['chunk_facts']) <= {'head_absent_arms', 'empty_wrists', 'head_max_coloured_objects',
                                       'wrist_large_objects'}
    assert job['chunk_facts_detector'] == 'wrist-structure-v1'
    text = Path(job['prompt']).read_text()
    assert 'follows Video 1 only' in text and 'No cable is visible' in text
    assert (run / 'cache/hedgate').is_dir()
    receipt = hcp.audit(out)
    assert receipt['prompt_lowering'] == 'h3-caption-chunkvis-notask'
    Path(job['prompt']).write_text(text + ' Also draw a dog.')
    with pytest.raises(ValueError, match='chunk prompt mismatch'):
        hcp.audit(out)


def test_lora_fuser_is_shipped_and_unmodified():
    digest = hashlib.sha256((HYBRID / 'h3_lora.py').read_bytes()).hexdigest()
    assert digest == '0e8222b76e086f8f100092e325a08e86bfc8bd33fb404c8b8b0cdd4d3d56362c'


def test_evaluator_prompts_have_no_machine_paths():
    for prompt in ('seed-screen.txt', 'blind-review-agent.txt'):
        text = (EVAL / 'prompts' / prompt).read_text()
        assert not MACHINE_PATH.search(text), prompt


def test_render_h3_evaluator_is_protected_from_candidates():
    from datafarm.evolution.controller import PROTECTED, allowed
    for rel in ('evaluations/render-h3/h3eval.py', 'evaluations/render-h3/reviewer-instructions-v1.md',
                'evaluations/render-h3/prompts/seed-screen.txt'):
        assert allowed(rel, PROTECTED), rel


def test_render_skill_routes_h3_to_the_control_hybrid():
    skill = (ROOT / '.agents/skills/render-batch/SKILL.md').read_text()
    for rel in ('docs/reference/rendering.md', 'configs/render/h3-control-hybrid-v1.json',
                'configs/render/h3-ood-restyle-v1.json', 'evaluations/render-h3/'):
        assert rel in skill and (ROOT / rel).exists(), rel
