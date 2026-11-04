import datetime
import json
import subprocess
from datafarm.common import atomic_write_json
from datafarm.common import sha256_file as sha
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write_json(path, value, sort_keys=False, mode=0o644, fsync=False)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,duration',
        '-of', 'json', str(path)], text=True))['streams'][0]


def verify_file(item):
    assert sha(item['path']) == item['sha256'], item['path']


def validate_clock(row, clock):
    """Check a source row against the batch config's clock: {transport_fps, canonical_fps, max_transport_frames}."""
    n, t = row['frames'], row['transport_frames']
    if not (0 < n <= t <= clock['max_transport_frames'] and t % 8 == 1 and 0 <= t - n < 8):
        raise ValueError('unsupported canonical/transport clock')
    if row['transport_fps'] != clock['transport_fps'] or row['canonical_fps'] != clock['canonical_fps']:
        raise ValueError('unsupported fps')


def finalize(raw, out, row, clock):
    validate_clock(row, clock)
    media = probe(raw)
    if int(media['nb_read_frames']) != row['transport_frames']:
        raise ValueError('provider frame count mismatch; raw retained')
    outputs = {}
    for fps in sorted({clock['transport_fps'], clock['canonical_fps']}):
        dest = out / f'render-{fps}fps.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(raw), '-vf', f"trim=end_frame={row['frames']},setpts=N/({fps}*TB)",
                        '-frames:v', str(row['frames']), '-r', str(fps), '-fps_mode', 'cfr', '-an',
                        '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)], check=True)
        info = probe(dest)
        assert int(info['nb_read_frames']) == row['frames'] and info['avg_frame_rate'] == f'{fps}/1'
        outputs[str(fps)] = {'path': str(dest), 'sha256': sha(dest), 'probe': info}
    return {'outputs': outputs, 'raw_sha256': sha(raw), 'raw_probe': media,
            'removed_terminal_padding_frames': row['transport_frames'] - row['frames'],
            'frame_map': 'output index equals canonical source index; no interpolation',
            'training_admitted': False}
