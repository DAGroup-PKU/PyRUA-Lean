"""Media helpers shared by the H3 render tools: hashes, ffprobe, frame reads, hard-link-or-copy."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import h3_paths

h3_paths.datafarm_src()
from datafarm.common import sha256_file as sha256  # noqa: E402

__all__ = ["VideoInfo", "link_or_copy", "nframes", "probe", "read_all", "read_frames", "sha256"]


class VideoInfo(NamedTuple):
    width: int
    height: int
    frames: int
    rate: str


def probe(path) -> VideoInfo:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,nb_read_packets,r_frame_rate",
                                   "-of", "json", str(path)], text=True)
    s = json.loads(out)["streams"][0]
    return VideoInfo(int(s["width"]), int(s["height"]), int(s["nb_read_packets"]), s["r_frame_rate"])


def nframes(path) -> int:
    return probe(path).frames


def read_frames(path, lo: int = 0, hi: int | None = None, step: int = 1, only: Iterable[int] | None = None,
                transform: Callable | None = None) -> dict:
    """Decoded BGR frames by index: every `step`-th frame of [lo, hi), or exactly the frames in `only`."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    want = set(only) if only is not None else None
    out, i = {}, 0
    while want or want is None:
        ok, frame = cap.read()
        if not ok or (hi is not None and i >= hi):
            break
        keep = i in want if want is not None else (i >= lo and (i - lo) % step == 0)
        if keep:
            out[i] = transform(frame) if transform else frame
            if want is not None:
                want.discard(i)
        i += 1
    cap.release()
    return out


def read_all(path):
    """All frames of a video as one array (frames, h, w, 3)."""
    import numpy as np
    return np.stack(list(read_frames(path).values()))


def link_or_copy(source, target) -> None:
    """Hard link `target` to `source` (a copy across filesystems); an existing target is left in place."""
    target = Path(target)
    try:
        os.link(source, target)
    except FileExistsError:
        pass
    except OSError:
        shutil.copy(source, target)
