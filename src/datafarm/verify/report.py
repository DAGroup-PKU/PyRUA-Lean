"""Human review page: every failing or abstaining check with visual evidence.

The page exists so a reviewer (human or agent) can confirm each verdict by
looking, which is how verifier calibration errors were caught before.  Evidence
tiles show carrier | generated | overlay at the evidence frame, cropped to the
evidence box (with margin) or to the whole panel.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datafarm.verify.spec import RenderVerifierSpec


def _read(path: str, frame: int, size: tuple[int, int]) -> np.ndarray | None:
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame))
    ok, image = capture.read()
    capture.release()
    if not ok:
        return None
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def _crop(image: np.ndarray, spec: RenderVerifierSpec, panel_name: str, bbox) -> np.ndarray:
    try:
        panel = spec.panel(panel_name)
        px, py, pw, ph = panel.box.x, panel.box.y, panel.box.w, panel.box.h
    except KeyError:
        px, py, pw, ph = 0, 0, image.shape[1], image.shape[0]
    if bbox:
        x, y, w, h = bbox
        margin = int(max(w, h) * 0.8) + 20
        x0, y0 = max(px, x - margin), max(py, y - margin)
        x1, y1 = min(px + pw, x + w + margin), min(py + ph, y + h + margin)
    else:
        x0, y0, x1, y1 = px, py, px + pw, py + ph
    return image[y0:y1, x0:x1].copy(), (x0, y0)


def _tile(spec, receipt, item: dict[str, Any], out: Path, name: str) -> str | None:
    inputs = receipt["inputs"]
    frame = int(item.get("frame", 0))
    frames = [frame]
    if item.get("kind") in ("jump", "luma_step", "freeze", "seam_jump", "seam_photometric_step",
                            "seam_structural_jump"):
        frames = [max(0, frame - 1), frame]  # a transition needs both sides
    elif item.get("end_frame") and item["end_frame"] != frame:
        frames.append(int((frame + item["end_frame"]) // 2))
    rows = []
    for f in frames:
        carrier = _read(inputs["carrier"], f, spec.layout_size)
        generated = _read(inputs["generated"], f, spec.layout_size)
        if carrier is None or generated is None:
            continue
        a, origin = _crop(carrier, spec, item.get("panel", ""), item.get("bbox"))
        b, _ = _crop(generated, spec, item.get("panel", ""), item.get("bbox"))
        blend = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        if item.get("bbox"):
            x, y, w, h = item["bbox"]
            for img in (a, b, blend):
                cv2.rectangle(img, (x - origin[0], y - origin[1]), (x - origin[0] + w, y - origin[1] + h),
                              (0, 0, 255), 2)
        row = np.hstack([a, b, blend])
        scale = 900 / max(row.shape[1], 1)
        row = cv2.resize(row, (900, max(1, int(row.shape[0] * scale))))
        cv2.putText(row, f"frame {f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(row)
    if not rows:
        return None
    path = out / f"{name}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return path.name


def _diverse(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin over (panel, kind) so every defect type gets a tile."""

    groups: dict[tuple, list] = {}
    for item in items:
        groups.setdefault((item.get("panel"), item.get("kind")), []).append(item)
    out: list[dict[str, Any]] = []
    while len(out) < limit and any(groups.values()):
        for key in list(groups):
            if groups[key] and len(out) < limit:
                out.append(groups[key].pop(0))
    return out


def write_report(spec: RenderVerifierSpec, receipt: dict[str, Any], out_dir: str | Path,
                 title: str = "Render verifier review") -> Path:
    out = Path(out_dir)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    colors = {"pass": "#2e7d32", "fail": "#c62828", "abstain": "#ef6c00", "not_applicable": "#757575"}
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>",
             "<style>body{font:14px system-ui;margin:16px;max-width:980px}"
             ".b{color:#fff;padding:1px 8px;border-radius:4px}details{margin:4px 0}"
             "img{max-width:100%;border:1px solid #ccc;margin:4px 0}pre{white-space:pre-wrap;font-size:12px}</style>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p>verdict <span class=b style='background:{colors.get(receipt['verdict'])}'>"
             f"{receipt['verdict']}</span> · spec <code>{html.escape(receipt['spec_id'])}</code> · "
             f"mode {receipt.get('realism_mode')} {receipt.get('perturbed_axes') or ''}</p>",
             "<p>Tiles: carrier | generated | 50% overlay; red box = evidence region.</p>"]
    for index, check in enumerate(receipt["checks"]):
        badge = f"<span class=b style='background:{colors.get(check['verdict'])}'>{check['verdict']}</span>"
        parts.append(f"<h3>{badge} {html.escape(check['name'])} <small>({check['family']})</small></h3>")
        if check.get("reason"):
            parts.append(f"<p>{html.escape(check['reason'])}</p>")
        if check["verdict"] in ("fail", "abstain"):
            for k, item in enumerate(_diverse(check.get("evidence", []), 12)):
                name = _tile(spec, receipt, item, evidence_dir, f"c{index:02d}-e{k:02d}")
                caption = html.escape(json.dumps(item))
                parts.append(f"<p><code>{caption}</code></p>")
                if name:
                    parts.append(f"<img src='evidence/{name}'>")
        parts.append("<details><summary>metrics</summary><pre>"
                     + html.escape(json.dumps(check.get("metrics", {}), indent=1, default=str)) + "</pre></details>")
    path = out / "review.html"
    path.write_text("\n".join(parts))
    return path
