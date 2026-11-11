#!/usr/bin/env python3
"""Render-only replay of one captured white attempt (SAPIEN task experts).

The H3 renderer copies what the white carrier shows: pale simulator robot links become white arms, a cylinder proxy
becomes a plain cylinder, a dim simulator wall becomes a charcoal wall, and a white captured at a low image rate becomes
judder.  This tool runs the attempt's own frozen expert entry (`W/task/expert/run_seed.py SEED OUTDIR --image`) again
with the same seed and changes only what the cameras see:

  wrist_fov_factor  wrist camera focal length x F (same pose and principal point), applied to every SAPIEN camera
                    component named *wrist* before each render, whatever way the task built it; the original and the
                    used intrinsics are recorded.  H3 renders the wider wrist views and `h3_wrist_crop.py` crops them
                    back by F, so the output keeps the white's own wrist camera model;
  image_stride      physics steps per captured image, for scripts that expose IMAGE_STRIDE (a dense white instead of a
                    low-rate one; never interpolate a low-rate white);
  colors            render-only base colours: an entity name, "@dynamic" (every free rigid body), "@robot" (every robot
                    link), "@robot_links" ({link name or fnmatch pattern: rgba}, wins over "@robot"), a two-tone
                    {"body": rgba, "top": rgba} cylinder (a thin render-only disc on its upward end), and "@states"
                    ({entity: [{"when": rgb, "to": rgba, "emission"?: rgba}]}: a material the script switches to `when`
                    is shown as `to`, every frame it is in that state);
  shapes            {entity name: [part, ...]}: the entity's visual model is replaced by parametric parts (recolor,
                    barrel, heads, studs, apple, stem, box, revolve_z, handle, hide); collision shapes, mass and
                    inertia are untouched;
  extras            world-frame render-only parts (box, cylinder) with no physics component: scene dressing such as a
                    real wall or counter the simulator scene lacks.

Physics runs unchanged, so the trajectory must be identical: the replay's state log is compared with the attempt's
(states.jsonl, trace.jsonl or physics.jsonl: every field except the PhysX contact report, whose ordering is not
deterministic; or every array of log.npz) and the result is written to
OUT/white-replay.json.  The tool exits 2 when the log differs or when a requested change was not applied.  The replay
is a render input only; it never replaces the attempt, its judge verdict or its admission.

  python -m datafarm.simulation.white_replay ATTEMPT_DIR OUT_DIR --spec replay.json
  python -m datafarm.simulation.white_replay ATTEMPT_DIR OUT_DIR --verify-only
  python -m datafarm.simulation.white_replay --check-spec replay.json

Run it with the task's simulator interpreter on a host that can render (SAPIEN).  The spec is a task product
(schema "datafarm.white-replay/1"); see docs/reference/rendering.md, "Preparing the white carrier".
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SPEC_SCHEMA = "datafarm.white-replay/1"
REPORT_NAME = "white-replay.json"
SHAPE_KINDS = ("recolor", "barrel", "heads", "studs", "apple", "stem", "box", "revolve_z", "handle", "hide")
EXTRA_KINDS = ("box", "cylinder")
SPEC_KEYS = ("schema", "wrist_fov_factor", "image_stride", "colors", "shapes", "extras", "notes")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ spec
def _rgba(value, where: str, problems: list, n_min: int = 3) -> None:
    if not (isinstance(value, (list, tuple)) and n_min <= len(value) <= 4
            and all(isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0 for x in value)):
        problems.append(f"{where}: colour must be {n_min}-4 numbers in [0, 1], got {value!r}")


def validate_spec(spec: dict) -> list[str]:
    """Problems of a replay spec (empty list = valid).  Pure: no simulator needed."""
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema must be {SPEC_SCHEMA}")
    for key in spec:
        if key not in SPEC_KEYS:
            problems.append(f"unknown key {key!r}")
    f = spec.get("wrist_fov_factor", 1.0)
    if not isinstance(f, (int, float)) or not 0.3 <= float(f) <= 1.0:
        problems.append("wrist_fov_factor must be in [0.3, 1.0] (1.0 = unchanged)")
    stride = spec.get("image_stride")
    if stride is not None and (not isinstance(stride, int) or stride < 1):
        problems.append("image_stride must be a positive integer or null")
    colors = spec.get("colors") or {}
    if not isinstance(colors, dict):
        problems.append("colors must be an object")
        colors = {}
    for name, col in colors.items():
        if name == "@robot_links":
            if not isinstance(col, dict) or not col:
                problems.append("@robot_links must map link names/patterns to rgba")
                continue
            for pat, c in col.items():
                _rgba(c, f"@robot_links[{pat}]", problems)
        elif name == "@states":
            if not isinstance(col, dict):
                problems.append("@states must map entity names to rule lists")
                continue
            for ent, rules in col.items():
                if not isinstance(rules, list) or not rules:
                    problems.append(f"@states[{ent}] must be a non-empty list")
                    continue
                for k, r in enumerate(rules):
                    if not isinstance(r, dict) or "when" not in r or "to" not in r:
                        problems.append(f"@states[{ent}][{k}] needs when and to")
                        continue
                    _rgba(r["when"], f"@states[{ent}][{k}].when", problems)
                    _rgba(r["to"], f"@states[{ent}][{k}].to", problems)
                    if "emission" in r:
                        _rgba(r["emission"], f"@states[{ent}][{k}].emission", problems)
        elif isinstance(col, dict):
            if set(col) != {"body", "top"}:
                problems.append(f"colors[{name}]: a two-tone colour has exactly body and top")
            for part in ("body", "top"):
                if part in col:
                    _rgba(col[part], f"colors[{name}].{part}", problems)
        else:
            _rgba(col, f"colors[{name}]", problems)
    shapes = spec.get("shapes") or {}
    if not isinstance(shapes, dict):
        problems.append("shapes must be an object")
        shapes = {}
    for name, parts in shapes.items():
        if not isinstance(parts, list) or not parts:
            problems.append(f"shapes[{name}] must be a non-empty list of parts")
            continue
        for k, part in enumerate(parts):
            kind = part.get("kind") if isinstance(part, dict) else None
            if kind not in SHAPE_KINDS:
                problems.append(f"shapes[{name}][{k}]: kind must be one of {SHAPE_KINDS}")
                continue
            if kind != "hide":
                _rgba(part.get("color"), f"shapes[{name}][{k}].color", problems)
            need = {"apple": ("radii", "flat_z"), "stem": ("top_z",), "box": ("half_size",),
                    "revolve_z": ("profile",), "handle": ("dir_deg", "r_attach", "z0", "z1", "bulge", "thickness")}
            for key in need.get(kind, ()):
                if key not in part:
                    problems.append(f"shapes[{name}][{k}] ({kind}) needs {key}")
    extras = spec.get("extras") or []
    if not isinstance(extras, list):
        problems.append("extras must be a list")
        extras = []
    names = set()
    for k, e in enumerate(extras):
        if not isinstance(e, dict) or e.get("kind") not in EXTRA_KINDS:
            problems.append(f"extras[{k}]: kind must be one of {EXTRA_KINDS}")
            continue
        if not (isinstance(e.get("pose"), list) and len(e["pose"]) == 7):
            problems.append(f"extras[{k}]: pose must be [x, y, z, qw, qx, qy, qz]")
        if e["kind"] == "box" and not (isinstance(e.get("half_size"), list) and len(e["half_size"]) == 3):
            problems.append(f"extras[{k}]: a box needs half_size [x, y, z]")
        if e["kind"] == "cylinder" and not ("radius" in e and "half_length" in e):
            problems.append(f"extras[{k}]: a cylinder needs radius and half_length")
        _rgba(e.get("color"), f"extras[{k}].color", problems)
        if "emission" in e:
            _rgba(e["emission"], f"extras[{k}].emission", problems)
        name = e.get("name") or f"dress-{k}"
        if name in names:
            problems.append(f"extras[{k}]: duplicate name {name!r}")
        names.add(name)
    return problems


def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text())
    problems = validate_spec(spec)
    if problems:
        raise SystemExit("invalid white-replay spec:\n" + "\n".join(problems))
    return spec


def scale_k(K, f: float):
    """Camera matrix with fx, fy scaled by f (principal point kept)."""
    K = [list(map(float, r)) for r in K]
    K[0][0] *= f
    K[1][1] *= f
    return K


# ------------------------------------------------------------------ attempt entry and identity
ARGV_RECORDS = ("execution.json", "launch.json", "summary.json", "receipt.json")
STATE_LOGS = ("states.jsonl", "trace.jsonl", "physics.jsonl")


def _read(path: Path) -> dict:
    try:
        d = json.loads(path.read_text()) if path.is_file() else {}
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _recorded_source_sha(attempt: Path) -> str | None:
    """sha256 of the run_seed.py the attempt ran, from expert.json/receipt.json source_sha256 (a hash) or from
    source-snapshot.json (the source text itself)."""
    for name in ("expert.json", "receipt.json"):
        for key, value in (_read(attempt / name).get("source_sha256") or {}).items():
            if Path(key).name == "run_seed.py":
                return value
    for key, value in _read(attempt / "source-snapshot.json").items():
        if Path(key).name == "run_seed.py" and isinstance(value, str):
            return hashlib.sha256(value.encode()).hexdigest()
    return None


def resolve_entry(attempt: Path) -> tuple[Path, int, str]:
    """(run_seed.py, seed, how) of a captured attempt W/capture/attempts/<a>.

    Either an argv recorded by the capture driver or the expert run (execution.json / launch.json / summary.json /
    receipt.json), or the attempt's seed (attempt.json / expert.json / seed.json) plus the source the expert run
    recorded (a sha256 in expert.json or receipt.json source_sha256, or the text in source-snapshot.json): the frozen
    entry is the W/task/*/run_seed.py whose sha256 still equals it.  An argv entry is also checked against the
    recorded source when there is one: a replay must run the frozen source."""
    attempt = Path(attempt)
    want = _recorded_source_sha(attempt)
    for name in ARGV_RECORDS:
        argv = _read(attempt / name).get("argv") or []
        for k, x in enumerate(argv):
            if str(x).endswith("run_seed.py") and k + 1 < len(argv) and str(argv[k + 1]).lstrip("-").isdigit():
                script = Path(x)
                if want and script.is_file() and sha256(script) != want:
                    raise SystemExit(f"{script} changed since attempt {attempt.name} ran (sha256 differs from the "
                                     "recorded source); the replay must run the frozen source")
                return script, int(argv[k + 1]), name
    seed = next((d["seed"] for d in (_read(attempt / n) for n in ("attempt.json", "expert.json", "seed.json"))
                 if d.get("seed") is not None), None)
    if seed is not None and want and len(attempt.parents) >= 3:
        work = attempt.parents[2]
        for script in [work / "task/expert/run_seed.py", *sorted((work / "task").glob("*/run_seed.py"))]:
            if script.exists() and sha256(script) == want:
                return script, int(seed), "recorded seed + frozen source"
    raise SystemExit(f"no frozen run_seed.py entry recorded for {attempt}; pass --entry RUN_SEED_PY SEED")


def compare_logs(attempt: Path, out: Path) -> dict:
    """Trajectory identity of the replay: states.jsonl / trace.jsonl / physics.jsonl compared line by line (every field
    except the PhysX contact report), or every array of log.npz exactly."""
    attempt, out = Path(attempt), Path(out)
    for log in STATE_LOGS:
        if (attempt / log).exists():
            ol = (attempt / log).read_text().splitlines()
            nl = (out / log).read_text().splitlines() if (out / log).exists() else []
            first = None
            if len(ol) == len(nl):
                for i, (x, y) in enumerate(zip(ol, nl)):
                    if x == y:
                        continue
                    dx, dy = json.loads(x), json.loads(y)
                    for d in (dx, dy):
                        if isinstance(d, dict):
                            d.pop("contacts", None)
                    if dx != dy:
                        first = i
                        break
            else:
                first = min(len(ol), len(nl))
            return {"state_log": log, "lines": [len(ol), len(nl)], "byte_identical": ol == nl,
                    "state_log_identical": first is None, "first_state_diff_line": first,
                    "compared": "all fields except the PhysX contact report"}
    if (attempt / "log.npz").exists():
        import numpy as np
        if not (out / "log.npz").exists():
            return {"state_log": "log.npz", "state_log_identical": False, "missing": True}
        a_, b_ = np.load(attempt / "log.npz"), np.load(out / "log.npz")
        keys = sorted(set(a_.files) | set(b_.files))
        diff = [k for k in keys if k not in a_.files or k not in b_.files or a_[k].shape != b_[k].shape
                or not np.array_equal(a_[k], b_[k])]
        return {"state_log": "log.npz", "arrays": keys, "state_log_identical": not diff, "differing_arrays": diff,
                "byte_identical": (attempt / "log.npz").read_bytes() == (out / "log.npz").read_bytes(),
                "compared": "every array of log.npz exactly"}
    return {"state_log": None, "state_log_identical": False}


def wrist_cameras_scaled(report: dict, f: float) -> bool:
    """Every wrist camera the renderer used has fx, fy = f x its original (and there is at least one)."""
    cams = {n: c for n, c in (report.get("camera_intrinsics") or {}).items() if "wrist" in n}
    if not cams:
        return False
    for c in cams.values():
        k0, k1 = c.get("K_original"), c.get("K_used")
        if not (k0 and k1):
            return False
        for i in (0, 1):
            if abs(float(k1[i][i]) - f * float(k0[i][i])) > 1e-3 * max(1.0, abs(float(k0[i][i]))):
                return False
        if abs(float(k1[0][2]) - float(k0[0][2])) > 1e-3 or abs(float(k1[1][2]) - float(k0[1][2])) > 1e-3:
            return False
    return True


def replay_ok(report: dict, spec: dict) -> bool:
    """Every requested change applied, no colour leaked to an entity it was not meant for, and the trajectory
    identical."""
    colors = spec.get("colors") or {}
    shapes = spec.get("shapes") or {}
    extras = spec.get("extras") or []
    f = float(spec.get("wrist_fov_factor", 1.0))
    return bool(report.get("state_log_identical")
                and (f == 1.0 or wrist_cameras_scaled(report, f))
                and (not colors or report.get("recoloured_entities"))
                and not report.get("recolour_leaks")
                and (not spec.get("image_stride") or report.get("image_stride"))
                and set(shapes) <= set(report.get("shaped") or {})
                and len(report.get("extras_added") or {}) == len(extras))


# ------------------------------------------------------------------ parametric render meshes (numpy; trimesh to write)
def _write_mesh(path: Path, V, F) -> Path:
    import trimesh
    m = trimesh.Trimesh(vertices=V, faces=F, process=True)
    m.fix_normals()
    path.write_text(trimesh.exchange.obj.export_obj(m, include_normals=True))
    return path


def revolve_x(xs, rs, n: int = 64):
    """Closed surface of revolution about the local x axis: radius rs[i] at x = xs[i] (a drum barrel or head)."""
    import numpy as np
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = [[x, r * np.cos(p), r * np.sin(p)] for x, r in zip(xs, rs) for p in ph]
    F = []
    for i in range(len(xs) - 1):
        for j in range(n):
            a0, a1 = i * n + j, i * n + (j + 1) % n
            F += [[a0, a0 + n, a1], [a1, a0 + n, a1 + n]]
    c0, c1 = len(V), len(V) + 1
    V += [[xs[0], 0, 0], [xs[-1], 0, 0]]
    last = (len(xs) - 1) * n
    for j in range(n):
        F += [[c0, (j + 1) % n, j], [c1, last + j, last + (j + 1) % n]]
    return np.array(V, float), np.array(F, int)


def apple_mesh(radii, flat_z, top_dimple=0.22, bottom_dimple=0.12, lobes=0.02, n_t=48, n_p=72):
    """Apple-like closed mesh (z up): ellipsoid with a stem dimple, a calyx dimple, five faint lobes and a flat foot."""
    import numpy as np
    V, F = [], []
    for t in np.linspace(0, np.pi, n_t):
        k = 1 - top_dimple * np.exp(-(t / 0.32) ** 2) - bottom_dimple * np.exp(-((np.pi - t) / 0.42) ** 2)
        for p in np.linspace(0, 2 * np.pi, n_p, endpoint=False):
            s = k * (1 + lobes * np.cos(5 * p) * np.sin(t) ** 2)
            v = np.array([radii[0] * np.sin(t) * np.cos(p), radii[1] * np.sin(t) * np.sin(p), radii[2] * np.cos(t)]) * s
            v[2] = max(v[2], flat_z)
            V.append(v)
    for i in range(n_t - 1):
        for j in range(n_p):
            a0, a1 = i * n_p + j, i * n_p + (j + 1) % n_p
            F += [[a0, a0 + n_p, a1], [a1, a0 + n_p, a1 + n_p]]
    return np.array(V, float), np.array(F, int)


def revolve_z(profile, n: int = 96):
    """Surface of revolution about local z of a (radius, z) polyline (a thin-walled vessel)."""
    import numpy as np
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = [[r * np.cos(p), r * np.sin(p), z] for r, z in profile for p in ph]
    F = []
    for i in range(len(profile) - 1):
        for j in range(n):
            a0, a1 = i * n + j, i * n + (j + 1) % n
            F += [[a0, a1, a0 + n], [a1, a1 + n, a0 + n]]
    return np.array(V, float), np.array(F, int)


def handle_mesh(dir_deg, r_attach, z0, z1, bulge, th, n_arc=24, n_tube=12):
    """Mug handle: a tube of radius th along a half-ellipse from (r_attach, z0) out by `bulge` to (r_attach, z1)."""
    import numpy as np
    az = np.radians(dir_deg)
    out_, up = np.array([np.cos(az), np.sin(az), 0.0]), np.array([0.0, 0.0, 1.0])
    side = np.cross(up, out_)
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    C = [out_ * (r_attach + bulge * np.cos(t)) + up * (zc + hz * np.sin(t)) for t in np.linspace(-np.pi / 2, np.pi / 2, n_arc)]
    V, F = [], []
    for i, c in enumerate(C):
        tang = C[min(i + 1, len(C) - 1)] - C[max(i - 1, 0)]
        tang /= np.linalg.norm(tang)
        nrm = np.cross(tang, side)
        for k in range(n_tube):
            a = 2 * np.pi * k / n_tube
            V.append(c + th * (np.cos(a) * side + np.sin(a) * nrm))
    for i in range(len(C) - 1):
        for k in range(n_tube):
            a0, a1 = i * n_tube + k, i * n_tube + (k + 1) % n_tube
            F += [[a0, a1, a0 + n_tube], [a1, a1 + n_tube, a0 + n_tube]]
    return np.array(V, float), np.array(F, int)


# ------------------------------------------------------------------ SAPIEN hooks (imported only when replaying)
def _material(sapien, color, metallic=0.0, roughness=0.6, emission=None):
    m = sapien.render.RenderMaterial(base_color=list(color)[:3] + [1.0])
    m.metallic, m.roughness = float(metallic), float(roughness)
    if emission:
        m.emission = list(emission)[:3] + [1.0]
    return m


def install_extras(extras: list, added: dict) -> None:
    """Add the world-frame render-only parts before the first render (render body only, no PhysX component)."""
    import sapien
    orig = sapien.Scene.update_render

    def update_render(scene, *x, **kw):
        if not added:
            for k, e in enumerate(extras):
                m = _material(sapien, e["color"], e.get("metallic", 0.0), e.get("roughness", 0.6), e.get("emission"))
                if e["kind"] == "box":
                    shape = sapien.render.RenderShapeBox(list(e["half_size"]), m)
                else:
                    shape = sapien.render.RenderShapeCylinder(float(e["radius"]), float(e["half_length"]), m)
                body = sapien.render.RenderBodyComponent()
                body.attach(shape)
                ent = sapien.Entity()
                ent.name = e.get("name") or f"dress-{k}"
                ent.add_component(body)
                ent.set_pose(sapien.Pose(list(e["pose"][:3]), list(e["pose"][3:7])))
                scene.add_entity(ent)
                added[ent.name] = e["kind"]
        return orig(scene, *x, **kw)
    sapien.Scene.update_render = update_render


def install_shapes(shapes: dict, shaped: dict, out: Path) -> None:
    """Wrap ActorBuilder.build (and build_static/build_kinematic): an entity named in `shapes` gets its visual records
    replaced by the listed parts, relative to the frame of its first original visual record."""
    import numpy as np
    import sapien
    mesh_dir = out / "shape-meshes"

    def reshape(builder, name):
        spec = shapes.get(name)
        if not spec or getattr(builder, "_dfr_reshaped", False) or not builder.visual_records:
            return
        builder._dfr_reshaped = True
        base = builder.visual_records[0]
        frame = base.pose
        originals = list(builder.visual_records)
        info = {"replaced": [r.type for r in originals], "parts": []}
        builder.visual_records = []
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for k, part in enumerate(spec):
            kind, col = part["kind"], part.get("color")
            if kind == "recolor":
                for r in originals:
                    r.material = _material(sapien, col, roughness=part.get("roughness", 0.6))
                    builder.visual_records.append(r)
            elif kind == "barrel":   # along the original cylinder axis (local x of its visual pose)
                hl, R = float(base.length), float(base.radius)
                end, mid, head = part.get("end", 0.92), part.get("mid", 1.06), part.get("head_frac", 0.0)
                xs = np.linspace(-hl * (1 - head), hl * (1 - head), 41)
                V, F = revolve_x(xs, R * (end + (mid - end) * (1 - (xs / xs[-1]) ** 2)))
                builder.add_visual_from_file(str(_write_mesh(mesh_dir / f"{name}-{k}-barrel.obj", V, F)), pose=frame,
                                             material=_material(sapien, col))
            elif kind == "heads":   # discs + skin band at both barrel ends, slightly past the end (no z-fighting)
                hl, R = float(base.length), float(base.radius)
                end, band = part.get("end", 0.92), part.get("band", 0.06)
                for s in (1.0, -1.0):
                    xs = np.sort(np.array([s * hl * (1 - band), s * hl * 1.004]))
                    V, F = revolve_x(xs, np.array([R * end * 1.012] * 2))
                    builder.add_visual_from_file(str(_write_mesh(mesh_dir / f"{name}-{k}-head{int(s)}.obj", V, F)),
                                                 pose=frame, material=_material(sapien, col, roughness=0.8))
            elif kind == "studs":
                hl, R = float(base.length), float(base.radius)
                end, at, n, r = part.get("end", 0.92), part.get("at", 0.075), int(part.get("n", 28)), part.get("r", 0.0032)
                for s in (1.0, -1.0):
                    for j in range(n):
                        p = 2 * np.pi * j / n
                        off = sapien.Pose([s * hl * (1 - at), R * end * 1.03 * np.cos(p), R * end * 1.03 * np.sin(p)])
                        builder.add_sphere_visual(pose=frame * off, radius=r,
                                                  material=_material(sapien, col, metallic=0.8, roughness=0.3))
            elif kind == "apple":
                V, F = apple_mesh(part["radii"], part["flat_z"], part.get("top_dimple", 0.22),
                                  part.get("bottom_dimple", 0.12), part.get("lobes", 0.02))
                builder.add_visual_from_file(str(_write_mesh(mesh_dir / f"{name}-{k}-apple.obj", V, F)), pose=frame,
                                             material=_material(sapien, col, roughness=0.45))
            elif kind == "stem":
                q = sapien.Pose(q=[np.sqrt(.5), 0, -np.sqrt(.5), 0])   # cylinder x axis -> world z
                tilt = sapien.Pose(q=[np.cos(.1), np.sin(.1), 0, 0])
                builder.add_cylinder_visual(pose=frame * sapien.Pose([0, 0, part["top_z"]]) * tilt * q,
                                            radius=part.get("r", 0.0022), half_length=part.get("half_length", 0.007),
                                            material=_material(sapien, col, roughness=0.9))
            elif kind == "box":
                pz = part.get("pose", [0, 0, 0])
                pose = sapien.Pose(list(pz[:3]), list(pz[3:7]) if len(pz) >= 7 else [1, 0, 0, 0])
                builder.add_box_visual(pose=pose, half_size=list(part["half_size"]),
                                       material=_material(sapien, col, part.get("metallic", 0.0),
                                                          part.get("roughness", 0.6), part.get("emission")))
            elif kind == "revolve_z":
                V, F = revolve_z(part["profile"])
                builder.add_visual_from_file(str(_write_mesh(mesh_dir / f"{name}-{k}-rev.obj", V, F)),
                                             pose=sapien.Pose(),
                                             material=_material(sapien, col, roughness=part.get("roughness", 0.4)))
            elif kind == "handle":
                V, F = handle_mesh(part["dir_deg"], part["r_attach"], part["z0"], part["z1"], part["bulge"],
                                   part["thickness"])
                builder.add_visual_from_file(str(_write_mesh(mesh_dir / f"{name}-{k}-handle.obj", V, F)),
                                             pose=sapien.Pose(),
                                             material=_material(sapien, col, roughness=part.get("roughness", 0.4)))
            # "hide": no visual part (a simulator proxy that contradicts the real scene)
            info["parts"].append(kind)
        shaped[name] = info

    orig_build = sapien.ActorBuilder.build

    def build(self, name="", *x, **kw):
        reshape(self, name or getattr(self, "name", "") or "")
        return orig_build(self, name, *x, **kw) if name else orig_build(self, *x, **kw)
    sapien.ActorBuilder.build = build
    for meth in ("build_static", "build_kinematic"):
        orig_m = getattr(sapien.ActorBuilder, meth, None)
        if orig_m is None:
            continue

        def wrapped(self, *x, _orig=orig_m, **kw):
            reshape(self, getattr(self, "name", "") or "")
            return _orig(self, *x, **kw)
        setattr(sapien.ActorBuilder, meth, wrapped)


def copy_material(sapien, m):
    """A new RenderMaterial with every value and texture of m."""
    c = sapien.render.RenderMaterial()
    for k in ("base_color", "emission", "specular", "roughness", "metallic", "transmission", "ior",
              "transmission_roughness", "base_color_texture", "emission_texture",
              "metallic_texture", "normal_texture", "roughness_texture", "transmission_texture"):
        try:
            v = getattr(m, k)
        except Exception:  # noqa: BLE001
            continue
        if v is not None:
            setattr(c, k, v)
    return c


def install_unshared_materials(unshared: dict) -> None:
    """Give every visual shape its own RenderMaterial.  Task scenes often build the table, objects and robot links with
    ONE material object (e.g. a single 'white'); changing its base_color would then recolour all of them.  Render-only:
    physics never reads render materials."""
    import sapien
    from sapien.wrapper.actor_builder import ActorBuilder   # LinkBuilder inherits build_render_component
    orig = ActorBuilder.build_render_component

    def build_render_component(self):
        for r in self.visual_records:
            if isinstance(r.material, sapien.render.RenderMaterial):
                r.material = copy_material(sapien, r.material)
                unshared["shapes"] = unshared.get("shapes", 0) + 1
        return orig(self)
    ActorBuilder.build_render_component = build_render_component


def _shape_materials(sh) -> list:
    mats = []
    try:
        mats.append(sh.get_material())
    except Exception:  # noqa: BLE001 - triangle meshes carry per-part materials
        pass
    try:
        mats += [pt.get_material() for pt in sh.get_parts()]
    except Exception:  # noqa: BLE001
        pass
    return [m for m in mats if m is not None]


def _entity_colors(e) -> list:
    return [tuple(round(float(x), 6) for x in m.base_color) for c in e.components
            for sh in (getattr(c, "render_shapes", []) or []) for m in _shape_materials(sh)]


def install_colors(colors: dict, shaped: dict, recoloured: dict, leaks: list | None = None) -> dict:
    """Render-only colours, applied before the first render; '@states' rules are applied every frame.  Every shape gets
    its own material first, and any untargeted entity whose colour changed anyway is appended to leaks.  Returns the
    two-tone record {entity: {radius, half_length, up_end}}."""
    import sapien
    leaks = [] if leaks is None else leaks
    install_unshared_materials(recoloured.setdefault("_unshared", {}))
    capped: dict = {}
    two_tone = {k: v for k, v in colors.items() if not k.startswith("@") and isinstance(v, dict)}
    if two_tone:
        orig_add_cyl, orig_build = sapien.ActorBuilder.add_cylinder_visual, sapien.ActorBuilder.build

        def add_cylinder_visual(self, *x, **kw):
            self.__dict__.setdefault("_dfr_cyl", []).append((x, dict(kw)))
            return orig_add_cyl(self, *x, **kw)

        def build(self, name="", *x, **kw):
            spec = two_tone.get(name)
            if spec:
                for args, kw0 in getattr(self, "_dfr_cyl", []):
                    pose = kw0.get("pose", args[0] if args else sapien.Pose())
                    radius = kw0.get("radius", args[1] if len(args) > 1 else 1.0)
                    hl = kw0.get("half_length", args[2] if len(args) > 2 else 1.0)
                    up = (self.initial_pose * pose).to_transformation_matrix()[:3, 0][2]   # cylinder axis = local x
                    s = 1.0 if up >= 0 else -1.0
                    orig_add_cyl(self, pose=pose * sapien.Pose([s * (hl + 0.0006), 0, 0]), radius=radius * 0.995,
                                 half_length=0.0006, material=list(spec["top"]))
                    capped[name] = {"radius": radius, "half_length": hl, "up_end": "+x" if s > 0 else "-x"}
            return orig_build(self, name, *x, **kw) if name else orig_build(self, *x, **kw)
        sapien.ActorBuilder.add_cylinder_visual = add_cylinder_visual
        sapien.ActorBuilder.build = build

    state_rules = colors.get("@states") or {}
    state_mats: dict = {}
    links = colors.get("@robot_links") or {}
    orig_update = sapien.Scene.update_render

    def apply_states(scene):
        if not state_rules:
            return
        if not state_mats:
            for e in scene.entities:
                if e.name in state_rules:
                    for c in e.components:
                        for sh in getattr(c, "render_shapes", []) or []:
                            try:
                                state_mats.setdefault(e.name, []).append(sh.get_material())
                            except Exception:  # noqa: BLE001 - triangle meshes carry per-part materials
                                pass
        for name, mats in state_mats.items():
            for m in mats:
                bc = list(m.base_color)[:3]
                for rule in state_rules[name]:
                    if max(abs(a - b) for a, b in zip(bc, rule["when"][:3])) < 0.02:
                        m.base_color = list(rule["to"])
                        if rule.get("emission"):
                            m.emission = list(rule["emission"])
                        recoloured[f"{name}@state"] = recoloured.get(f"{name}@state", 0) + 1
                        break

    def update_render(scene, *x, **kw):
        apply_states(scene)
        if not recoloured.get("_done"):
            recoloured["_done"] = True
            ents = list(scene.entities)
            before = [_entity_colors(e) for e in ents]   # to catch a colour leaking through a shared material
            targeted = set()
            for i, e in enumerate(ents):
                comps = e.components
                if e.name in shaped:   # its new visual parts carry their own materials
                    continue
                is_link = any("ArticulationLink" in type(c).__name__ for c in comps)
                dynamic = any(type(c).__name__ == "PhysxRigidDynamicComponent" for c in comps)
                if is_link:
                    col = next((c_ for pat, c_ in links.items() if fnmatch.fnmatchcase(e.name, pat)), None) \
                        or colors.get("@robot")
                else:
                    col = colors.get(e.name) if not e.name.startswith("@") else None
                col = col or (colors.get("@dynamic") if dynamic else None)
                if not col:
                    continue
                targeted.add(i)
                if isinstance(col, dict):
                    col = col["body"]
                col = list(col)[:3] + [1.0]
                n = 0
                for c in comps:
                    shapes_ = list(getattr(c, "render_shapes", []) or [])
                    if e.name in capped and shapes_:
                        shapes_ = shapes_[:-1]   # the added top disc keeps its own colour
                    for sh in shapes_:
                        for m in _shape_materials(sh):
                            m.base_color = col
                            n += 1
                recoloured[e.name] = n
            leaks.extend(e.name or f"<entity {i}>" for i, e in enumerate(ents)
                         if i not in targeted and _entity_colors(e) != before[i])
        return orig_update(scene, *x, **kw)
    sapien.Scene.update_render = update_render
    return capped


def install_wrist_fov(f: float, patched: list, cam_intrinsics: dict) -> None:
    """Scale fx, fy of every camera component whose name contains 'wrist' by f, at the renderer: before each render,
    a wrist camera still at its original focal length (first render, or the task reset it) is set to f x it, principal
    point kept.  The original and the used intrinsics of every camera are recorded, so the report shows what was
    rendered, whatever way the task built its cameras."""
    import sapien
    orig_ur = sapien.Scene.update_render

    def update_render(scene, *x, **kw):
        for e in scene.entities:
            for c in e.components:
                if type(c).__name__ != "RenderCameraComponent":
                    continue
                nm = getattr(c, "name", "") or e.name
                K = [list(map(float, r)) for r in c.get_intrinsic_matrix()]
                rec = cam_intrinsics.setdefault(nm, {"K_original": K, "width": int(c.width), "height": int(c.height),
                                                     "applied": 0})
                if "wrist" in nm and f != 1.0 and abs(K[0][0] - rec["K_original"][0][0]) < 1e-6:
                    c.set_focal_lengths(rec["K_original"][0][0] * f, rec["K_original"][1][1] * f)
                    rec["applied"] += 1
                    if nm not in patched:
                        patched.append(nm)
                rec["K_used"] = [list(map(float, r)) for r in c.get_intrinsic_matrix()]
        return orig_ur(scene, *x, **kw)
    sapien.Scene.update_render = update_render


def replay(attempt: Path, out: Path, spec: dict, entry: tuple | None = None) -> dict:
    if entry:   # explicit entry, still checked against the source the attempt recorded
        script, seed, how = Path(entry[0]).resolve(), int(entry[1]), "caller --entry"
        want = _recorded_source_sha(attempt)
        if want and sha256(script) != want:
            raise SystemExit(f"--entry {script} is not the source attempt {attempt.name} ran (sha256 differs)")
    else:
        script, seed, how = resolve_entry(attempt)
    # load by the resolved path: task entries often derive their task root from Path(__file__).resolve() and then
    # take paths relative to it, which fails when the run directory is reached through a symlink
    script = Path(script).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(script.parent))
    sys.path.insert(1, str(script.parent.parent))
    mod_spec = importlib.util.spec_from_file_location("run_seed_white_replay", script)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.argv = [str(script), str(seed), str(out), "--image"]
    mod_spec.loader.exec_module(mod)
    stride = None
    if spec.get("image_stride") and hasattr(mod, "IMAGE_STRIDE"):
        stride = [int(mod.IMAGE_STRIDE), int(spec["image_stride"])]
        mod.IMAGE_STRIDE = int(spec["image_stride"])
    patched, cam_intrinsics, added, shaped, recoloured, leaks = [], {}, {}, {}, {}, []
    f = float(spec.get("wrist_fov_factor", 1.0))
    install_wrist_fov(f, patched, cam_intrinsics)
    if spec.get("extras"):
        install_extras(spec["extras"], added)
    if spec.get("shapes"):
        install_shapes(spec["shapes"], shaped, out)
    capped = install_colors(spec["colors"], shaped, recoloured, leaks) if spec.get("colors") else {}
    rc = mod.main() if hasattr(mod, "main") else mod.run(seed, str(out), True)
    report = {"schema": "datafarm.white-replay-report/1", "attempt": str(attempt), "out": str(out), "seed": seed,
              "script": str(script), "script_sha256": sha256(script), "entry_record": how, "rc": rc, "spec": spec,
              "wrist_fov_factor": f, "patched_cameras": sorted(set(patched)), "camera_intrinsics": cam_intrinsics,
              "image_stride": stride, "recoloured_entities": {k: v for k, v in recoloured.items() if not k.startswith("_")},
              "unshared_material_shapes": (recoloured.get("_unshared") or {}).get("shapes", 0),
              "recolour_leaks": sorted(set(leaks)),
              "top_discs": capped, "shaped": shaped, "extras_added": added}
    report.update(compare_logs(attempt, out))
    report["ok"] = replay_ok(report, spec)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("attempt", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--spec", help="white-replay spec JSON (schema datafarm.white-replay/1)")
    ap.add_argument("--verify-only", action="store_true", help="recompute the identity check of an existing replay")
    ap.add_argument("--check-spec", metavar="SPEC", help="validate a spec and exit")
    ap.add_argument("--entry", nargs=2, metavar=("RUN_SEED_PY", "SEED"),
                    help="explicit frozen entry when the attempt records none (checked against its recorded source)")
    a = ap.parse_args(argv)
    if a.check_spec:
        problems = validate_spec(json.loads(Path(a.check_spec).read_text()))
        print(json.dumps({"spec": a.check_spec, "problems": problems}, indent=1))
        return 1 if problems else 0
    if not (a.attempt and a.out):
        ap.error("ATTEMPT_DIR and OUT_DIR are required")
    attempt, out = Path(a.attempt).resolve(), Path(a.out).resolve()
    if a.verify_only:
        rep = json.loads((out / REPORT_NAME).read_text())
        rep.update(compare_logs(attempt, out))
        rep["ok"] = replay_ok(rep, rep.get("spec") or {})
        (out / REPORT_NAME).write_text(json.dumps(rep, indent=2))
        print(json.dumps({k: rep.get(k) for k in ("ok", "state_log_identical", "first_state_diff_line")}))
        return 0 if rep["ok"] else 2
    if not a.spec:
        ap.error("--spec is required for a replay")
    rep = replay(attempt, out, load_spec(a.spec), tuple(a.entry) if a.entry else None)
    print(json.dumps({k: rep.get(k) for k in ("ok", "state_log_identical", "patched_cameras", "image_stride",
                                              "shaped", "extras_added")}))
    return 0 if rep["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
