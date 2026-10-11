# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""An object measured from its masks in colour cameras, without a depth image.

The scenes are ray cast: boxes, upright cylinders and cone frustums on the table, seen by
RoboDojo's head camera and by the two wrist cameras where the arms start.  The masks are exact,
so what is tested is the geometry.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import robodojo_sight as sight
from pyrualean.hosts import robodojo_tools as tools

#: The end-effector poses the arms start an episode with (the simulator's ``<arm>_ee_pose``).
HOME = {
    "left": [-0.2995, -0.3523, 0.9215, 0.7071068, 0.0, 0.0, 0.7071068],
    "right": [0.3005, -0.3523, 0.9215, 0.7071068, 0.0, 0.0, 0.7071068],
}


def render(view: sight.View, shapes: list[dict[str, Any]]) -> np.ndarray:
    """Per pixel, the index of the shape a view shows (-1: the table or nothing)."""

    rows, cols = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    d = view.rays(rows, cols)
    pos = view.pos
    hit = np.full(view.shape, np.inf)
    owner = np.full(view.shape, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for index, shape in enumerate(shapes):
            cx, cy = shape["center"]
            z0 = kin.TABLE_Z
            if "size" in shape:
                lx, ly, h = shape["size"]
                a = math.radians(shape.get("yaw", 0.0))
                to_box = np.array(
                    [[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]]
                )
                o = to_box @ (pos - np.array([cx, cy, z0 + h / 2]))
                db = d @ to_box.T
                half = np.array([lx / 2, ly / 2, h / 2])
                t1, t2 = (-half - o) / db, (half - o) / db
                near, far = np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)
                t = np.where((near <= far) & (near > 0), near, np.inf)
            else:  # a cylinder, or a cone frustum when "top_radius" differs
                rad, h = shape["radius"], shape["height"]
                top_rad = shape.get("top_radius", rad)
                slope = (top_rad - rad) / h
                r0, rk = rad + slope * (pos[2] - z0), slope * d[..., 2]
                ox, oy = pos[0] - cx, pos[1] - cy
                qa = d[..., 0] ** 2 + d[..., 1] ** 2 - rk**2
                qb = 2 * (ox * d[..., 0] + oy * d[..., 1] - r0 * rk)
                disc = qb**2 - 4 * qa * (ox**2 + oy**2 - r0**2)
                root = np.sqrt(np.where(disc >= 0, disc, np.nan))
                t = np.full(view.shape, np.inf)
                for side in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)):
                    z = pos[2] + side * d[..., 2]
                    ok = (disc >= 0) & (side > 0) & (z >= z0) & (z <= z0 + h)
                    t = np.minimum(t, np.where(ok, side, np.inf))
                for zc, rc in ((z0 + h, top_rad), (z0, rad)):
                    cap = (zc - pos[2]) / d[..., 2]
                    px, py = pos[0] + cap * d[..., 0] - cx, pos[1] + cap * d[..., 1] - cy
                    t = np.minimum(t, np.where((cap > 0) & (px**2 + py**2 <= rc**2), cap, np.inf))
            owner = np.where(t < hit, index, owner)
            hit = np.minimum(hit, t)
        table = (kin.TABLE_Z - pos[2]) / d[..., 2]
        table = np.where(table > 0, table, np.inf)
    return np.where(hit < table, owner, -1)


def views() -> dict[str, sight.View]:
    return {
        "cam_head": sight.head_view(),
        "cam_left_wrist": sight.wrist_view(HOME["left"]),
        "cam_right_wrist": sight.wrist_view(HOME["right"]),
    }


def measured(shapes, index: int = 0, cameras=("cam_left_wrist", "cam_right_wrist"), **kwargs):
    """(geometry, measurement) of shape ``index`` from the head camera and the cameras named."""

    v = views()
    masks = {name: render(view, shapes) == index for name, view in v.items()}
    assert masks["cam_head"].sum() > 100
    others = [sight.Sight(v[name], masks[name]) for name in cameras if masks[name].sum() >= 30]
    res = sight.measure(sight.Sight(v["cam_head"], masks["cam_head"]), others, **kwargs)
    assert res is not None
    return tools.object_geometry(res["points"]), res


def off(geo: dict[str, Any], shape: dict[str, Any]) -> float:
    return math.hypot(
        geo["center_xyz"][0] - shape["center"][0], geo["center_xyz"][1] - shape["center"][1]
    )


def test_the_cameras_are_where_the_simulator_reports_them():
    head = sight.head_view()
    assert np.allclose(head.pos, [0.0, -0.41, 1.308])
    # it looks forward and 60 degrees down
    assert np.allclose(-head.rot[:, 2], [0.0, 0.5, -math.sin(math.radians(60))], atol=1e-6)
    assert head.fx == pytest.approx(288.1325)
    left = sight.wrist_view(HOME["left"])
    # what RoboDojo's simulator gave for the left wrist camera with the arm where it starts
    assert np.allclose(left.pos, [-0.299554, -0.267458, 0.97244], atol=2e-4)
    assert np.allclose(
        left.rot,
        [[1.0, 0.0, 0.0], [0.0, 0.5, -0.866], [0.0, 0.866, 0.5]],
        atol=5e-4,
    )
    # a point straight ahead of the camera lands in the middle of its image
    r, c, depth = left.project(left.pos - left.rot[:, 2] * 0.4)
    assert (float(r), float(c)) == pytest.approx((240.0, 320.0))
    assert float(depth) == pytest.approx(0.4)
    xy, ok = left.on_plane(np.array([240]), np.array([320]), kin.TABLE_Z)
    assert ok[0]
    assert xy[0, 0] == pytest.approx(left.pos[0], abs=1e-3)


@pytest.mark.parametrize(
    "shape",
    [
        {"center": (0.321, -0.055), "size": (0.035, 0.035, 0.0355), "yaw": 0},
        {"center": (0.25, 0.05), "size": (0.035, 0.035, 0.035), "yaw": 30},
        {"center": (-0.22, -0.05), "size": (0.03, 0.03, 0.03), "yaw": 15},
        {"center": (0.05, 0.10), "size": (0.062, 0.038, 0.038), "yaw": 40},
        {"center": (0.10, -0.10), "size": (0.057, 0.056, 0.014), "yaw": 20},
    ],
)
def test_a_box_seen_by_two_cameras_is_measured_to_millimetres(shape):
    geo, res = measured([shape])
    assert res["views"] >= 2 and res["kind"] == "prism"
    assert off(geo, shape) < 0.0025
    assert geo["height_m"] == pytest.approx(shape["size"][2], abs=0.003)
    assert geo["length_m"] == pytest.approx(max(shape["size"][:2]), abs=0.004)
    assert geo["width_m"] == pytest.approx(min(shape["size"][:2]), abs=0.004)
    # the fingers close across two faces
    turn = (geo["grasp_yaw"] - shape["yaw"]) % 90.0
    assert min(turn, 90.0 - turn) < 4.0


@pytest.mark.parametrize(
    "shape",
    [
        {"center": (0.28, 0.02), "radius": 0.033, "height": 0.12},
        {"center": (-0.25, 0.05), "radius": 0.027, "height": 0.19},
    ],
)
def test_a_can_and_a_tall_bottle_seen_by_two_cameras(shape):
    geo, res = measured([shape])
    assert res["views"] == 2
    assert off(geo, shape) < 0.002
    assert geo["height_m"] == pytest.approx(shape["height"], abs=0.003)
    assert geo["length_m"] == pytest.approx(2 * shape["radius"], abs=0.004)


def test_a_bowl_is_read_from_its_rim_once_a_second_camera_sees_it():
    bowl = {"center": (0.20, 0.0), "radius": 0.03, "top_radius": 0.058, "height": 0.05}
    geo, res = measured([bowl])
    assert res["views"] == 2 and res["kind"] == "hang"
    assert off(geo, bowl) < 0.0025
    assert geo["height_m"] == pytest.approx(0.05, abs=0.003)
    assert geo["length_m"] == pytest.approx(0.116, abs=0.006)
    # one camera cannot know: the same mask read as something standing on its whole footprint
    alone, res = measured([bowl], cameras=())
    assert res["views"] == 1 and res["kind"] == "prism"
    assert off(alone, bowl) > 0.008


def test_one_camera_alone_reads_a_box_or_a_can_closely():
    cube = {"center": (0.25, 0.05), "size": (0.035, 0.035, 0.035), "yaw": 30}
    geo, res = measured([cube], cameras=())
    assert res["views"] == 1 and res["match"] is None
    assert off(geo, cube) < 0.004
    assert geo["height_m"] == pytest.approx(0.035, abs=0.006)
    can = {"center": (0.28, 0.02), "radius": 0.033, "height": 0.12}
    geo, _ = measured([can], cameras=())
    assert off(geo, can) < 0.004


def test_a_height_measured_before_serves_when_only_one_camera_sees_the_object():
    bowl = {"center": (0.20, 0.0), "radius": 0.03, "top_radius": 0.058, "height": 0.05}
    geo, res = measured([bowl], cameras=(), known=(0.05, "hang"))
    assert res["views"] == 1 and res["kind"] == "hang"
    assert off(geo, bowl) < 0.003
    # where it stands now does not matter: the height is the object's
    moved = dict(bowl, center=(0.05, 0.12))
    geo, _ = measured([moved], cameras=(), known=(0.05, "hang"))
    assert off(geo, moved) < 0.003


def test_a_mask_of_something_else_in_the_other_camera_is_not_taken():
    cube = {"center": (0.25, 0.05), "size": (0.035, 0.035, 0.035), "yaw": 0}
    other = {"center": (0.36, 0.16), "size": (0.05, 0.05, 0.09), "yaw": 0}
    v = views()
    head = render(v["cam_head"], [cube, other]) == 0
    wrong = render(v["cam_right_wrist"], [cube, other]) == 1
    assert wrong.sum() > 200
    res = sight.measure(
        sight.Sight(v["cam_head"], head), [sight.Sight(v["cam_right_wrist"], wrong)]
    )
    assert res is not None and res["views"] == 1
    # of two masks from one camera the object's own is the one that fits
    right = render(v["cam_right_wrist"], [cube, other]) == 0
    both = [sight.Sight(v["cam_right_wrist"], wrong), sight.Sight(v["cam_right_wrist"], right)]
    res = sight.measure(sight.Sight(v["cam_head"], head), [both])
    assert res is not None and res["views"] == 2
    assert off(tools.object_geometry(res["points"]), cube) < 0.0025


def test_seeds_fall_on_the_object_in_the_other_camera():
    cube = {"center": (0.25, 0.05), "size": (0.05, 0.05, 0.05), "yaw": 0}
    v = views()
    head = sight.Sight(v["cam_head"], render(v["cam_head"], [cube]) == 0)
    spots = sight.seeds(head)
    assert len(spots) >= 3
    there = render(v["cam_right_wrist"], [cube]) == 0
    r, c, depth = v["cam_right_wrist"].project(spots)
    on = [
        there[int(round(ri)), int(round(ci))]
        for ri, ci, di in zip(r, c, depth, strict=True)
        if di > 0 and 0 <= ri < 479 and 0 <= ci < 639
    ]
    assert any(on)


def test_the_largest_part_of_a_mask_is_one_object():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:140, 200:240] = True  # the object
    mask[100:140, 245:250] = True  # the rest of it, behind a finger
    mask[300:320, 500:520] = True  # another object the same words fit
    kept = sight.largest_part(mask)
    assert kept[120, 220] and kept[120, 247]
    assert not kept[310, 510]
    assert sight.largest_part(kept).sum() == kept.sum()


def test_a_wrist_camera_knows_its_own_fingers():
    state = {
        "left_arm_joint_state": [0.0] * 6,
        "right_arm_joint_state": [0.0] * 6,
        "left_ee_joint_state": [1.0],
        "right_ee_joint_state": [1.0],
        "left_ee_pose": HOME["left"],
        "right_ee_pose": HOME["right"],
    }
    own = sight.robot_pixels(sight.wrist_view(HOME["left"]), state, own="left")
    # the two fingers fill the lower corners and leave the middle of the image free
    assert own[400, 20] and own[400, 620]
    assert not own[300:480, 250:390].any()
    assert not own[:200].any()
    closed = dict(state, left_ee_joint_state=[0.0])
    shut = sight.robot_pixels(sight.wrist_view(HOME["left"]), closed, own="left")
    assert shut[440, 320]  # closed, they meet in front of the camera
    # the head camera sees both arms
    assert sight.robot_pixels(sight.head_view(), state)[400:, :].sum() > 2000
