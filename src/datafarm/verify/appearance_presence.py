"""Observed boundary and local contrast fallback for explicit presence keys.

Only actual rendered pixels can supplement colour presence. New-instance and
extent checks keep their original masks; this is not an identity classifier.
"""
import cv2
import numpy as np


def source_boundary_presence(source, generated_mask, generated_bgr, min_area=1000):
    result = generated_mask.copy()
    closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    gray = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 30, 60), np.ones((11, 11), np.uint8)) > 0
    lab = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    all_near = cv2.dilate(source, np.ones((11, 11), np.uint8)) > 0
    evidence = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        mask = labels == i
        boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8)) > 0
        # The panel border is not an observed object boundary.
        boundary[:3] = boundary[-3:] = False
        boundary[:, :3] = boundary[:, -3:] = False
        ring = (cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~all_near
        if boundary.sum() < 32 or ring.sum() < 100:
            continue
        background = np.median(lab[ring], axis=0)
        foreground = np.linalg.norm(lab - background, axis=2) > 20
        edge_support = float(edges[boundary].mean())
        contrast_fraction = float(foreground[mask].mean())
        median_separation = float(np.linalg.norm(np.median(lab[mask], axis=0) - background))
        # Coherent colour separation rejects textured empty table even when
        # its individual pixels and dense edges would pass the two fractions.
        # At least half of the source body must have actual contrast, matching
        # the dense missing-object recall threshold; unobserved pixels
        # are never filled from the source alone.
        accepted = edge_support >= .70 and contrast_fraction >= .50 and median_separation > 20
        if accepted:
            result[mask & foreground] = 255
        evidence.append(dict(bbox=stats[i, :4].tolist(), edge_support=edge_support,
                             contrast_fraction=contrast_fraction, median_separation=median_separation,
                             supplemented=accepted))
    return result, evidence
