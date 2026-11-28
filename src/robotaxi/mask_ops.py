"""Mask <-> polygon conversion, overlap resolution and small-object rules.

Rasterisation uses pycocotools ``frPyObjects`` -- the same call the grader uses to turn label polygons into ground
truth masks -- so areas and IoUs computed here match the scorer. Conventions from the labeling guideline:

* masks of different instances never overlap: the object in FRONT keeps the pixels;
* one instance = one polygon line, so a mask split into pieces (occlusion by a pole, a wire...) is joined into a
  single polygon with zero-width bridges (rasterises to the union of the pieces);
* objects whose visible box has a short side < 1.25% of the image long side are not labelled, and ground truth whose
  area at 640 scale is < 64 px is ignored by the scorer anyway.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from shapely import make_valid
from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString, Point, Polygon

from .classes import IMGSZ, MIN_GT_AREA_PX, MIN_SHORT_SIDE_RATIO

LABEL_DECIMALS = 6  # yoloseg_io.DEFAULT_PRECISION


# ----------------------------------------------------------------------------------------------- rasterisation
def poly_to_rle(poly: np.ndarray, height: int, width: int) -> dict:
    """pycocotools RLE of one NORMALISED polygon at (height, width): identical to the grader's ``rasterize_polygon``."""
    flat = (np.asarray(poly, dtype=np.float64).reshape(-1, 2) * np.array([width, height], dtype=np.float64)).reshape(-1).tolist()
    return mask_utils.frPyObjects([flat], height, width)[0]


def poly_to_mask(poly: np.ndarray, height: int, width: int) -> np.ndarray:
    """Boolean (H, W) mask of one normalised polygon."""
    return mask_utils.decode(poly_to_rle(poly, height, width)).astype(bool)


def mask_to_rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def area_at_scale(pixels: float, height: int, width: int, imgsz: int = IMGSZ) -> float:
    """Pixel area -> area at the 640 letterbox scale used by the scorer's ignore rule."""
    r = imgsz / max(height, width)
    return float(pixels) * r * r


def mask_iou(a: np.ndarray, b: np.ndarray, box_a=None, box_b=None) -> float:
    """IoU of two boolean masks. Computed inside the union of their boxes only (full-frame logical ops on a
    1778x857 image made drafting a busy street ~15 s/image); boxes are recomputed when not given."""
    box_a = bbox_of_mask(a) if box_a is None else box_a
    box_b = bbox_of_mask(b) if box_b is None else box_b
    if box_a is None or box_b is None:
        return 0.0
    if box_a[2] <= box_b[0] or box_b[2] <= box_a[0] or box_a[3] <= box_b[1] or box_b[3] <= box_a[1]:
        return 0.0  # disjoint boxes: no shared pixel
    x0, y0 = min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    x1, y1 = max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
    ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    inter = np.count_nonzero(ca & cb)
    union = np.count_nonzero(ca | cb)
    return float(inter / union) if union else 0.0


# ----------------------------------------------------------------------------------------------- size rules
def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) inclusive-exclusive of the True pixels, or None for an empty mask.
    Row/column projections (any) instead of np.nonzero: ~10x faster on a full-size frame."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def too_small(mask: np.ndarray, min_short_side_ratio: float = MIN_SHORT_SIDE_RATIO, min_area_640: float = MIN_GT_AREA_PX) -> bool:
    """True when the guideline says "do not label": visible box short side < 1.25% of the image long side, or the
    mask area at 640 scale is below the scorer's ignore threshold."""
    h, w = mask.shape
    box = bbox_of_mask(mask)
    if box is None:
        return True
    short = min(box[2] - box[0], box[3] - box[1])
    if short < min_short_side_ratio * max(h, w):
        return True
    return area_at_scale(mask.sum(), h, w) < min_area_640


# ----------------------------------------------------------------------------------------------- mask -> polygon
def _crosses_any(p: np.ndarray, q: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> bool:
    """True when segment p-q PROPERLY crosses one of the segments starts[k]-ends[k] (touching at an end does not
    count): the same test as the official notebook's polygon_self_intersects."""
    if len(starts) == 0:
        return False

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    return bool(((orient(p, q, starts) * orient(p, q, ends) < 0) & (orient(starts, ends, p) * orient(starts, ends, q) < 0)).any())


def join_components(contours: Sequence[np.ndarray], candidates: int = 64) -> np.ndarray:
    """Join several closed contours (each (P, 2), pixel xy) into ONE polygon.

    The pieces are linked by a minimum spanning tree (Prim, from the largest piece): each piece hangs from the
    NEAREST piece already linked. The bridge is the shortest vertex pair whose segment crosses NO edge of any piece and
    no earlier bridge (up to ``candidates`` pairs are tried per piece pair). Plain "closest vertices" was not enough:
    in 3 of 15k COCO polygons the bridge cut through a thin, concave piece and the notebook reported the polygon as
    self-crossing. The walk goes out along a bridge, round the child piece (and its own children) and back along the
    same bridge, so bridges enclose no area and the polygon rasterises (even-odd, as pycocotools and cv2.fillPoly
    do) to the union of the pieces.
    """
    rings = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in contours if len(c) >= 3]
    if not rings:
        return np.zeros((0, 2))
    if len(rings) == 1:
        return rings[0]
    starts = np.concatenate(rings)
    ends = np.concatenate([np.roll(r, -1, axis=0) for r in rings])
    root = int(np.argmax([abs(cv2.contourArea(r.astype(np.float32))) for r in rings]))
    linked, todo = [root], set(range(len(rings))) - {root}
    attach: dict[int, dict[int, list[tuple[int, int]]]] = {k: {} for k in range(len(rings))}  # node -> vertex -> [(child, child vertex)]
    while todo:
        best = fallback = None
        for a in linked:
            for b in sorted(todo):
                d = ((rings[a][:, None, :] - rings[b][None, :, :]) ** 2).sum(-1)
                order = np.argsort(d, axis=None)[:candidates]
                i0, j0 = np.unravel_index(int(order[0]), d.shape)
                if fallback is None or d[i0, j0] < fallback[0]:
                    fallback = (float(d[i0, j0]), a, int(i0), b, int(j0))
                for flat in order:
                    i, j = np.unravel_index(int(flat), d.shape)
                    if best is not None and d[i, j] >= best[0]:
                        break
                    if not _crosses_any(rings[a][i], rings[b][j], starts, ends):
                        best = (float(d[i, j]), a, int(i), b, int(j))
                        break
        _, a, i, b, j = best if best is not None else fallback
        attach[a].setdefault(i, []).append((b, j))
        starts = np.vstack([starts, rings[a][i][None]])  # later bridges must not cross this one either
        ends = np.vstack([ends, rings[b][j][None]])
        linked.append(b)
        todo.discard(b)

    def walk(node: int, start: int, close: bool) -> list[np.ndarray]:
        ring, seq = rings[node], []
        for t in range(len(ring)):
            v = (start + t) % len(ring)
            seq.append(ring[v])
            for child, cv in attach[node].get(v, []):
                seq.extend(walk(child, cv, True))  # out along the bridge, round the child, back to its link vertex
                seq.append(ring[v])                # ... and back along the same bridge
        if close:
            seq.append(ring[start])
        return seq

    return np.asarray(walk(root, 0, False))


_MITRE = dict(join_style=JOIN_STYLE.mitre, mitre_limit=2.0)


def _grown_outer(contour_px: np.ndarray):
    """Region (shapely) of one OpenCV OUTER contour, grown by half a pixel.

    OpenCV contours run through the CENTRES of the boundary pixels, so a polygon drawn through them loses half a
    pixel on every side (about 10% of the area of a 20 px sign, enough to fail the high IoU thresholds). Shifting to
    continuous coordinates (+0.5) and buffering by 0.5 px with mitred joins gives back the pixel edges: measured
    round-trip IoU = 1.00 on small squares/discs, 0.999 on large shapes (vs 0.87-0.99 without).
    """
    c = contour_px.reshape(-1, 2).astype(np.float64) + 0.5
    grown = make_valid(Polygon(c)).buffer(0.5, **_MITRE) if len(c) >= 3 else Polygon()
    if grown.is_empty:  # one-pixel-wide line or single pixel: buffer the skeleton instead
        skeleton = LineString(c) if len(c) >= 2 else Point(c[0])
        grown = skeleton.buffer(0.5, cap_style=CAP_STYLE.square, **_MITRE)
    return grown


def _shrunk_hole(contour_px: np.ndarray):
    """Region of one OpenCV HOLE contour. The contour runs through the centres of the foreground pixels around the
    hole, so the hole itself is that polygon shrunk by half a pixel (a 1-pixel hole vanishes)."""
    c = contour_px.reshape(-1, 2).astype(np.float64) + 0.5
    if len(c) < 3:
        return Polygon()
    return make_valid(Polygon(c)).buffer(-0.5, **_MITRE)


def _polygons(geom) -> list[Polygon]:
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return [p for p in parts if isinstance(p, Polygon) and not p.is_empty]


def mask_to_polygon(mask: np.ndarray, epsilon_px: float = 0.25, min_component_px: int = 9) -> np.ndarray | None:
    """Normalised polygon (P, 2) of a boolean mask -- pieces AND holes in one line; ``None`` when nothing remains.

    Holes matter: after :func:`resolve_overlaps`, an object behind a smaller one (a bus behind a pedestrian) has a
    hole where the pedestrian is. Filling it would make the two labels overlap again. Every ring (outer boundaries
    and holes) is joined by zero-width bridges; pycocotools (the grader) and cv2.fillPoly (Ultralytics training) both
    fill even-odd, so holes stay empty (both checked).

    ``epsilon_px``: Douglas-Peucker tolerance in pixels. 0.25 re-rasterises EXACTLY (IoU 1.0 measured); 0.5 halves the
    vertex count but adds ~2% pixels on a 15 px disc. Rings (pieces or holes) smaller than ``min_component_px``
    pixels are dropped (anti-aliasing specks).
    """
    h, w = mask.shape
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hier = hier[0]
    rings = []
    for k, c in enumerate(contours):
        if hier[k][3] != -1:  # a hole: handled with its outer contour
            continue
        region = _grown_outer(c)
        child = hier[k][2]
        while child != -1:
            region = region.difference(_shrunk_hole(contours[child]))
            child = hier[child][0]
        for poly in _polygons(region):
            for ring in [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]:
                rings.append(ring)
    kept = []
    for ring in rings:
        if abs(cv2.contourArea(ring.astype(np.float32))) < min_component_px:
            continue
        if epsilon_px > 0:
            approx = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), epsilon_px, True).reshape(-1, 2)
            ring = approx.astype(np.float64) if len(approx) >= 3 else ring
        kept.append(ring)
    if not kept:
        return None
    poly = join_components(kept)
    poly[:, 0] /= w
    poly[:, 1] /= h
    return np.clip(poly, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- overlaps
def depth_order_by_bottom(masks: Sequence[np.ndarray]) -> list[int]:
    """Indices front-to-back for a road camera: the lower the bottom edge of an object in the image, the closer it is.
    Ties: the smaller object first (it is usually the one in front, e.g. a pedestrian before a bus)."""
    keys = []
    for k, m in enumerate(masks):
        box = bbox_of_mask(m)
        bottom = box[3] if box else -1
        keys.append((-bottom, int(m.sum()), k))
    return [k for *_, k in sorted(keys)]


def resolve_overlaps(masks: Sequence[np.ndarray], front_to_back: Sequence[int] | None = None) -> list[np.ndarray]:
    """Make masks disjoint: walking front to back, every mask loses the pixels already taken by objects in front.
    Returns masks in the ORIGINAL order (some may become empty or tiny: filter them afterwards)."""
    if not masks:
        return []
    order = list(front_to_back) if front_to_back is not None else depth_order_by_bottom(masks)
    if sorted(order) != list(range(len(masks))):
        raise ValueError("front_to_back must be a permutation of the mask indices")
    taken = np.zeros_like(masks[0], dtype=bool)
    out: list[np.ndarray] = [np.zeros(0)] * len(masks)
    for k in order:
        m = masks[k].astype(bool) & ~taken
        out[k] = m
        taken |= m
    return out


def disjoint_polygons(masks: Sequence[np.ndarray], front_to_back: Sequence[int] | None = None,
                      max_rounds: int = 4) -> list[np.ndarray | None]:
    """Polygons (normalised) of objects made disjoint, checked AFTER rasterisation; ``None`` for dropped objects.

    resolve_overlaps makes the MASKS disjoint, but a polygon re-rasterised by pycocotools can still spill one pixel
    onto a touching neighbour. Walking front to back, the pixels of a polygon that fall on the raster of an object in
    front are cut from its mask and the polygon is redrawn (a couple of rounds settle it). Objects that become too
    small (guideline size rule, or < 64 px at 640 scale after rasterisation) are dropped.
    """
    if not masks:
        return []
    h, w = masks[0].shape
    order = list(front_to_back) if front_to_back is not None else depth_order_by_bottom(masks)
    vis = resolve_overlaps(masks, order)
    def polygon(m: np.ndarray) -> np.ndarray | None:
        # quantised exactly like the label file (6 decimals), so the check rasterises what will be written
        p = None if too_small(m) else mask_to_polygon(m)
        return None if p is None else np.round(p, LABEL_DECIMALS)

    polys: list[np.ndarray | None] = [polygon(m) for m in vis]
    raster: list[np.ndarray | None] = [poly_to_mask(p, h, w) if p is not None else None for p in polys]
    for rnd in range(max_rounds):
        changed = False
        taken = np.zeros((h, w), dtype=bool)
        for k in order:
            if raster[k] is None:
                continue
            clash = raster[k] & taken
            if clash.any():
                if rnd > 0:  # a 1-pixel notch can be re-covered by the redrawn polygon: cut 1 px wider next time
                    clash = cv2.dilate(clash.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                vis[k] = vis[k] & ~clash
                polys[k] = polygon(vis[k])
                raster[k] = poly_to_mask(polys[k], h, w) if polys[k] is not None else None
                changed = True
            if raster[k] is not None:
                taken |= raster[k]
        if not changed:
            break
    for k in range(len(polys)):
        # size rules again on what will actually be rasterised (a polygon can come out a pixel thinner than its mask)
        if raster[k] is not None and too_small(raster[k]):
            polys[k] = None
    return polys


def objects_to_instances(objects: Sequence[tuple[int, np.ndarray]]) -> list["Instance"]:
    """``[(class id, mask)]`` -> label instances: disjoint after rasterisation, small objects removed."""
    from .yoloseg_io import Instance  # noqa: PLC0415 - keeps mask_ops importable on its own

    polys = disjoint_polygons([m for _, m in objects])
    return [Instance(cid, p) for (cid, _), p in zip(objects, polys) if p is not None and len(p) >= 3]


def overlap_pairs(masks: Sequence[np.ndarray], min_pixels: int = 1) -> list[tuple[int, int, int]]:
    """``(i, j, overlapping pixels)`` for every pair of masks sharing at least ``min_pixels`` pixels."""
    out = []
    boxes = [bbox_of_mask(m) for m in masks]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = boxes[i], boxes[j]
            if a is None or b is None or a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]:
                continue
            n = int(np.logical_and(masks[i], masks[j]).sum())
            if n >= min_pixels:
                out.append((i, j, n))
    return out
