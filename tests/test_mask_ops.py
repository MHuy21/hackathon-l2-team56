import cv2
import numpy as np
import pytest

from robotaxi import mask_ops as mo

H, W = 540, 960


def _mask(draw):
    m = np.zeros((H, W), np.uint8)
    draw(m)
    return m.astype(bool)


@pytest.mark.parametrize("name, draw, min_iou", [
    ("square20", lambda m: cv2.rectangle(m, (100, 100), (119, 119), 1, -1), 0.999),
    ("disc8", lambda m: cv2.circle(m, (100, 100), 8, 1, -1), 0.999),
    ("ellipse", lambda m: cv2.ellipse(m, (400, 300), (10, 40), 30, 0, 360, 1, -1), 0.995),
    ("big", lambda m: (cv2.circle(m, (300, 200), 80, 1, -1), cv2.rectangle(m, (500, 100), (700, 400), 1, -1)), 0.995),
    ("thin_line", lambda m: cv2.line(m, (10, 10), (10, 60), 1, 1), 0.95),
])
def test_mask_polygon_roundtrip_matches_grader_rasterisation(name, draw, min_iou):
    m = _mask(draw)
    poly = mo.mask_to_polygon(m, min_component_px=0)
    assert poly is not None and poly.shape[1] == 2
    back = mo.poly_to_mask(poly, H, W)
    assert mo.mask_iou(m, back) >= min_iou, name


def test_split_mask_becomes_one_polygon_covering_all_pieces():
    # a car cut in two by a pole: one instance, one polygon line
    m = _mask(lambda x: (cv2.rectangle(x, (100, 100), (180, 160), 1, -1), cv2.rectangle(x, (190, 100), (260, 160), 1, -1)))
    poly = mo.mask_to_polygon(m)
    back = mo.poly_to_mask(poly, H, W)
    assert mo.mask_iou(m, back) > 0.99
    assert not back[130, 185]  # the pole gap stays empty (bridge has no area)


def test_three_pieces_joined():
    m = _mask(lambda x: [cv2.circle(x, c, 15, 1, -1) for c in [(100, 100), (300, 120), (200, 300)]])
    back = mo.poly_to_mask(mo.mask_to_polygon(m), H, W)
    assert mo.mask_iou(m, back) > 0.98


def _proper_crossings(p):
    """Same test as the official notebook's polygon_self_intersects (proper crossings of non-adjacent edges)."""
    n = len(p)
    a, b = p, np.roll(p, -1, axis=0)

    def orient(P, Q, R):
        return (Q[..., 0] - P[..., 0]) * (R[..., 1] - P[..., 1]) - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0])

    pa, pb, pc, pd = a[:, None], b[:, None], a[None], b[None]
    cr = (orient(pa, pb, pc) * orient(pa, pb, pd) < 0) & (orient(pc, pd, pa) * orient(pc, pd, pb) < 0)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None])
    return int((cr & ~((gap <= 1) | (gap == n - 1))).sum())


def test_joined_pieces_do_not_cross_a_middle_piece():
    # big piece left, small piece in the middle, piece on the right: a bridge big->right would cut the middle one
    m = _mask(lambda x: (cv2.ellipse(x, (150, 265), (100, 60), 0, 0, 360, 1, -1), cv2.rectangle(x, (280, 240), (300, 290), 1, -1),
                         cv2.rectangle(x, (330, 250), (400, 280), 1, -1)))
    poly = mo.mask_to_polygon(m)
    assert _proper_crossings(poly * [W, H]) == 0
    assert mo.mask_iou(m, mo.poly_to_mask(poly, H, W)) > 0.999


def test_hole_left_by_front_object_stays_empty():
    # a bus with a pedestrian standing fully inside its silhouette: after resolve_overlaps the bus has a hole
    bus = _mask(lambda x: cv2.rectangle(x, (100, 100), (400, 300), 1, -1))
    ped = _mask(lambda x: cv2.rectangle(x, (200, 150), (240, 260), 1, -1))
    bus_vis, ped_vis = mo.resolve_overlaps([bus, ped], front_to_back=[1, 0])
    back = mo.poly_to_mask(mo.mask_to_polygon(bus_vis), H, W)
    assert not np.logical_and(back, ped_vis).any()           # no overlap with the pedestrian
    assert mo.mask_iou(back, bus_vis) > 0.999
    cvm = np.zeros((H, W), np.uint8)                          # Ultralytics rasterises with cv2.fillPoly
    cv2.fillPoly(cvm, [np.round(mo.mask_to_polygon(bus_vis) * [W, H]).astype(np.int32)], 1)
    assert cvm[200, 220] == 0
    assert _proper_crossings(mo.mask_to_polygon(bus_vis) * [W, H]) == 0


def test_island_inside_hole_is_kept():
    m = _mask(lambda x: (cv2.rectangle(x, (100, 100), (400, 300), 1, -1), cv2.rectangle(x, (180, 140), (320, 260), 0, -1),
                         cv2.rectangle(x, (230, 180), (270, 220), 1, -1)))
    back = mo.poly_to_mask(mo.mask_to_polygon(m), H, W)
    assert mo.mask_iou(back, m) > 0.999


def test_disjoint_polygons_do_not_overlap_after_rasterisation():
    rng = np.random.default_rng(61)
    for _ in range(5):
        masks = []
        for _ in range(12):  # many overlapping / touching blobs
            cx, cy, r = int(rng.integers(50, W - 50)), int(rng.integers(50, H - 50)), int(rng.integers(8, 60))
            masks.append(_mask(lambda x, cx=cx, cy=cy, r=r: cv2.circle(x, (cx, cy), r, 1, -1)))
        polys = mo.disjoint_polygons(masks)
        rasters = [mo.poly_to_mask(np.round(p, 6), H, W) for p in polys if p is not None]
        assert rasters
        assert mo.overlap_pairs(rasters) == []


def test_objects_to_instances_drops_tiny_and_keeps_classes():
    big = _mask(lambda x: cv2.rectangle(x, (100, 100), (300, 300), 1, -1))
    tiny = _mask(lambda x: cv2.rectangle(x, (400, 400), (403, 403), 1, -1))
    insts = mo.objects_to_instances([(2, big), (9, tiny)])
    assert [i.cls for i in insts] == [2]


def test_specks_dropped_and_empty_mask():
    assert mo.mask_to_polygon(np.zeros((H, W), bool)) is None
    m = _mask(lambda x: x.__setitem__((5, 5), 1))
    assert mo.mask_to_polygon(m) is None


def test_resolve_overlaps_front_keeps_pixels():
    bus = _mask(lambda x: cv2.rectangle(x, (100, 100), (400, 300), 1, -1))     # bottom at 300 (farther)
    moto = _mask(lambda x: cv2.rectangle(x, (350, 250), (420, 360), 1, -1))   # bottom at 360 (closer)
    out = mo.resolve_overlaps([bus, moto])
    assert not np.logical_and(out[0], out[1]).any()
    assert out[1].sum() == moto.sum()           # front object untouched
    assert out[0].sum() < bus.sum()
    assert mo.overlap_pairs(out) == []
    assert mo.overlap_pairs([bus, moto])[0][:2] == (0, 1)


def test_resolve_overlaps_explicit_order():
    a = _mask(lambda x: cv2.rectangle(x, (0, 0), (50, 50), 1, -1))
    b = _mask(lambda x: cv2.rectangle(x, (25, 25), (75, 75), 1, -1))
    out = mo.resolve_overlaps([a, b], front_to_back=[0, 1])
    assert out[0].sum() == a.sum() and out[1].sum() < b.sum()
    with pytest.raises(ValueError):
        mo.resolve_overlaps([a, b], front_to_back=[0, 0])


def test_too_small_rules():
    # 1280 px long side -> 1.25% = 16 px short side
    big = np.zeros((720, 1280), bool)
    big[100:130, 100:110] = True   # 10 px wide -> too small
    assert mo.too_small(big)
    big[:] = False
    big[100:130, 100:120] = True   # 20 x 30 -> keep
    assert not mo.too_small(big)


def test_area_at_scale():
    assert mo.area_at_scale(100, 1280, 1280) == pytest.approx(25.0)
    assert mo.area_at_scale(100, 540, 960) == pytest.approx(100 * (640 / 960) ** 2)
