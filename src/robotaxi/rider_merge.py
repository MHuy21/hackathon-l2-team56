"""Turn generic person / motorcycle / bicycle masks (COCO-style detectors and datasets) into the competition classes.

Guideline (hackathon-l2-docs/docs/labeling_guideline.md, sections 3-4):

* a motorcycle with at least one person SITTING on it -> ONE ``Motorcyclist`` instance covering the vehicle and every
  person on it (passenger, child in front, rider waiting at a red light, ...);
* a bicycle with its rider -> ONE ``Cyclist`` instance;
* a vehicle nobody sits on (parked, or pushed by someone standing next to it) -> ``Motorcycle`` / ``Bicycle``;
* every other person -> ``Pedestrian`` (including the person walking a bike: two separate instances).

The "sits on" test is geometric, made on boxes and masks only:

1. the person's horizontal centre lies over the vehicle box (slightly widened);
2. the head is above the top of the vehicle box (a rider's torso sticks out above the bike);
3. the feet do not go below the vehicle's ground line (a pedestrian closer to the camera stands lower in the image);
4. most of the person's LOWER half (legs, seat) lies inside the vehicle box.

Each person joins at most one vehicle (the best scoring one); a vehicle may take several persons. Thresholds live in
:class:`RiderMergeConfig` so they can be tuned against hand-checked images without touching the logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .classes import BICYCLE, CYCLIST, MOTORCYCLE, MOTORCYCLIST, PEDESTRIAN
from .mask_ops import bbox_of_mask


@dataclass(frozen=True)
class RiderMergeConfig:
    center_margin: float = 0.10     # widen the vehicle box by this fraction of its width (each side) for test 1
    head_above: float = 0.0         # person top must be above vehicle top + head_above * vehicle height (test 2)
    feet_margin: float = 0.08       # person bottom may go below vehicle bottom by at most this * vehicle height (test 3)
    min_lower_inside: float = 0.45  # fraction of the person's lower-half mask pixels inside the vehicle box (test 4)


DEFAULT_CONFIG = RiderMergeConfig()


def rider_score(person: np.ndarray, vehicle: np.ndarray, cfg: RiderMergeConfig = DEFAULT_CONFIG) -> float:
    """Score in (0, 1] when ``person`` sits on ``vehicle`` according to the four tests, else 0.0."""
    pb, vb = bbox_of_mask(person), bbox_of_mask(vehicle)
    if pb is None or vb is None:
        return 0.0
    px0, py0, px1, py1 = pb
    vx0, vy0, vx1, vy1 = vb
    vw, vh = vx1 - vx0, vy1 - vy0
    cx = (px0 + px1) / 2
    if not (vx0 - cfg.center_margin * vw <= cx <= vx1 + cfg.center_margin * vw):
        return 0.0
    if py0 > vy0 + cfg.head_above * vh:
        return 0.0
    if py1 > vy1 + cfg.feet_margin * vh:
        return 0.0
    mid = (py0 + py1) // 2
    lower = person[mid:py1, px0:px1]
    n_lower = int(lower.sum())
    if n_lower == 0:
        return 0.0
    ys, xs = np.nonzero(lower)
    ys, xs = ys + mid, xs + px0
    inside = (xs >= vx0) & (xs < vx1) & (ys >= vy0) & (ys < vy1)
    frac = float(inside.mean())
    return frac if frac >= cfg.min_lower_inside else 0.0


def merge_riders(persons: Sequence[np.ndarray], motorcycles: Sequence[np.ndarray] = (), bicycles: Sequence[np.ndarray] = (),
                 cfg: RiderMergeConfig = DEFAULT_CONFIG) -> list[tuple[int, np.ndarray]]:
    """``[(class id, mask)]`` in competition classes from generic person / motorcycle / bicycle masks.

    Output order: merged and plain vehicles first (motorcycles, then bicycles, in input order), then pedestrians.
    Masks are unions and may still overlap with other objects: run ``mask_ops.resolve_overlaps`` afterwards.
    """
    vehicles = [(MOTORCYCLE, MOTORCYCLIST, m) for m in motorcycles] + [(BICYCLE, CYCLIST, m) for m in bicycles]
    riders: dict[int, list[int]] = {v: [] for v in range(len(vehicles))}
    for pi, p in enumerate(persons):
        scores = [rider_score(p, v[2], cfg) for v in vehicles]
        if scores and max(scores) > 0:
            riders[int(np.argmax(scores))].append(pi)
    out: list[tuple[int, np.ndarray]] = []
    used: set[int] = set()
    for vi, (plain_cls, rider_cls, vmask) in enumerate(vehicles):
        on = riders[vi]
        if on:
            union = vmask.astype(bool).copy()
            for pi in on:
                union |= persons[pi].astype(bool)
            out.append((rider_cls, union))
            used.update(on)
        else:
            out.append((plain_cls, vmask.astype(bool)))
    out += [(PEDESTRIAN, p.astype(bool)) for pi, p in enumerate(persons) if pi not in used]
    return out
