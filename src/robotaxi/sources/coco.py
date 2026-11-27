"""COCO 2017 instances -> 11-class YOLO-seg (traffic subset).

Mapping (COCO category -> competition class):

    car -> Car, bus -> Bus, truck -> Truck, traffic light -> Traffic light, stop sign -> Sign,
    motorcycle (+ persons sitting on it) -> Motorcyclist, motorcycle alone -> Motorcycle,
    bicycle (+ rider) -> Cyclist, bicycle alone -> Bicycle,
    person seen through a car/bus/truck window -> dropped, any other person -> Pedestrian.

Known gaps (documented in configs/sources.yaml): COCO "truck" also contains pickups (Car in the guideline), only stop
signs are labelled (other traffic signs in the photos are unlabelled), no tricycle/cyclo/trailer classes.

Image selection: road scenes only (at least one vehicle / traffic light / stop sign), no image whose relevant
categories contain crowd (``iscrowd``) regions or a category we cannot map (train), then a seeded weighted sample that
favours images with the rarer classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from pycocotools import mask as mask_utils

from ..classes import BUS, CAR, NC, SIGN, TRAFFIC_LIGHT, TRUCK
from ..mask_ops import objects_to_instances
from ..rider_merge import drop_passengers, merge_riders, rider_candidates
from ..yoloseg_io import Instance

DIRECT = {"car": CAR, "bus": BUS, "truck": TRUCK, "traffic light": TRAFFIC_LIGHT, "stop sign": SIGN}
RIDEABLE = ("motorcycle", "bicycle")
USED = set(DIRECT) | set(RIDEABLE) | {"person"}
ROAD_EVIDENCE = set(DIRECT) | set(RIDEABLE)       # an image needs one of these to count as a road scene
EXCLUDE_IF_PRESENT = {"train", "airplane", "boat"}  # unmappable vehicle or not a road scene (airport, harbour) -> skip


@dataclass
class CocoIndex:
    images: dict[int, dict]
    anns_by_image: dict[int, list[dict]]
    cat_name: dict[int, str]
    licenses: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "CocoIndex":
        cat_name = {c["id"]: c["name"] for c in data["categories"]}
        by_img: dict[int, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            by_img[a["image_id"]].append(a)
        return cls({im["id"]: im for im in data["images"]}, dict(by_img), cat_name,
                   {lic["id"]: lic for lic in data.get("licenses", [])})

    def merge(self, other: "CocoIndex") -> "CocoIndex":
        """Combine train2017 and val2017 (image ids are unique across the two)."""
        clash = set(self.images) & set(other.images)
        if clash:
            raise ValueError(f"{len(clash)} image ids appear in both COCO files")
        return CocoIndex({**self.images, **other.images}, {**self.anns_by_image, **other.anns_by_image},
                         {**self.cat_name, **other.cat_name}, {**self.licenses, **other.licenses})


def eligible(index: CocoIndex, image_id: int) -> bool:
    anns = index.anns_by_image.get(image_id, [])
    names = {index.cat_name[a["category_id"]] for a in anns}
    if names & EXCLUDE_IF_PRESENT or not names & ROAD_EVIDENCE:
        return False
    return not any(a.get("iscrowd", 0) and index.cat_name[a["category_id"]] in USED for a in anns)


def select_images(index: CocoIndex, max_images: int, seed: int) -> list[int]:
    """Seeded weighted sample of eligible road-scene images; rarer classes weigh more.

    weight(image) = max over the mapped categories present of 1 / (#eligible images containing that category).
    """
    ids = sorted(i for i in index.images if eligible(index, i))
    present = {i: {index.cat_name[a["category_id"]] for a in index.anns_by_image[i]} & USED for i in ids}
    freq = Counter(n for names in present.values() for n in names)
    weights = np.array([max(1.0 / freq[n] for n in present[i] if n != "person") for i in ids], dtype=np.float64)
    if max_images >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ids), size=max_images, replace=False, p=weights / weights.sum())
    return sorted(ids[k] for k in picked)


def ann_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    elif isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


RIDER_POLICIES = ("skip", "merge")


@dataclass
class Converted:
    """Result for one image: ``instances`` (None when the image is skipped) and the unresolved rider candidates
    ``[(person annotation id, "motorcycle" | "bicycle")]`` that caused a skip (for the review list)."""

    instances: list[Instance] | None
    ambiguous: list[tuple[int, str]] = field(default_factory=list)


def convert_image(anns: Iterable[dict], cat_name: dict[int, str], height: int, width: int, rider_policy: str = "skip",
                  decisions: dict[int, bool] | None = None) -> Converted:
    """COCO annotations of one image -> competition instances (disjoint masks, small objects removed).

    rider_policy: what to do with a person that MIGHT sit on a motorcycle / bicycle (geometric candidate) and has no
    entry in ``decisions`` (person annotation id -> True rider / False not rider):
      * "skip"  -> skip the whole image and report the candidates (default: ~1 in 4 COCO candidates is a person
                   standing next to a parked bike, a wrong Motorcyclist/Cyclist label is worse than a missing image);
      * "merge" -> trust the geometry.
    """
    if rider_policy not in RIDER_POLICIES:
        raise ValueError(f"rider_policy must be one of {RIDER_POLICIES}")
    decisions = decisions or {}
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    person_ids: list[int] = []
    for a in anns:
        name = cat_name[a["category_id"]]
        if name in USED and not a.get("iscrowd", 0):
            groups[name].append(ann_mask(a, height, width))
            if name == "person":
                person_ids.append(a.get("id", -1))
    big_vehicles = groups["car"] + groups["bus"] + groups["truck"]
    keep = drop_passengers(groups["person"], big_vehicles)
    persons, pids = [groups["person"][i] for i in keep], [person_ids[i] for i in keep]
    n_moto = len(groups["motorcycle"])
    cands = rider_candidates(persons, groups["motorcycle"], groups["bicycle"])
    unresolved = [(pids[pi], "motorcycle" if vi < n_moto else "bicycle") for pi, vi in cands.items() if pids[pi] not in decisions]
    if unresolved and rider_policy == "skip":
        return Converted(None, unresolved)
    objects = merge_riders(persons, groups["motorcycle"], groups["bicycle"], decisions=[decisions.get(p) for p in pids])
    for name, cid in DIRECT.items():
        objects += [(cid, m) for m in groups[name]]
    return Converted(objects_to_instances(objects))


def class_counts(instances: Iterable[Instance]) -> list[int]:
    c = Counter(i.cls for i in instances)
    return [c.get(k, 0) for k in range(NC)]
