"""Box-only datasets (e.g. Vietnamese traffic sets from Roboflow / Kaggle) -> 11-class YOLO-seg with SAM masks.

Each source class is mapped (configs/mappings/<dataset>.yaml) to one of

* a competition class name (``Car``, ``Truck``, ``Motorcyclist``, ...): the box is segmented and kept as that class;
* a GENERIC part ``person`` / ``motorcycle`` / ``bicycle``: segmented, then combined by ``rider_merge`` (a person
  sitting on a bike becomes ONE Motorcyclist / Cyclist instance, other persons Pedestrian, passengers in cars dropped);
* ``null``: ignored.

The segmenter is any callable ``(image RGB (H, W, 3) uint8, boxes (N, 4) xyxy pixels) -> list of N bool masks``; the
production one wraps Ultralytics SAM 2 (GPU), tests use a fake. A SAM mask is clipped to its (slightly widened) prompt
box: SAM sometimes leaks into neighbours, and a label must not cover pixels outside the object.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import yaml

from ..classes import BUS, CAR, CLASSES, TRUCK, class_id
from ..mask_ops import objects_to_instances
from ..rider_merge import drop_passengers, merge_riders
from ..yoloseg_io import Instance

GENERIC = ("person", "motorcycle", "bicycle")
Segmenter = Callable[[np.ndarray, np.ndarray], Sequence[np.ndarray]]


@dataclass
class ClassMapping:
    """source class name -> competition class id, generic part name, or None (drop)."""

    target: dict[str, int | str | None]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassMapping":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data["mapping"])

    @classmethod
    def from_dict(cls, mapping: dict) -> "ClassMapping":
        target: dict[str, int | str | None] = {}
        for src, dst in mapping.items():
            if dst is None:
                target[str(src)] = None
            elif str(dst).lower() in GENERIC:
                target[str(src)] = str(dst).lower()
            else:
                target[str(src)] = class_id(str(dst))  # raises on a typo
        return cls(target)

    def resolve(self, source_name: str) -> int | str | None:
        if source_name not in self.target:
            raise KeyError(f"source class {source_name!r} has no entry in the mapping (map it, or map it to null)")
        return self.target[source_name]


def yolo_boxes_to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    """YOLO detection rows (cx, cy, w, h normalised) -> pixel xyxy."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    cx, cy, bw, bh = rows[:, 0] * width, rows[:, 1] * height, rows[:, 2] * width, rows[:, 3] * height
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).clip(0, [width, height, width, height])


def clip_to_box(mask: np.ndarray, box: np.ndarray, pad: float = 0.03) -> np.ndarray:
    h, w = mask.shape
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    xa, ya = max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - py)))
    xb, yb = min(w, int(np.ceil(x1 + px))), min(h, int(np.ceil(y1 + py)))
    out = np.zeros_like(mask, dtype=bool)
    out[ya:yb, xa:xb] = mask[ya:yb, xa:xb]
    return out


def convert_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, source_names: Sequence[str], mapping: ClassMapping,
                  segmenter: Segmenter) -> list[Instance]:
    """Segment every mapped box and build competition instances (disjoint masks, tiny objects removed)."""
    targets = [mapping.resolve(n) for n in source_names]
    keep = [k for k, t in enumerate(targets) if t is not None]
    if not keep:
        return []
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)[keep]
    masks = [clip_to_box(np.asarray(m, dtype=bool), b) for m, b in zip(segmenter(image, boxes), boxes)]
    direct: list[tuple[int, np.ndarray]] = []
    generic: dict[str, list[np.ndarray]] = {g: [] for g in GENERIC}
    for k, m in zip(keep, masks):
        t = targets[k]
        if isinstance(t, str):
            generic[t].append(m)
        else:
            direct.append((t, m))
    big = [m for c, m in direct if c in (CAR, BUS, TRUCK)]
    persons = [generic["person"][i] for i in drop_passengers(generic["person"], big)]
    objects = merge_riders(persons, generic["motorcycle"], generic["bicycle"]) + direct
    if not objects:
        return []
    return objects_to_instances(objects)


def ultralytics_sam_segmenter(weights: str = "sam2.1_b.pt", device: str | None = None) -> Segmenter:
    """SAM 2 through Ultralytics (installed by the official notebook); one call per image with all its boxes."""
    from ultralytics import SAM  # noqa: PLC0415 - heavy import, GPU environments only

    model = SAM(weights)

    def run(image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        if len(boxes) == 0:
            return []
        res = model(image[..., ::-1].copy(), bboxes=boxes.tolist(), device=device, verbose=False)[0]  # expects BGR like cv2
        if res.masks is None:
            return [np.zeros(image.shape[:2], dtype=bool) for _ in boxes]
        data = res.masks.data.cpu().numpy().astype(bool)
        if data.shape[1:] != image.shape[:2]:
            raise RuntimeError(f"SAM returned masks of shape {data.shape[1:]} for an image of {image.shape[:2]}")
        return list(data)

    return run


__all__ = ["ClassMapping", "convert_boxes", "yolo_boxes_to_xyxy", "ultralytics_sam_segmenter", "CLASSES"]
