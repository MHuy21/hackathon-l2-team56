"""Draft labels for unlabelled Vietnamese street images: detectors propose, a human (B) corrects in CVAT.

Pipeline for one image:

1. detectors return :class:`Detection` (source class name, score, full-resolution mask, box) -- a COCO-pretrained
   large segmentation model for car / bus / truck / motorcycle / bicycle / person / traffic light / stop sign, and an
   open-vocabulary model (YOLOE) prompted with "traffic sign", "cyclo", "trailer", ... for what COCO lacks;
2. detections below their source's confidence threshold are dropped; duplicates ACROSS detectors (mask IoU above
   ``dup_iou``) keep the higher score;
3. optional: masks are re-segmented by SAM 2 from their boxes (sharper edges than a 160 x 160 YOLO prototype);
4. source names are mapped (ClassMapping, same rules as the box converter): generic person / motorcycle / bicycle go
   through rider_merge (policy "merge": the reviewer checks every rider anyway), passengers inside cars are dropped;
5. masks made disjoint, objects below the size rules removed, polygons written.

Recall matters more than precision here: deleting a wrong polygon in CVAT takes a click, drawing a missing one takes
a minute. Hence the low default thresholds; :func:`review_flags` tells the reviewer where to look first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .classes import BUS, CAR, CLASSES, MOTORCYCLIST, CYCLIST, NC, TRUCK
from .mask_ops import bbox_of_mask, mask_iou, objects_to_instances
from .rider_merge import drop_passengers, merge_riders, rider_candidates
from .sources.bbox_sam import GENERIC, ClassMapping, Segmenter
from .yoloseg_io import Instance

# Source class name -> target. COCO names as Ultralytics reports them; YOLOE prompts as passed to set_classes.
COCO_MAPPING = {
    "person": "person", "bicycle": "bicycle", "motorcycle": "motorcycle", "car": "Car", "bus": "Bus", "truck": "Truck",
    "traffic light": "Traffic light", "stop sign": "Sign",
}
OPEN_VOCAB_PROMPTS = {
    # Shape words matter: on VN camera frames YOLOE scored "circular traffic sign" 0.55-0.71 on a no-entry sign that
    # "traffic sign" scored 0.07 (diagnostic 28/11 01:15, 3 golden-val frames).
    "circular traffic sign": "Sign",
    "triangular warning sign": "Sign",
    "rectangular blue road sign": "Sign",
    "traffic sign": "Sign",
    "cyclo": "Other vehicle",
    "three-wheeled cargo motorbike": "Other vehicle",
    "tuk-tuk": "Other vehicle",
    "trailer": "Other vehicle",
    "shipping container on a truck": "Other vehicle",
    "excavator": "Other vehicle",
    "road roller": "Other vehicle",
    "tractor": "Other vehicle",
}


@dataclass
class Detection:
    name: str          # source class name / prompt
    score: float
    mask: np.ndarray   # bool (H, W) at the image's resolution
    source: str = ""   # which detector


Detector = Callable[[np.ndarray], list[Detection]]


@dataclass
class AutoLabelConfig:
    mapping: ClassMapping = field(default_factory=lambda: ClassMapping.from_dict({**COCO_MAPPING, **OPEN_VOCAB_PROMPTS}))
    min_score: dict[str, float] = field(default_factory=lambda: {"coco": 0.25, "openvocab": 0.25})
    default_min_score: float = 0.25
    # Per source-class overrides. Far motorbikes / people on VN camera frames score 0.12-0.39 with the COCO model
    # (same diagnostic): with a single 0.30 threshold most of the traffic was missing from the drafts.
    min_score_by_name: dict[str, float] = field(default_factory=lambda: {"motorcycle": 0.12, "person": 0.15, "bicycle": 0.15})
    # A large object (truck partly out of frame, container) is rarely a false positive but scored 0.11-0.14.
    big_object_frac: float = 0.015
    big_object_min_score: float = 0.10
    big_object_names: tuple[str, ...] = ("car", "bus", "truck", "trailer", "shipping container on a truck")
    dup_iou: float = 0.6
    low_score: float = 0.5     # kept detections below this are flagged for review


@dataclass
class DraftResult:
    instances: list[Instance]
    flags: dict[str, int]      # counters for the review report


def dedupe(dets: Sequence[Detection], iou: float) -> list[Detection]:
    """Greedy cross-detector de-duplication by mask IoU, highest score first."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(mask_iou(d.mask, k.mask) < iou for k in kept):
            kept.append(d)
    return kept


def refine_with_sam(image: np.ndarray, dets: Sequence[Detection], segmenter: Segmenter) -> list[Detection]:
    """Replace each mask by SAM 2's mask for the detection's box (keeps the original when SAM returns nothing)."""
    boxes = [bbox_of_mask(d.mask) for d in dets]
    idx = [k for k, b in enumerate(boxes) if b is not None]
    if not idx:
        return list(dets)
    sam = segmenter(image, np.array([boxes[k] for k in idx], dtype=np.float64))
    out = list(dets)
    for k, m in zip(idx, sam):
        m = np.asarray(m, dtype=bool)
        if m.any():
            out[k] = Detection(dets[k].name, dets[k].score, m, dets[k].source)
    return out


def draft_labels(image: np.ndarray, detectors: Sequence[tuple[str, Detector]], cfg: AutoLabelConfig | None = None,
                 segmenter: Segmenter | None = None) -> DraftResult:
    """Run every detector on ``image`` (RGB uint8) and build draft instances in the 11 competition classes."""
    cfg = cfg or AutoLabelConfig()
    dets: list[Detection] = []
    big_px = cfg.big_object_frac * image.shape[0] * image.shape[1]
    for source, detect in detectors:
        src_thr = cfg.min_score.get(source, cfg.default_min_score)
        for d in detect(image):
            if d.name not in cfg.mapping.target or cfg.mapping.target[d.name] is None:
                continue
            thr = cfg.min_score_by_name.get(d.name, src_thr)
            if d.name in cfg.big_object_names and d.mask.sum() >= big_px:
                thr = min(thr, cfg.big_object_min_score)
            if d.score >= thr:
                d.source = source
                dets.append(d)
    dets = dedupe(dets, cfg.dup_iou)
    if segmenter is not None and dets:
        dets = refine_with_sam(image, dets, segmenter)

    generic: dict[str, list[np.ndarray]] = {g: [] for g in GENERIC}
    direct: list[tuple[int, np.ndarray]] = []
    low = 0
    for d in dets:
        t = cfg.mapping.target[d.name]
        low += d.score < cfg.low_score
        if isinstance(t, str):
            generic[t].append(d.mask)
        else:
            direct.append((t, d.mask))
    big = [m for c, m in direct if c in (CAR, BUS, TRUCK)]
    keep = drop_passengers(generic["person"], big)
    persons = [generic["person"][i] for i in keep]
    n_candidates = len(rider_candidates(persons, generic["motorcycle"], generic["bicycle"]))
    objects = merge_riders(persons, generic["motorcycle"], generic["bicycle"]) + direct
    out = objects_to_instances(objects)
    flags = {"detections": len(dets), "low_score": low, "rider_merges": n_candidates,
             "passengers_dropped": len(generic["person"]) - len(keep),
             "riders": sum(i.cls in (MOTORCYCLIST, CYCLIST) for i in out)}
    return DraftResult(out, flags)


def review_priority(flags: dict[str, int]) -> int:
    """Higher = look first: every rider merge and every low-score detection is a likely edit."""
    return 2 * flags.get("rider_merges", 0) + flags.get("low_score", 0) + flags.get("passengers_dropped", 0)


# --------------------------------------------------------------------------------------- Ultralytics wrappers (GPU)
def _results_to_detections(res, names: dict[int, str] | Sequence[str]) -> list[Detection]:
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return []
    masks = res.masks.data.cpu().numpy() > 0.5
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    h, w = res.orig_shape
    if masks.shape[1:] != (h, w):
        raise RuntimeError(f"masks {masks.shape[1:]} != image {(h, w)}: call predict with retina_masks=True")
    return [Detection(names[c], float(s), m) for c, s, m in zip(cls, conf, masks)]


def ultralytics_seg_detector(weights: str = "yolo26x-seg.pt", conf: float = 0.08, device: str | None = None,
                             imgsz: int = 1280) -> Detector:
    """COCO-pretrained Ultralytics segmentation model (any size: it only drafts labels, it is not submitted)."""
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(weights)

    def run(image: np.ndarray) -> list[Detection]:
        res = model.predict(image[..., ::-1].copy(), conf=conf, imgsz=imgsz, retina_masks=True, device=device, verbose=False)[0]
        return _results_to_detections(res, model.names)

    return run


def ultralytics_yoloe_detector(prompts: Sequence[str] = tuple(OPEN_VOCAB_PROMPTS), weights: str = "yoloe-11l-seg.pt",
                               conf: float = 0.08, device: str | None = None, imgsz: int = 1280) -> Detector:
    """Open-vocabulary YOLOE segmentation prompted with text (classes COCO does not have)."""
    from ultralytics import YOLOE  # noqa: PLC0415

    model = YOLOE(weights)
    names = list(prompts)
    try:
        model.set_classes(names, model.get_text_pe(names))
    except TypeError:  # newer Ultralytics computes the text embeddings itself
        model.set_classes(names)

    def run(image: np.ndarray) -> list[Detection]:
        res = model.predict(image[..., ::-1].copy(), conf=conf, imgsz=imgsz, retina_masks=True, device=device, verbose=False)[0]
        return _results_to_detections(res, names)

    return run


__all__ = ["AutoLabelConfig", "Detection", "DraftResult", "draft_labels", "review_priority", "CLASSES", "NC"]
