"""Read-only QA for an extracted YOLO segmentation dataset (11 competition classes).

Usage, from the team56 repo::

    python scripts/qa/validate_labels.py ../hackathon-l2-docs/data/sample_data
    python scripts/qa/validate_labels.py data/interim/coco --report qa/coco.json
    python scripts/qa/validate_labels.py data/processed/r01 --check-val --strict

Supports images/ + labels/ and images/{train,val}/ + labels/{train,val}/,
as well as train/images/ + train/labels/. Coordinates are checked STRICTLY in
[0,1], even though the notebook reader tolerates small excursions. Missing
labels are errors; an explicit empty label file represents a reviewed background.
Tiny objects are warnings (two independent rules); --strict fails on warnings
too. All other violations are errors. Exit codes: 0 pass, 1 QA failed, 2 input
or CLI error. No labels or images are modified. JSON includes every issue;
terminal details are capped by --max-details. --check-val checks >=10 usable
instances per class in val/valid/validation (or a supplied validation-only root).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.classes import (  # noqa: E402
    CLASSES, MIN_GT_AREA_PX, MIN_GT_PER_CLASS_VAL, MIN_SHORT_SIDE_RATIO, NC,
)
from robotaxi.mask_ops import area_at_scale, poly_to_rle  # noqa: E402
from robotaxi.yoloseg_io import iter_images, label_path_for, parse_line  # noqa: E402


def has_proper_crossing(poly: np.ndarray) -> bool:
    """Catch crossing edges, allowing repeated vertices and zero-width bridges.

    A Shapely is_valid check would reject the team's joined multipart masks.
    Like the official notebook, only strict crossings count here. Vectorising
    one edge at a time avoids an NxN allocation on detailed SAM polygons.
    """
    ends = np.roll(poly, -1, axis=0)

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    for i in range(len(poly) - 2):
        stop = len(poly) - 1 if i == 0 else len(poly)
        c, d = poly[i + 2:stop], ends[i + 2:stop]
        a, b = poly[i], ends[i]
        if np.any((orient(a, b, c) * orient(a, b, d) < 0)
                  & (orient(c, d, a) * orient(c, d, b) < 0)):
            return True
    return False


def validate_dataset(root: Path, *, check_val: bool = False) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")
    images = list(iter_images(root))
    if not images:
        raise ValueError(f"no supported images under {root}; extract the dataset first")
    issues: list[dict] = []
    counts = [0] * NC
    val_counts = [0] * NC
    backgrounds = instances = 0
    expected_labels: set[Path] = set()
    val_names = {"val", "valid", "validation"}
    split_dataset = any(val_names.intersection(p.relative_to(root).parts[:-1]) for p in images)
    val_images = 0

    def issue(code, message, image=None, label=None, line=None, severity="error", **details):
        issues.append(dict(severity=severity, code=code, message=message,
                           image=str(image) if image else None,
                           label=str(label) if label else None, line=line, **details))

    for image in images:
        label = label_path_for(image)
        expected_labels.add(label)
        in_val = check_val and (not split_dataset or bool(val_names.intersection(image.relative_to(root).parts[:-1])))
        val_images += int(in_val)
        try:
            with Image.open(image) as im:
                im.load()  # detect truncated/corrupt image data, not just header errors
                width, height = ImageOps.exif_transpose(im).size
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            issue("image_read", str(exc), image, label)
            continue
        if not label.is_file():
            issue("missing_label", "Missing label; use an explicit empty file for reviewed backgrounds", image, label)
            continue
        try:
            lines = label.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            issue("label_read", str(exc), image, label)
            continue
        if not any(line.strip() for line in lines):
            backgrounds += 1
        # Store compressed RLEs instead of one full-resolution array per instance.
        masks = []
        for number, line in enumerate(lines, 1):
            fields = line.split()
            if not fields:
                continue
            def fail(code, message, **details):
                issue(code, message, image, label, number, **details)
            if len(fields) < 7 or len(fields) % 2 == 0:
                fail("format", f"Expected odd field count >=7 (class + >=3 xy pairs), got {len(fields)}; bbox labels are not accepted")
                continue
            try:
                cls = float(fields[0])
                coords = np.array([float(v) for v in fields[1:]])
            except ValueError:
                fail("numeric", "Class and coordinates must be numeric")
                continue
            invalid = False
            if not np.isfinite(cls) or not cls.is_integer() or not 0 <= cls < NC:
                fail("class_id", f"Class ID must be an integer in 0..{NC - 1}, got {fields[0]}")
                invalid = True
            if not np.isfinite(coords).all() or np.any((coords < 0) | (coords > 1)):
                fail("coordinates", "Coordinates must be finite and strictly within [0,1]")
                invalid = True
            if invalid:
                continue
            inst = parse_line(line)  # reuse competition-compatible IO after stricter QA checks
            poly = inst.poly
            if len(np.unique(poly, axis=0)) < 3 or np.linalg.matrix_rank(poly - poly[0]) < 2:
                fail("degenerate", "Polygon needs at least 3 distinct, non-collinear vertices")
                continue
            crossing = has_proper_crossing(poly)
            if crossing:
                fail("self_intersection", "Polygon edges cross; review its shape and multipart bridges")
            try:
                rle = poly_to_rle(poly, height, width)
                area = int(mask_utils.area(rle))
                box = mask_utils.toBbox(rle).tolist()
            except (ValueError, TypeError, OverflowError) as exc:
                fail("rasterize", str(exc))
                continue
            if area == 0:
                fail("empty_mask", "Polygon rasterizes to zero pixels at image resolution")
                continue
            instances += 1
            counts[inst.cls] += 1
            scaled_area = area_at_scale(area, height, width)
            short = min(box[2:])
            tiny_side = short < MIN_SHORT_SIDE_RATIO * max(height, width)
            tiny_area = scaled_area < MIN_GT_AREA_PX
            if tiny_side:
                fail("tiny_short_side", "Visible bbox short side is below 1.25% of image long side",
                     severity="warning", short_side_px=short,
                     threshold_px=MIN_SHORT_SIDE_RATIO * max(height, width))
            if tiny_area:
                fail("tiny_area", "Mask area at 640 scale is below 64 px; ignored by scorer",
                     severity="warning", area_px=area, area_at_640=scaled_area)
            if in_val and not tiny_side and not tiny_area and not crossing:
                val_counts[inst.cls] += 1
            masks.append((number, rle, box, inst.cls))
        for i, (line_a, a, box_a, cls_a) in enumerate(masks):
            ax, ay, aw, ah = box_a
            for line_b, b, (bx, by, bw, bh), cls_b in masks[i + 1:]:
                if ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay:
                    continue
                overlap = int(mask_utils.area(mask_utils.merge([a, b], intersect=True)))
                if overlap:
                    issue("overlap", f"Masks on lines {line_a} and {line_b} overlap by {overlap} pixels",
                          image, label, line_a, other_line=line_b, overlap_px=overlap,
                          class_ids=[cls_a, cls_b])

    # Restrict orphan scanning to label directories, so README/source .txt files
    # elsewhere in a dataset are never mistaken for annotations.
    for label in sorted(root.rglob("*.txt")):
        parts = label.relative_to(root).parts[:-1]
        if "labels" in parts and not any(p.startswith(".") or p == "__MACOSX" for p in parts) and label not in expected_labels:
            issue("orphan_label", "Label has no matching supported image", label=label)
    if check_val:
        for cls, count in enumerate(val_counts):
            if count < MIN_GT_PER_CLASS_VAL:
                issue("val_class_count", f"Validation {CLASSES[cls]} has {count} usable instances; need >= {MIN_GT_PER_CLASS_VAL}",
                      class_id=cls, count=count)
    return dict(root=str(root), images=len(images), backgrounds=backgrounds,
                instances=instances, class_counts=dict(zip(CLASSES, counts)),
                validation=dict(images=val_images, usable_class_counts=dict(zip(CLASSES, val_counts))) if check_val else None,
                errors=sum(i["severity"] == "error" for i in issues),
                warnings=sum(i["severity"] == "warning" for i in issues), issues=issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="extracted dataset directory")
    parser.add_argument("--report", type=Path, help="write full JSON report outside the input dataset")
    parser.add_argument("--strict", action="store_true", help="fail on warnings as well as errors")
    parser.add_argument("--check-val", action="store_true", help="require >=10 usable instances/class in validation")
    parser.add_argument("--max-details", type=int, default=50, help="maximum issues printed (JSON always contains all)")
    args = parser.parse_args(argv)
    if args.max_details < 0:
        parser.error("--max-details must be >=0")
    if args.report and args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error("--report must be outside the input dataset to preserve read-only QA")
    try:
        report = validate_dataset(args.root, check_val=args.check_val)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"Input/report error: {exc}", file=sys.stderr)
        return 2
    print(f"Images: {report['images']}; explicit backgrounds: {report['backgrounds']}; rasterized instances: {report['instances']}")
    print(f"Errors: {report['errors']}; warnings: {report['warnings']}")
    for cls, count in report["class_counts"].items():
        print(f"  {cls}: {count}")
    for item in report["issues"][:args.max_details]:
        location = item["label"] or item["image"] or "validation"
        if item["line"] is not None:
            location += f":{item['line']}"
        print(f"{item['severity'].upper()} [{item['code']}] {location}: {item['message']}")
    if len(report["issues"]) > args.max_details:
        print(f"... {len(report['issues']) - args.max_details} additional issues; use --report for full details")
    failed = bool(report["errors"] or (args.strict and report["warnings"]))
    print("FAIL" if failed else "PASS" + (" (warnings need review)" if report["warnings"] else ""))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
