"""Draft 11-class YOLO-seg labels for unlabelled images (GPU: Colab / Kaggle), for a human to correct in CVAT.

    python scripts/label/autolabel.py --images /content/vn_batch_01 --out /content/drafts/batch_01
    python scripts/label/autolabel.py --images ... --out ... --sam sam2.1_b.pt        # sharper masks, ~2x slower
    python scripts/label/autolabel.py --images ... --out ... --no-openvocab           # COCO classes only
    python scripts/label/autolabel.py --images kaggle_vn/ --out ... --sample 1500     # seeded subset of a big source

Output (a complete YOLO-seg folder, usable by build_dataset.py and to_cvat.py):
    <out>/images/<stem>.<ext>   copy of each input image
    <out>/labels/<stem>.txt     draft polygons (may be empty: the reviewer must still check the image)
    <out>/autolabel_report.csv  per image: detections, low-score count, rider merges, review priority (sorted)
Resumable: images whose label file exists in <out> are skipped.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.autolabel import (  # noqa: E402
    AutoLabelConfig, draft_labels, review_priority, ultralytics_seg_detector, ultralytics_yoloe_detector,
)
from robotaxi.classes import CLASSES, NC, TEAM_ID  # noqa: E402
from robotaxi.sources.bbox_sam import ultralytics_sam_segmenter  # noqa: E402
from robotaxi.sources.coco import class_counts  # noqa: E402
from robotaxi.yoloseg_io import iter_images, write_labels  # noqa: E402

FLAG_KEYS = ("detections", "low_score", "rider_merges", "riders", "passengers_dropped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder of unlabelled images (searched recursively)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolo26x-seg.pt", help="COCO-pretrained Ultralytics seg model (drafting only)")
    ap.add_argument("--yoloe-weights", default="yoloe-11l-seg.pt")
    ap.add_argument("--no-openvocab", action="store_true")
    ap.add_argument("--sam", default=None, help="SAM 2 weights to refine masks from boxes, e.g. sam2.1_b.pt")
    ap.add_argument("--coco-min", type=float, default=None, help="override the COCO-model score threshold")
    ap.add_argument("--openvocab-min", type=float, default=None, help="override the YOLOE score threshold")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (quick test)")
    ap.add_argument("--sample", type=int, default=0, help="seeded random sample of N images (large sources)")
    ap.add_argument("--seed", type=int, default=TEAM_ID)
    args = ap.parse_args(argv)

    detectors = [("coco", ultralytics_seg_detector(args.weights, device=args.device, imgsz=args.imgsz))]
    if not args.no_openvocab:
        detectors.append(("openvocab", ultralytics_yoloe_detector(weights=args.yoloe_weights, device=args.device, imgsz=args.imgsz)))
    segmenter = ultralytics_sam_segmenter(args.sam, args.device) if args.sam else None
    cfg = AutoLabelConfig()
    if args.coco_min is not None:
        cfg.min_score["coco"] = args.coco_min
    if args.openvocab_min is not None:
        cfg.min_score["openvocab"] = args.openvocab_min

    images = list(iter_images(args.images))
    if args.sample and len(images) > args.sample:
        images = sorted(random.Random(args.seed).sample(images, args.sample))
    images = images[: args.limit or None]
    report_path = args.out / "autolabel_report.csv"
    rows = []
    if report_path.is_file():
        with open(report_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    done = {r["file"] for r in rows}
    t0 = time.time()
    for k, img in enumerate(images, 1):
        out_img = args.out / "images" / img.name
        out_lbl = args.out / "labels" / f"{img.stem}.txt"
        if out_lbl.is_file() and img.name in done:
            continue
        with Image.open(img) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        res = draft_labels(rgb, detectors, cfg, segmenter)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img, out_img)
        write_labels(out_lbl, res.instances)
        row = {"file": img.name, "instances": len(res.instances), **{f: res.flags[f] for f in FLAG_KEYS},
               "priority": review_priority(res.flags)}
        row.update({f"n_{c.replace(' ', '_')}": n for c, n in zip(CLASSES, class_counts(res.instances))})
        rows.append(row)
        if k % 25 == 0:
            print(f"  {k}/{len(images)}  {(time.time() - t0) / k:.2f} s/image")

    args.out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (-int(r["priority"]), r["file"]))
    cols = ["file", "priority", "instances", *FLAG_KEYS] + [f"n_{c.replace(' ', '_')}" for c in CLASSES]
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    totals = [sum(int(r[f"n_{c.replace(' ', '_')}"]) for r in rows) for c in CLASSES]
    print(f"drafted {len(rows)} images -> {args.out}  (review order: {report_path})")
    for c in range(NC):
        print(f"  {c:>2} {CLASSES[c]:<14} {totals[c]:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
