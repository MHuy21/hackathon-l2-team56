"""Box-only dataset (YOLO detection export, e.g. Roboflow "YOLOv8") -> 11-class YOLO-seg, masks from SAM 2.

Needs a GPU environment with Ultralytics (Colab / Kaggle, after Step 0 of the official notebook, or
``pip install ultralytics==8.4.214``). Resumable: images whose label file already exists in --out are skipped.

    python scripts/data/bbox_to_seg_sam.py --src /content/vn_traffic --mapping configs/mappings/vn_traffic.yaml \
        --name vntraffic --out data/interim/vntraffic --weights sam2.1_b.pt

Input layout: any folder tree with images/ and labels/ (train/valid/test splits are all used), plus a data.yaml whose
``names`` give the source class names. Lines with 5 fields are boxes; longer lines (polygons) are reduced to their box.
Output: <out>/images/<name>_<stem>.<ext>, <out>/labels/<name>_<stem>.txt, <out>/manifest_<name>.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.classes import CLASSES, NC  # noqa: E402
from robotaxi.sources.bbox_sam import ClassMapping, convert_boxes, ultralytics_sam_segmenter, yolo_boxes_to_xyxy  # noqa: E402
from robotaxi.sources.coco import class_counts  # noqa: E402
from robotaxi.yoloseg_io import iter_images, label_path_for, write_labels  # noqa: E402


def source_names(src: Path) -> list[str]:
    cands = sorted(src.rglob("data.yaml"))
    if not cands:
        raise SystemExit(f"no data.yaml under {src}: cannot know the source class names")
    names = yaml.safe_load(cands[0].read_text(encoding="utf-8"))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def read_boxes(label: Path) -> tuple[list[int], np.ndarray]:
    ids, rows = [], []
    if not label.is_file():
        return ids, np.zeros((0, 4))
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        vals = np.array([float(x) for x in parts[1:]])
        if len(parts) == 5:
            rows.append(vals)
        else:  # polygon line: use its bounding box
            pts = vals[: len(vals) // 2 * 2].reshape(-1, 2)
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            rows.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
        ids.append(int(float(parts[0])))
    return ids, np.array(rows).reshape(-1, 4)


def safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--name", required=True, help="short source id, used as file prefix and in the manifest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="sam2.1_b.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    args = ap.parse_args(argv)

    names = source_names(args.src)
    mapping = ClassMapping.from_yaml(args.mapping)
    missing = [n for n in names if n not in mapping.target]
    if missing:
        raise SystemExit(f"mapping {args.mapping} has no entry for source classes {missing}")
    segment = ultralytics_sam_segmenter(args.weights, args.device)

    images = list(iter_images(args.src))[: args.limit or None]
    rows, n_skip_empty = [], 0
    for k, img_path in enumerate(images, 1):
        stem = f"{args.name}_{safe_stem(img_path.stem)}"
        out_lbl = args.out / "labels" / f"{stem}.txt"
        out_img = args.out / "images" / f"{stem}{img_path.suffix.lower()}"
        if out_lbl.is_file():
            continue
        ids, yolo = read_boxes(label_path_for(img_path))
        with Image.open(img_path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = rgb.shape[:2]
        insts = convert_boxes(rgb, yolo_boxes_to_xyxy(yolo, w, h), [names[i] for i in ids], mapping, segment) if ids else []
        if not insts:
            n_skip_empty += 1
            continue
        out_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, out_img)
        write_labels(out_lbl, insts)
        rows.append([out_img.name, args.name, str(img_path.relative_to(args.src)), w, h, len(insts), *class_counts(insts)])
        if k % 100 == 0:
            print(f"  {k}/{len(images)}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"manifest_{args.name}.csv"
    new = not manifest.is_file()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        if new:
            wr.writerow(["file", "source", "source_path", "width", "height", "n_instances"] + [f"n_{c.replace(' ', '_')}" for c in CLASSES])
        wr.writerows(rows)
    totals = [sum(r[6 + c] for r in rows) for c in range(NC)]
    print(f"wrote {len(rows)} images, skipped {n_skip_empty} without usable objects; manifest {manifest}")
    for c, name in enumerate(CLASSES):
        print(f"  {c:>2} {name:<14} {totals[c]:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
