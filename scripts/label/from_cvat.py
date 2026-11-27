"""Read a CVAT export ("Ultralytics YOLO Segmentation 1.0") into a YOLO-seg folder with competition class ids.

    python scripts/label/from_cvat.py --zip golden_val_export.zip --out data/interim/golden_val
    python scripts/label/from_cvat.py --zip export_without_images.zip --out ... --images-dir <folder with the photos>

Class ids are re-derived from the NAMES in the export's data.yaml (CVAT numbers labels in creation order), so a
project whose labels were created in another order is still imported correctly; an unknown label name stops the
import. Run scripts/qa/validate_labels.py on --out afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.classes import CLASSES  # noqa: E402
from robotaxi.cvat import import_cvat_export  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    rep = import_cvat_export(args.zip, args.out, args.images_dir)
    print(f"imported {rep.images} images, {rep.instances} instances -> {args.out}")
    if rep.remapped:
        print("class ids remapped by name (export id -> competition id): "
              + ", ".join(f"{k}->{v} ({CLASSES[v]})" for k, v in sorted(rep.remapped.items())))
    if rep.missing_images:
        print(f"WARNING: {len(rep.missing_images)} label files have no image (pass --images-dir): {rep.missing_images[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
