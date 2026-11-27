"""Pack a YOLO-seg folder (e.g. autolabel.py output) into a zip for CVAT ("Ultralytics YOLO Segmentation 1.0").

    python scripts/label/to_cvat.py --src /content/drive/MyDrive/team56/drafts/golden_val_150 \
        --zip /content/drive/MyDrive/team56/cvat/golden_val_150.zip

In CVAT: Project (11 labels, competition order) -> Import dataset -> format "Ultralytics YOLO Segmentation 1.0" ->
this zip. Or create a task with the images, then Actions -> Upload annotations with the same zip and format.
A draft instance made of several pieces is ONE polygon with thin "bridges" between the pieces: keep it as one shape
(redrawing it as separate polygons would export them as separate instances).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.cvat import pack_for_cvat  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--only", type=Path, default=None, help="text file with one image stem per line to include")
    args = ap.parse_args(argv)
    only = [s.strip() for s in args.only.read_text(encoding="utf-8").splitlines() if s.strip()] if args.only else None
    n = pack_for_cvat(args.src, args.zip, only=only)
    print(f"packed {n} images -> {args.zip} ({args.zip.stat().st_size / 2 ** 20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
