"""Convert a seeded traffic subset of COCO 2017 to the competition's 11-class YOLO-seg format.

Runs the same on Colab / Kaggle / a laptop. Only the selected images are downloaded (not the 18 GB of train2017).

    # annotations already extracted
    python scripts/data/coco_to_yoloseg.py --ann data/raw/coco/annotations/instances_train2017.json \
        --ann data/raw/coco/annotations/instances_val2017.json --out data/interim/coco --max-images 3500

    # let the script fetch annotations_trainval2017.zip (241 MB) into --raw-dir first
    python scripts/data/coco_to_yoloseg.py --download-ann --out data/interim/coco --max-images 3500

    # selection statistics only, nothing downloaded or written
    python scripts/data/coco_to_yoloseg.py --ann ... --max-images 3500 --dry-run

Output: <out>/images/coco_<id>.jpg, <out>/labels/coco_<id>.txt, <out>/manifest_coco.csv (one row per image with
source URL, Flickr license and per-class counts: what the report needs for the public-dataset table).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from robotaxi.classes import CLASSES, NC, TEAM_ID  # noqa: E402
from robotaxi.sources.coco import RIDER_POLICIES, CocoIndex, class_counts, convert_image, select_images  # noqa: E402
from robotaxi.yoloseg_io import write_labels  # noqa: E402

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_MEMBERS = ("annotations/instances_train2017.json", "annotations/instances_val2017.json")
USER_AGENT = "dataset-prep-script/1.0"


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return dest
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return dest


def download_annotations(raw_dir: Path) -> list[Path]:
    paths = [raw_dir / m for m in ANN_MEMBERS]
    if all(p.is_file() for p in paths):
        return paths
    zpath = raw_dir / "annotations_trainval2017.zip"
    if not zpath.is_file():
        print(f"downloading {ANN_URL} -> {zpath}")
        fetch(ANN_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        for m in ANN_MEMBERS:
            zf.extract(m, raw_dir)
    return paths


def load_index(paths: list[Path]) -> tuple[CocoIndex, dict[int, str]]:
    index, split_of = None, {}
    for p in paths:
        print(f"loading {p} ...")
        with open(p, "r", encoding="utf-8") as fh:
            part = CocoIndex.from_json(json.load(fh))
        split = p.stem.replace("instances_", "")
        split_of.update({i: split for i in part.images})
        index = part if index is None else index.merge(part)
    return index, split_of


def find_local(image: dict, dirs: list[Path]) -> Path | None:
    for d in dirs:
        p = d / image["file_name"]
        if p.is_file():
            return p
    return None


def load_decisions(path: Path | None) -> dict[int, bool]:
    """review_riders.csv filled in by a reviewer: column ``decision`` = rider / not_rider (blank rows ignored)."""
    if path is None:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = (r.get("decision") or "").strip().lower()
            if d in ("rider", "not_rider"):
                out[int(r["person_ann_id"])] = d == "rider"
            elif d:
                raise SystemExit(f"{path}: decision must be 'rider' or 'not_rider', got {d!r} (person {r['person_ann_id']})")
    return out


def process(image_id: int, index: CocoIndex, out: Path, local_dirs: list[Path], policy: str, decisions: dict[int, bool]) -> dict:
    im = index.images[image_id]
    name = f"coco_{image_id:012d}"
    res = convert_image(index.anns_by_image.get(image_id, []), index.cat_name, im["height"], im["width"], policy, decisions)
    if res.instances is None:
        return {"status": "ambiguous", "coco_id": image_id, "ambiguous": res.ambiguous, "url": im.get("coco_url", "")}
    if not res.instances:
        return {"status": "tiny", "coco_id": image_id}  # only tiny objects: not a clean background either, skip
    insts = res.instances
    img_path = out / "images" / f"{name}.jpg"
    if not img_path.is_file():
        src = find_local(im, local_dirs)
        if src is not None:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(src.read_bytes())
        else:
            fetch(im["coco_url"], img_path)
    write_labels(out / "labels" / f"{name}.txt", insts)
    lic = index.licenses.get(im.get("license"), {})
    return {"status": "ok", "file": img_path.name, "coco_id": image_id, "license_id": im.get("license", ""), "license": lic.get("name", ""),
            "url": im.get("coco_url", ""), "flickr_url": im.get("flickr_url", ""), "width": im["width"], "height": im["height"],
            "n_instances": len(insts), "counts": class_counts(insts)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", action="append", type=Path, default=[], help="instances_*.json (repeatable)")
    ap.add_argument("--download-ann", action="store_true", help=f"fetch annotations from {ANN_URL}")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/coco"))
    ap.add_argument("--images-dir", action="append", type=Path, default=[], help="local train2017/val2017 folders (optional)")
    ap.add_argument("--out", type=Path, default=Path("data/interim/coco"))
    ap.add_argument("--max-images", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=TEAM_ID)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rider-policy", choices=RIDER_POLICIES, default="skip",
                    help="person that might sit on a two-wheeler and has no decision: skip the image (default) or merge")
    ap.add_argument("--rider-decisions", type=Path, default=None, help="review_riders.csv with a filled 'decision' column")
    ap.add_argument("--dry-run", action="store_true", help="print the selection statistics and stop")
    args = ap.parse_args(argv)

    ann_paths = args.ann or (download_annotations(args.raw_dir) if args.download_ann else [])
    if not ann_paths:
        ap.error("give --ann paths or --download-ann")
    index, split_of = load_index(ann_paths)
    picked = select_images(index, args.max_images, args.seed)
    cat_counts = Counter(index.cat_name[a["category_id"]] for i in picked for a in index.anns_by_image[i])
    print(f"selected {len(picked)} images (seed {args.seed}); COCO annotations in them: {dict(cat_counts.most_common())}")
    if args.dry_run:
        return 0

    decisions = load_decisions(args.rider_decisions)
    rows, failed, ambiguous, n_tiny = [], [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, i, index, args.out, args.images_dir, args.rider_policy, decisions): i for i in picked}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad download must not stop the run
                failed.append((futs[fut], repr(exc)))
                continue
            if row["status"] == "ok":
                row["coco_split"] = split_of[row["coco_id"]]
                rows.append(row)
            elif row["status"] == "ambiguous":
                ambiguous.append(row)
            else:
                n_tiny += 1
            if n % 250 == 0:
                print(f"  {n}/{len(picked)}")
    rows.sort(key=lambda r: r["coco_id"])

    # A rerun (other policy / decisions) must not leave images of skipped ids behind: without a label file the
    # official notebook would treat them as BACKGROUND images, i.e. teach the model that their objects are not there.
    written = {r["file"] for r in rows}
    stale = [p for p in (args.out / "images").glob("coco_*.jpg") if p.name not in written]
    for p in stale:
        p.unlink()
        (args.out / "labels" / f"{p.stem}.txt").unlink(missing_ok=True)
    if stale:
        print(f"removed {len(stale)} stale images from a previous run")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest_coco.csv"
    cols = ["file", "source", "coco_id", "coco_split", "license_id", "license", "url", "flickr_url", "width", "height", "n_instances"]
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols + [f"n_{c.replace(' ', '_')}" for c in CLASSES])
        for r in rows:
            w.writerow([r["file"], "coco2017", *(r[c] for c in cols[2:])] + r["counts"])

    review = args.out / "review_riders.csv"
    with open(review, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["coco_id", "person_ann_id", "vehicle", "url", "decision"])
        for r in sorted(ambiguous, key=lambda r: r["coco_id"]):
            for pid, veh in r["ambiguous"]:
                w.writerow([r["coco_id"], pid, veh, r["url"], ""])

    totals = [sum(r["counts"][k] for r in rows) for k in range(NC)]
    print(f"\nwrote {len(rows)} images | skipped: {len(ambiguous)} with unresolved rider candidates, {n_tiny} only tiny objects"
          f" | {len(failed)} failed")
    print(f"manifest: {manifest}")
    print(f"rider review list: {review} (fill 'decision' = rider / not_rider, rerun with --rider-decisions)")
    for k, name in enumerate(CLASSES):
        print(f"  {k:>2} {name:<14} {totals[k]:>7}")
    for i, err in failed[:10]:
        print(f"  FAILED coco_id={i}: {err}")
    return 1 if failed and len(failed) > 0.05 * len(picked) else 0


if __name__ == "__main__":
    sys.exit(main())
