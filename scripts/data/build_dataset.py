"""Build one round's dataset zip from a recipe (see src/robotaxi/dataset_build.py for the recipe format).

    python scripts/data/build_dataset.py configs/datasets/r00.yaml            # check + write zip + manifest
    python scripts/data/build_dataset.py configs/datasets/r00.yaml --check    # checks and counts only

Writes data/processed/<name>.zip (upload to Drive / Kaggle and point DATASET_PATH at it), and commits-friendly
manifests/<name>.csv + manifests/<name>.json (per-image source/split/sha256/counts, zip sha256). Paths in the recipe are
relative to --base (default: repo root), so the same recipe works on Colab after cloning the repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robotaxi.classes import CLASSES  # noqa: E402
from robotaxi.dataset_build import Recipe, build, dump_summary, summary, write_manifest, write_zip  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("--base", type=Path, default=ROOT, help="folder the recipe paths are relative to")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--manifest-dir", type=Path, default=ROOT / "manifests")
    ap.add_argument("--check", action="store_true", help="run the checks and print counts, write nothing")
    args = ap.parse_args(argv)

    recipe = Recipe.from_yaml(args.recipe, args.base)
    report = build(recipe)
    tr, va = report.counts("train"), report.counts("val")
    n_tr = sum(it.split == "train" for it in report.items)
    n_va = sum(it.split == "val" for it in report.items)
    print(f"{recipe.name}: {n_tr} train / {n_va} val images, {len(report.dropped)} dropped")
    print(f"  {'id':>2} {'class':<14} {'train':>7} {'val':>6}")
    for k, name in enumerate(CLASSES):
        print(f"  {k:>2} {name:<14} {tr[k]:>7} {va[k]:>6}")
    reasons: dict[str, int] = {}
    for _, why in report.dropped:
        key = why.split(" of ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for why, n in reasons.items():
        print(f"  dropped: {n} x {why}")
    for w in report.warnings:
        print(f"  WARNING: {w}")
    for e in report.errors[:30]:
        print(f"  ERROR: {e}")
    if report.errors:
        print(f"{len(report.errors)} error(s): nothing written. Fix the sources and rerun.")
        return 1
    if args.check:
        return 0

    zip_path = args.out_dir / f"{recipe.name}.zip"
    zip_sha = write_zip(report, zip_path)
    write_manifest(report, args.manifest_dir / f"{recipe.name}.csv")
    data = summary(report, recipe, zip_sha)
    data["recipe"] = str(args.recipe.as_posix())
    data["dropped_items"] = [{"file": f, "reason": r} for f, r in report.dropped]
    dump_summary(data, args.manifest_dir / f"{recipe.name}.json")
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 2 ** 20:.1f} MiB) sha256 {zip_sha}")
    print(f"manifest: {args.manifest_dir / (recipe.name + '.csv')} + .json")
    print(json.dumps(data["images"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
