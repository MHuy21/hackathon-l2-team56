"""Assemble a training round's dataset (``du_lieu_rXX.zip``) from converted sources, following a recipe.

Recipe (configs/datasets/rXX.yaml)::

    name: du_lieu_r00
    seed: 61                      # TEAM_ID
    sources:
      - name: coco2017
        path: data/interim/coco   # folder with images/ and labels/ (any depth)
        role: train               # train | val | auto
        max_images: null          # optional seeded subsample
      - name: vn_bbox
        path: data/interim/vntraffic
        role: auto                # split into train / val
        val_ratio: 0.2
        group_regex: '^(.*)_f\\d+$'   # optional: stems with the same group go to the same split (video frames)

Layout produced (what the official notebook reads as "already split"): images/{train,val}/, labels/{train,val}/.
Checks done here, before anything reaches the notebook:

* every label file parses with the grader's rules; an image WITHOUT a label file is an error (the notebook would
  silently take it as a background image);
* exact duplicate images (sha256) are kept once;
* train images whose dHash is within ``near_dup_hamming`` of a val image are dropped (train/val leakage);
* per-class val counts below ``min_val_per_class`` are reported (the notebook warns at 10).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .classes import CLASSES, MIN_GT_PER_CLASS_VAL, NC, TEAM_ID
from .yoloseg_io import iter_images, label_path_for, read_labels

ROLES = ("train", "val", "auto")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass
class SourceSpec:
    name: str
    path: Path
    role: str
    val_ratio: float = 0.2
    max_images: int | None = None
    group_regex: str | None = None
    prefix: str | None = None  # output file names start with "<prefix>_" (default: the source name)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"source {self.name}: role must be one of {ROLES}, got {self.role!r}")
        self.path = Path(self.path)
        self.prefix = self.prefix or self.name


@dataclass
class Recipe:
    name: str
    sources: list[SourceSpec]
    seed: int = TEAM_ID
    near_dup_hamming: int = 2
    min_val_per_class: int = MIN_GT_PER_CLASS_VAL

    @classmethod
    def from_yaml(cls, path: str | Path, base: str | Path = ".") -> "Recipe":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        srcs = [SourceSpec(**{**s, "path": Path(base) / s["path"]}) for s in data["sources"]]
        extra = {k: data[k] for k in ("seed", "near_dup_hamming", "min_val_per_class") if k in data}
        return cls(data["name"], srcs, **extra)


@dataclass
class Item:
    source: str
    image: Path
    label: Path
    out_stem: str
    group: str
    split: str = ""
    counts: list[int] = field(default_factory=lambda: [0] * NC)
    sha256: str = ""
    dhash: int = 0


@dataclass
class BuildReport:
    items: list[Item]
    errors: list[str]
    warnings: list[str]
    dropped: list[tuple[str, str]]  # (out_stem, reason)

    @property
    def ok(self) -> bool:
        return not self.errors

    def counts(self, split: str) -> list[int]:
        tot = [0] * NC
        for it in self.items:
            if it.split == split:
                tot = [a + b for a, b in zip(tot, it.counts)]
        return tot


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash64(path: Path) -> int:
    """64-bit difference hash (grey 9x8): near-identical frames differ by a few bits."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


def _group_of(stem: str, regex: str | None) -> str:
    if regex:
        m = re.match(regex, stem)
        if m and m.groups():
            return m.group(1)
    return stem


def collect(spec: SourceSpec, errors: list[str]) -> list[Item]:
    """Items of one source (labels validated, per-class counts filled)."""
    if not spec.path.is_dir():
        errors.append(f"{spec.name}: folder not found: {spec.path}")
        return []
    items = []
    for img in iter_images(spec.path):
        lab = label_path_for(img)
        if not lab.is_file():
            errors.append(f"{spec.name}: {img.relative_to(spec.path)} has no label file (would become a background image)")
            continue
        try:
            insts = read_labels(lab)
        except ValueError as exc:
            errors.append(f"{spec.name}: {exc}")
            continue
        stem = img.stem if img.stem.startswith(f"{spec.prefix}_") else f"{spec.prefix}_{img.stem}"
        c = Counter(i.cls for i in insts)
        items.append(Item(spec.name, img, lab, stem, _group_of(img.stem, spec.group_regex), counts=[c.get(k, 0) for k in range(NC)]))
    return items


def assign_splits(items: list[Item], spec: SourceSpec, seed: int) -> None:
    if spec.role != "auto":
        for it in items:
            it.split = spec.role
        return
    groups = sorted({it.group for it in items})
    random.Random(f"{seed}:{spec.name}").shuffle(groups)
    n_val = round(len(groups) * spec.val_ratio)
    val_groups = set(groups[:n_val])
    for it in items:
        it.split = "val" if it.group in val_groups else "train"


def build(recipe: Recipe) -> BuildReport:
    """Collect, split, dedupe and check; nothing is written."""
    errors: list[str] = []
    warnings: list[str] = []
    dropped: list[tuple[str, str]] = []
    items: list[Item] = []
    for spec in recipe.sources:
        got = collect(spec, errors)
        if spec.max_images and len(got) > spec.max_images:
            got = sorted(random.Random(f"{recipe.seed}:{spec.name}:sub").sample(got, spec.max_images), key=lambda it: it.out_stem)
        assign_splits(got, spec, recipe.seed)
        items += got

    names = Counter(it.out_stem for it in items)
    for stem, n in names.items():
        if n > 1:
            errors.append(f"file name {stem} comes from {n} images: rename one of the sources")

    for it in items:
        it.sha256 = sha256_file(it.image)
        it.dhash = dhash64(it.image)
    seen: dict[str, Item] = {}
    kept: list[Item] = []
    for it in sorted(items, key=lambda it: (it.split != "val", it.out_stem)):  # val copies win
        if it.sha256 in seen:
            dropped.append((it.out_stem, f"exact duplicate of {seen[it.sha256].out_stem}"))
            continue
        seen[it.sha256] = it
        kept.append(it)

    val = [it for it in kept if it.split == "val"]
    if val:
        vh = np.array([it.dhash for it in val], dtype=np.uint64)
        final = []
        for it in kept:
            if it.split == "train":
                dist = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(vh, np.uint64(it.dhash))])
                j = int(dist.argmin())
                if dist[j] <= recipe.near_dup_hamming:
                    dropped.append((it.out_stem, f"near-duplicate of val image {val[j].out_stem} (dHash distance {dist[j]})"))
                    continue
            final.append(it)
        kept = final
    else:
        errors.append("val split is empty: add a source with role val or auto")

    report = BuildReport(sorted(kept, key=lambda it: (it.split, it.out_stem)), errors, warnings, dropped)
    vc = report.counts("val")
    low = [f"{CLASSES[k]} ({vc[k]})" for k in range(NC) if vc[k] < recipe.min_val_per_class]
    if low:
        warnings.append(f"val has fewer than {recipe.min_val_per_class} instances for: {', '.join(low)}")
    if not any(it.split == "train" for it in kept):
        errors.append("train split is empty")
    return report


def write_zip(report: BuildReport, zip_path: Path) -> str:
    """Deterministic zip (sorted entries, fixed timestamps): same data -> same sha256. Returns the zip sha256."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w") as zf:
        for it in report.items:
            for path, folder, ext, comp in ((it.image, "images", it.image.suffix.lower(), zipfile.ZIP_STORED),
                                            (it.label, "labels", ".txt", zipfile.ZIP_DEFLATED)):
                info = zipfile.ZipInfo(f"{folder}/{it.split}/{it.out_stem}{ext}", date_time=ZIP_DATE)
                info.compress_type = comp
                info.external_attr = 0o644 << 16
                zf.writestr(info, path.read_bytes())
    tmp.replace(zip_path)
    return sha256_file(zip_path)


def write_manifest(report: BuildReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "split", "source", "group", "sha256"] + [f"n_{c.replace(' ', '_')}" for c in CLASSES])
        for it in report.items:
            w.writerow([f"{it.out_stem}{it.image.suffix.lower()}", it.split, it.source, it.group, it.sha256, *it.counts])


def summary(report: BuildReport, recipe: Recipe, zip_sha256: str | None = None) -> dict:
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    for it in report.items:
        per_source[it.source][it.split] += 1
    return {
        "name": recipe.name, "seed": recipe.seed, "zip_sha256": zip_sha256,
        "images": {s: sum(1 for it in report.items if it.split == s) for s in ("train", "val")},
        "instances": {s: dict(zip(CLASSES, report.counts(s))) for s in ("train", "val")},
        "per_source": dict(per_source), "dropped": len(report.dropped), "warnings": report.warnings,
    }


def dump_summary(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
