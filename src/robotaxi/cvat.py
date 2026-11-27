"""Hand-off with CVAT: pack drafts for import, read corrected exports back with a class-id check.

Format: CVAT "Ultralytics YOLO Segmentation 1.0" (import and export), i.e.

    data.yaml                 names: {0: Car, 1: Bus, ...}   (+ path / train)
    images/<subset>/*.jpg
    labels/<subset>/*.txt     class x1 y1 x2 y2 ... (normalised)

Annotation tools may renumber classes (CVAT numbers labels in the order they were created in the project), so an
export is never trusted by id: ``data.yaml`` names are matched to the competition classes BY NAME and every label line
is remapped. A name that is not a competition class, or a file without ``data.yaml``, is an error.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

from .classes import CLASSES, NAME_TO_ID
from .yoloseg_io import IMG_EXTS, Instance, iter_images, label_path_for, parse_line, write_labels

ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class CvatFormatError(ValueError):
    pass


def data_yaml_text(subset: str = "train") -> str:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    return f"path: .\n{subset}: images/{subset}\nnames:\n{names}\n"


def pack_for_cvat(src: str | Path, zip_path: str | Path, subset: str = "train", only: Iterable[str] | None = None) -> int:
    """Zip a YOLO-seg folder (images/ + labels/, e.g. autolabel.py output) for CVAT import. ``only``: image stems to
    include (e.g. the golden-val selection). Images without a label file get an empty one (CVAT then shows no
    shapes). Returns the number of images packed."""
    src, zip_path = Path(src), Path(zip_path)
    keep = set(only) if only is not None else None
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("data.yaml", date_time=ZIP_DATE), data_yaml_text(subset))
        names = []
        for img in iter_images(src):
            if keep is not None and img.stem not in keep:
                continue
            lab = label_path_for(img)
            info = zipfile.ZipInfo(f"images/{subset}/{img.name}", date_time=ZIP_DATE)
            zf.writestr(info, img.read_bytes())
            info = zipfile.ZipInfo(f"labels/{subset}/{img.stem}.txt", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, lab.read_bytes() if lab.is_file() else b"")
            names.append(f"images/{subset}/{img.name}")
            n += 1
        zf.writestr(zipfile.ZipInfo(f"{subset}.txt", date_time=ZIP_DATE), "".join(f"{x}\n" for x in names))
    if keep is not None and n < len(keep):
        missing = len(keep) - n
        raise CvatFormatError(f"{missing} requested image(s) not found in {src}")
    return n


@dataclass
class ImportReport:
    images: int = 0
    instances: int = 0
    remapped: dict[int, int] = field(default_factory=dict)  # export id -> competition id (only those that changed)
    missing_images: list[str] = field(default_factory=list)


def class_remap(names: dict | list) -> dict[int, int]:
    """Export class id -> competition class id, matched by name (case-insensitive)."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    lookup = {n.lower(): i for n, i in NAME_TO_ID.items()}
    out = {}
    for k, name in items:
        key = str(name).strip().lower()
        if key not in lookup:
            raise CvatFormatError(f"export class {k}: {name!r} is not one of the 11 competition classes {list(CLASSES)}")
        out[int(k)] = lookup[key]
    return out


def import_cvat_export(zip_path: str | Path, out_dir: str | Path, images_dir: str | Path | None = None) -> ImportReport:
    """Read a CVAT "Ultralytics YOLO Segmentation" export into ``out_dir``/images + labels with competition ids.

    Images come from the zip when the export includes them, otherwise from ``images_dir`` (matched by file stem).
    """
    out_dir = Path(out_dir)
    rep = ImportReport()
    with zipfile.ZipFile(zip_path) as zf:
        yamls = [n for n in zf.namelist() if PurePosixPath(n).name == "data.yaml"]
        if not yamls:
            raise CvatFormatError(f"{zip_path}: no data.yaml (export with format 'Ultralytics YOLO Segmentation 1.0')")
        remap = class_remap(yaml.safe_load(zf.read(yamls[0]).decode("utf-8"))["names"])
        rep.remapped = {k: v for k, v in remap.items() if k != v}
        zipped_images = {PurePosixPath(n).stem: n for n in zf.namelist() if PurePosixPath(n).suffix.lower() in IMG_EXTS}
        local_images = {p.stem: p for p in iter_images(images_dir)} if images_dir else {}
        for name in zf.namelist():
            pp = PurePosixPath(name)
            if pp.suffix != ".txt" or "labels" not in pp.parts:
                continue
            insts: list[Instance] = []
            for n, line in enumerate(zf.read(name).decode("utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if int(float(parts[0])) not in remap:
                    raise CvatFormatError(f"{name}:{n}: class id {parts[0]} is not listed in data.yaml")
                parts[0] = str(remap[int(float(parts[0]))])
                inst = parse_line(" ".join(parts), where=f"{name}:{n}")
                if inst is not None:
                    insts.append(inst)
            stem = pp.stem
            if stem in zipped_images:
                img_name = PurePosixPath(zipped_images[stem]).name
                (out_dir / "images").mkdir(parents=True, exist_ok=True)
                (out_dir / "images" / img_name).write_bytes(zf.read(zipped_images[stem]))
            elif stem in local_images:
                (out_dir / "images").mkdir(parents=True, exist_ok=True)
                (out_dir / "images" / local_images[stem].name).write_bytes(local_images[stem].read_bytes())
            else:
                rep.missing_images.append(stem)
                continue
            write_labels(out_dir / "labels" / f"{stem}.txt", insts)
            rep.images += 1
            rep.instances += len(insts)
    return rep

