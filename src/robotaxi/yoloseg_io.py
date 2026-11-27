"""Read / write YOLO-seg label files exactly the way the competition grader parses them.

Format: one instance per line, ``class x1 y1 x2 y2 x3 y3 ...`` with coordinates normalised to [0, 1] by the image
size (EXIF orientation applied). An image without a label file, or with an empty one, is a background image.
The checks mirror ``detscore.score.read_yolo_seg_labels`` (the host scorer) so a file accepted here is accepted there.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .classes import NC

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COORD_TOLERANCE = 1e-3  # same as the official notebook: slightly outside [0, 1] is clipped, further out is an error
DEFAULT_PRECISION = 6


class LabelError(ValueError):
    """A label line the grader would refuse (message names file and line)."""


@dataclass
class Instance:
    """One labelled object: class id and ONE polygon (P, 2) of normalised (x, y)."""

    cls: int
    poly: np.ndarray

    def __post_init__(self) -> None:
        self.cls = int(self.cls)
        self.poly = np.asarray(self.poly, dtype=np.float64).reshape(-1, 2)

    def to_line(self, precision: int = DEFAULT_PRECISION) -> str:
        coords = " ".join(f"{v:.{precision}f}" for v in np.clip(self.poly, 0.0, 1.0).reshape(-1))
        return f"{self.cls} {coords}"


def parse_line(line: str, nc: int = NC, where: str = "") -> Instance | None:
    """Parse one label line; ``None`` for a blank line. Raises :class:`LabelError` for anything the grader refuses."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) == 5:
        raise LabelError(f"{where}: 5 fields = a BOX label (class cx cy w h); the competition needs polygons")
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise LabelError(f"{where}: {len(parts)} fields, need 'class x1 y1 x2 y2 x3 y3 ...' (odd count, >= 7)")
    try:
        cls_f = float(parts[0])
        vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
    except ValueError as exc:
        raise LabelError(f"{where}: non numeric field") from exc
    if cls_f != int(cls_f) or not 0 <= int(cls_f) < nc:
        raise LabelError(f"{where}: class id {parts[0]} out of range 0..{nc - 1}")
    if not np.isfinite(vals).all():
        raise LabelError(f"{where}: non finite coordinate")
    if vals.min() < -COORD_TOLERANCE or vals.max() > 1 + COORD_TOLERANCE:
        raise LabelError(f"{where}: coordinate outside [0, 1] ({vals.min():.4f}..{vals.max():.4f}); labels must be normalised")
    return Instance(int(cls_f), np.clip(vals, 0.0, 1.0).reshape(-1, 2))


def read_labels(path: str | Path | None, nc: int = NC) -> list[Instance]:
    """All instances of a label file (missing file = background image = ``[]``)."""
    if path is None or not Path(path).is_file():
        return []
    out = []
    text = Path(path).read_text(encoding="utf-8")
    for n, line in enumerate(text.splitlines(), 1):
        inst = parse_line(line, nc, f"{path}:{n}")
        if inst is not None:
            out.append(inst)
    return out


def write_labels(path: str | Path, instances: list[Instance], precision: int = DEFAULT_PRECISION) -> Path:
    """Write instances (an empty list writes an empty file = background image). LF line endings, UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(inst.to_line(precision) + "\n" for inst in instances if len(inst.poly) >= 3)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return path


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) as the grader sees the image: EXIF orientation applied."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(274)
        except Exception:  # noqa: BLE001 - unreadable EXIF = no rotation
            orientation = None
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def label_path_for(image: str | Path) -> Path:
    """``.../images/<sub>/x.jpg`` -> ``.../labels/<sub>/x.txt`` (last ``images`` folder), else ``x.txt`` beside the image."""
    parts = list(Path(image).parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return Path(image).with_suffix(".txt")


def iter_images(root: str | Path) -> Iterator[Path]:
    """Images under ``root`` (sorted, hidden folders and __MACOSX skipped)."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).parts
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not any(s == "__MACOSX" or s.startswith(".") for s in rel):
            yield p


def iter_dataset(root: str | Path, nc: int = NC) -> Iterator[tuple[Path, list[Instance]]]:
    """``(image path, instances)`` for every image under ``root``."""
    for img in iter_images(root):
        yield img, read_labels(label_path_for(img), nc)
