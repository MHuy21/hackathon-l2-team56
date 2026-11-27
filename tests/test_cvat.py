import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image

from robotaxi.classes import CLASSES
from robotaxi.cvat import CvatFormatError, class_remap, import_cvat_export, pack_for_cvat
from robotaxi.yoloseg_io import read_labels

LINE_CAR = "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"
LINE_SIGN = "9 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6"


def drafts(tmp_path, n=3):
    (tmp_path / "d" / "images").mkdir(parents=True)
    (tmp_path / "d" / "labels").mkdir(parents=True)
    for k in range(n):
        Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(tmp_path / "d" / "images" / f"img{k}.jpg")
        (tmp_path / "d" / "labels" / f"img{k}.txt").write_text(f"{LINE_CAR}\n{LINE_SIGN}\n", encoding="utf-8")
    return tmp_path / "d"


def test_pack_layout_and_class_names(tmp_path):
    n = pack_for_cvat(drafts(tmp_path), tmp_path / "cvat.zip")
    assert n == 3
    zf = zipfile.ZipFile(tmp_path / "cvat.zip")
    names = zf.namelist()
    assert "data.yaml" in names and "images/train/img0.jpg" in names and "labels/train/img0.txt" in names
    data = yaml.safe_load(zf.read("data.yaml"))
    assert [data["names"][i] for i in range(11)] == list(CLASSES)


def test_pack_only_selected_and_missing_is_error(tmp_path):
    src = drafts(tmp_path)
    assert pack_for_cvat(src, tmp_path / "a.zip", only=["img1"]) == 1
    with pytest.raises(CvatFormatError):
        pack_for_cvat(src, tmp_path / "b.zip", only=["img1", "nope"])


def export_zip(path, names, lines, with_images=True):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.yaml", yaml.safe_dump({"names": names}))
        zf.writestr("labels/train/img0.txt", "\n".join(lines) + "\n")
        if with_images:
            buf = path.parent / "tmp.jpg"
            Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(buf)
            zf.writestr("images/train/img0.jpg", buf.read_bytes())


def test_import_remaps_renumbered_classes(tmp_path):
    # CVAT project where labels were created in another order: Sign got id 0, Car id 1
    names = {0: "Sign", 1: "Car"}
    export_zip(tmp_path / "e.zip", names, ["1 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4", "0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6"])
    rep = import_cvat_export(tmp_path / "e.zip", tmp_path / "out")
    assert rep.images == 1 and rep.instances == 2 and rep.remapped == {0: 9, 1: 0}
    assert [i.cls for i in read_labels(tmp_path / "out" / "labels" / "img0.txt")] == [0, 9]
    assert (tmp_path / "out" / "images" / "img0.jpg").is_file()


def test_import_rejects_unknown_class_and_missing_yaml(tmp_path):
    with pytest.raises(CvatFormatError, match="not one of"):
        class_remap({0: "Car", 1: "Tank"})
    with zipfile.ZipFile(tmp_path / "x.zip", "w") as zf:
        zf.writestr("labels/train/img0.txt", LINE_CAR)
    with pytest.raises(CvatFormatError, match="data.yaml"):
        import_cvat_export(tmp_path / "x.zip", tmp_path / "o")


def test_import_takes_images_from_local_folder(tmp_path):
    src = drafts(tmp_path, n=1)
    export_zip(tmp_path / "e.zip", {i: n for i, n in enumerate(CLASSES)}, [LINE_SIGN], with_images=False)
    rep = import_cvat_export(tmp_path / "e.zip", tmp_path / "out", images_dir=src / "images")
    assert rep.images == 1 and rep.remapped == {}
    assert (tmp_path / "out" / "images" / "img0.jpg").is_file()


def test_roundtrip_pack_then_import(tmp_path):
    pack_for_cvat(drafts(tmp_path), tmp_path / "cvat.zip")
    rep = import_cvat_export(tmp_path / "cvat.zip", tmp_path / "back")
    assert rep.images == 3 and rep.instances == 6
