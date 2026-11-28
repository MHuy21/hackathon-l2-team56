import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from robotaxi.dataset_build import Recipe, SourceSpec, build, sha256_file, summary, write_manifest, write_zip

LINE = {0: "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4", 8: "8 0.5 0.5 0.6 0.5 0.6 0.9 0.5 0.9"}


def make_source(root, name, n, classes=(0, 8), seed=0, stems=None, label=True):
    rng = np.random.default_rng(seed)
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    stems = stems or [f"img{k:03d}" for k in range(n)]
    for stem in stems:
        arr = (rng.random((48, 64, 3)) * 255).astype(np.uint8)  # random noise: no near-duplicates by accident
        Image.fromarray(arr).save(root / name / "images" / f"{stem}.jpg", quality=95)
        if label:
            (root / name / "labels" / f"{stem}.txt").write_text("\n".join(LINE[c] for c in classes) + "\n", encoding="utf-8")
    return root / name


def test_auto_split_counts_and_prefix(tmp_path):
    make_source(tmp_path, "coco", 10, seed=1, stems=[f"coco_{k:012d}" for k in range(10)])
    make_source(tmp_path, "vn", 20, seed=2)
    r = Recipe("t", [SourceSpec("coco2017", tmp_path / "coco", "train", prefix="coco"),
                     SourceSpec("vn", tmp_path / "vn", "auto", val_ratio=0.25)], min_val_per_class=1)
    rep = build(r)
    assert rep.ok, rep.errors
    assert sum(it.split == "val" for it in rep.items) == 5
    assert all(it.source == "vn" for it in rep.items if it.split == "val")
    stems = {it.out_stem for it in rep.items}
    assert "coco_000000000003" in stems and "vn_img007" in stems
    assert rep.counts("val")[8] == 5


def test_group_regex_keeps_video_frames_together(tmp_path):
    stems = [f"clip{c}_f{f}" for c in range(6) for f in range(4)]
    make_source(tmp_path, "vn", 0, stems=stems)
    r = Recipe("t", [SourceSpec("vn", tmp_path / "vn", "auto", val_ratio=0.5, group_regex=r"^(.*)_f\d+$")], min_val_per_class=1)
    rep = build(r)
    split_of_clip = {}
    for it in rep.items:
        split_of_clip.setdefault(it.group, set()).add(it.split)
    assert all(len(s) == 1 for s in split_of_clip.values())


def test_missing_label_is_an_error(tmp_path):
    make_source(tmp_path, "vn", 3, label=False)
    rep = build(Recipe("t", [SourceSpec("vn", tmp_path / "vn", "val")]))
    assert not rep.ok and any("no label file" in e for e in rep.errors)


def test_bad_label_is_an_error(tmp_path):
    src = make_source(tmp_path, "vn", 2)
    (src / "labels" / "img000.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    rep = build(Recipe("t", [SourceSpec("vn", src, "val")]))
    assert any("BOX" in e for e in rep.errors)


def test_exact_and_near_duplicates_removed_from_train(tmp_path):
    val = make_source(tmp_path, "val", 3, seed=5)
    tr = make_source(tmp_path, "tr", 3, seed=6)
    img = Image.open(val / "images" / "img000.jpg")
    img.save(tr / "images" / "img000.jpg", quality=95)                    # (almost) same picture as a val image
    (tr / "images" / "img001.jpg").write_bytes((val / "images" / "img001.jpg").read_bytes())  # exact copy
    rep = build(Recipe("t", [SourceSpec("val", val, "val"), SourceSpec("tr", tr, "train")], min_val_per_class=1))
    kept_train = sorted(it.out_stem for it in rep.items if it.split == "train")
    assert kept_train == ["tr_img002"]
    assert len(rep.dropped) == 2


def test_low_val_class_warning(tmp_path):
    make_source(tmp_path, "vn", 4, classes=(0,))
    rep = build(Recipe("t", [SourceSpec("vn", tmp_path / "vn", "val"), SourceSpec("vn2", make_source(tmp_path, "vn2", 2, seed=3), "train")]))
    assert any("Car (4)" in w for w in rep.warnings)


def test_zip_is_deterministic_and_notebook_layout(tmp_path):
    make_source(tmp_path, "vn", 6, seed=9)
    r = Recipe("t", [SourceSpec("vn", tmp_path / "vn", "auto", val_ratio=0.5)], min_val_per_class=1)
    sha1 = write_zip(build(r), tmp_path / "a.zip")
    sha2 = write_zip(build(r), tmp_path / "b.zip")
    assert sha1 == sha2 == sha256_file(tmp_path / "a.zip")
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert {n.split("/")[0] for n in names} == {"images", "labels"}
    assert {n.split("/")[1] for n in names} == {"train", "val"}
    rep = build(r)
    write_manifest(rep, tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text(encoding="utf-8").count("\n") == 7
    assert json.dumps(summary(rep, r, sha1))


def test_recipe_from_yaml(tmp_path):
    (tmp_path / "r.yaml").write_text("name: x\nsources:\n  - {name: a, path: d/a, role: train}\n", encoding="utf-8")
    r = Recipe.from_yaml(tmp_path / "r.yaml", base=tmp_path)
    assert r.seed == 61 and r.sources[0].path == tmp_path / "d/a"
    with pytest.raises(ValueError):
        SourceSpec("a", tmp_path, "test")


def test_exclude_list_and_frame_window_with_b_csv_format(tmp_path):
    stems = [f"cam_01_{f:05d}" for f in (30, 50, 57, 65, 90)] + ["cam_02_00057", "cam_02_00200"]
    src = make_source(tmp_path, "kg", 0, stems=stems)
    excl = tmp_path / "exclude_from_train_same_frames.csv"
    excl.write_text("\ufeffselected_filename,exclude_source_relative_path,reason\n"
                    "cam_01_00057.jpg,daytime-dataset/daytime/cam_01_00057.jpg,selected_original_frame\n"
                    "cam_02_00200.jpg,daytime-dataset/daytime/cam_02_00200.jpg,selected_original_frame\n", encoding="utf-8")
    spec = SourceSpec("kg", src, "train", prefix="kgvn", exclude_list=excl, exclude_window=10)
    rep = build(Recipe("t", [spec, SourceSpec("val", make_source(tmp_path, "v", 2, seed=4), "val")], min_val_per_class=1))
    kept = sorted(it.out_stem for it in rep.items if it.source == "kg")
    # cam_01: 57 listed, 50 and 65 within 10 frames; 30 and 90 kept. cam_02: 200 listed; 57 kept (other camera window)
    assert kept == ["kgvn_cam_01_00030", "kgvn_cam_01_00090", "kgvn_cam_02_00057"]
    reasons = dict(rep.dropped)
    assert "listed" in reasons["kgvn_cam_01_00057"] and "within 10 frames" in reasons["kgvn_cam_01_00050"]


def test_missing_exclude_list_is_an_error(tmp_path):
    src = make_source(tmp_path, "kg", 2)
    rep = build(Recipe("t", [SourceSpec("kg", src, "val", exclude_list=tmp_path / "nope.csv")]))
    assert any("exclude list not found" in e for e in rep.errors)
