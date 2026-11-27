from pathlib import Path

import numpy as np
import pytest

from robotaxi.classes import CLASSES, NC, class_id
from robotaxi.yoloseg_io import Instance, LabelError, iter_dataset, label_path_for, parse_line, read_labels, write_labels

SAMPLE = Path(__file__).resolve().parents[2] / "hackathon-l2-docs" / "data" / "sample_data"


def test_classes_fixed_order():
    assert NC == 11
    assert CLASSES[5] == "Motorcyclist" and CLASSES[10] == "Traffic light"
    assert class_id("traffic light") == 10


def test_roundtrip(tmp_path):
    insts = [Instance(5, [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]), Instance(9, [[0.5, 0.5], [0.6, 0.5], [0.6, 0.7], [0.5, 0.7]])]
    p = write_labels(tmp_path / "labels" / "a.txt", insts)
    back = read_labels(p)
    assert [i.cls for i in back] == [5, 9]
    np.testing.assert_allclose(back[1].poly, insts[1].poly, atol=1e-6)
    assert b"\r\n" not in p.read_bytes()


def test_missing_and_empty_file_is_background(tmp_path):
    assert read_labels(tmp_path / "nope.txt") == []
    (tmp_path / "e.txt").write_text("\n\n", encoding="utf-8")
    assert read_labels(tmp_path / "e.txt") == []


@pytest.mark.parametrize("line, msg", [
    ("0 0.5 0.5 0.1 0.1", "BOX"),
    ("0 0.1 0.1 0.2 0.2", "fields"),
    ("11 0.1 0.1 0.2 0.2 0.3 0.3", "out of range"),
    ("0 0.1 0.1 0.2 0.2 0.3 1.5", "outside"),
    ("x 0.1 0.1 0.2 0.2 0.3 0.3", "non numeric"),
])
def test_rejects_what_grader_rejects(line, msg):
    with pytest.raises(LabelError, match=msg):
        parse_line(line, where="f:1")


def test_slightly_outside_is_clipped():
    inst = parse_line("0 -0.0005 0.1 0.2 0.2 1.0004 0.3")
    assert inst.poly.min() >= 0 and inst.poly.max() <= 1


def test_label_path_for():
    assert label_path_for(Path("d/images/train/x.jpg")) == Path("d/labels/train/x.txt")
    assert label_path_for(Path("d/x.png")) == Path("d/x.txt")


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="hackathon-l2-docs sample data not found")
def test_reads_official_sample_data():
    items = list(iter_dataset(SAMPLE))
    assert len(items) == 10
    assert sum(len(insts) for _, insts in items) == 105
    assert {i.cls for _, insts in items for i in insts} == set(range(NC))
