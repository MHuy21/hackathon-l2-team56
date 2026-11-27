import numpy as np
import pytest

from robotaxi.classes import CAR, MOTORCYCLE, MOTORCYCLIST, PEDESTRIAN, TRUCK
from robotaxi.sources.bbox_sam import ClassMapping, clip_to_box, convert_boxes, yolo_boxes_to_xyxy

H, W = 480, 640
IMG = np.zeros((H, W, 3), np.uint8)


def box_filler(image, boxes):
    """Fake SAM: the mask is the prompt box itself, plus a leak outside it that must be clipped."""
    out = []
    for x0, y0, x1, y1 in boxes:
        m = np.zeros(image.shape[:2], bool)
        m[int(y0):int(y1), int(x0):int(x1)] = True
        m[0:5, 0:5] = True  # leak far away
        out.append(m)
    return out


MAPPING = ClassMapping.from_dict({"car": "Car", "truck": "Truck", "motorbike": "motorcycle", "person": "person",
                                  "motorbike_rider": "Motorcyclist", "bus_stop": None})


def test_mapping_rejects_typos_and_unknown_names():
    with pytest.raises(KeyError):
        ClassMapping.from_dict({"car": "Carr"})
    with pytest.raises(KeyError):
        MAPPING.resolve("tank")
    assert MAPPING.resolve("bus_stop") is None and MAPPING.resolve("motorbike") == "motorcycle"


def test_yolo_boxes_to_xyxy():
    xyxy = yolo_boxes_to_xyxy(np.array([[0.5, 0.5, 0.5, 0.25]]), W, H)
    np.testing.assert_allclose(xyxy, [[160, 180, 480, 300]])


def test_clip_to_box_removes_leak():
    m = np.ones((H, W), bool)
    c = clip_to_box(m, np.array([100, 100, 200, 200]), pad=0)
    assert c.sum() == 100 * 100


def test_convert_boxes_direct_generic_and_dropped():
    boxes = np.array([
        [20, 200, 260, 330],   # car
        [300, 150, 450, 330],  # truck
        [470, 300, 560, 400],  # motorbike
        [495, 220, 535, 385],  # person on it -> Motorcyclist
        [580, 250, 620, 420],  # standing person -> Pedestrian
        [100, 400, 200, 470],  # motorbike, nobody -> Motorcycle
        [10, 10, 60, 60],      # bus_stop -> dropped
    ], dtype=float)
    names = ["car", "truck", "motorbike", "person", "person", "motorbike", "bus_stop"]
    insts = convert_boxes(IMG, boxes, names, MAPPING, box_filler)
    assert sorted(i.cls for i in insts) == sorted([CAR, TRUCK, MOTORCYCLIST, PEDESTRIAN, MOTORCYCLE])
    for inst in insts:  # the fake leak at the top-left corner never survives
        assert inst.poly.min(axis=0).min() > 5 / W


def test_nothing_mapped_returns_empty():
    assert convert_boxes(IMG, np.array([[0, 0, 50, 50]], float), ["bus_stop"], MAPPING, box_filler) == []
