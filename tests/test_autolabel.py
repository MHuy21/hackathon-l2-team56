import cv2
import numpy as np

from robotaxi.autolabel import AutoLabelConfig, Detection, dedupe, draft_labels, review_priority
from robotaxi.classes import CAR, MOTORCYCLE, MOTORCYCLIST, OTHER_VEHICLE, PEDESTRIAN, SIGN, TRUCK

H, W = 480, 640
IMG = np.zeros((H, W, 3), np.uint8)


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (x0, y0), (x1 - 1, y1 - 1), 1, -1)
    return m.astype(bool)


def coco_detector(image):
    return [
        Detection("car", 0.9, rect(20, 200, 260, 330)),
        Detection("person", 0.8, rect(120, 215, 160, 270)),     # passenger in the car window -> dropped
        Detection("motorcycle", 0.7, rect(470, 300, 560, 400)),
        Detection("person", 0.85, rect(495, 220, 535, 385)),    # sits on the motorcycle -> one Motorcyclist
        Detection("person", 0.4, rect(580, 250, 620, 420)),     # pedestrian, low score -> flagged
        Detection("truck", 0.6, rect(300, 100, 440, 290)),
        Detection("person", 0.1, rect(10, 10, 60, 120)),        # below threshold -> ignored
        Detection("dog", 0.9, rect(200, 400, 260, 460)),        # unmapped COCO class -> ignored
    ]


def openvocab_detector(image):
    return [
        Detection("traffic sign", 0.5, rect(560, 60, 600, 100)),
        Detection("trailer", 0.45, rect(300, 120, 440, 290)),  # same object as the truck (IoU > 0.6) -> lower score dropped
        Detection("cyclo", 0.35, rect(60, 380, 180, 470)),
    ]


def test_draft_labels_end_to_end():
    res = draft_labels(IMG, [("coco", coco_detector), ("openvocab", openvocab_detector)])
    classes = sorted(i.cls for i in res.instances)
    assert classes == sorted([CAR, MOTORCYCLIST, PEDESTRIAN, TRUCK, SIGN, OTHER_VEHICLE])
    assert res.flags["passengers_dropped"] == 1
    assert res.flags["rider_merges"] == 1 and res.flags["riders"] == 1
    # kept detections scoring < 0.5: pedestrian 0.4 and cyclo 0.35 (trailer 0.45 was removed as a duplicate of the truck,
    # the 0.1 person is below the threshold, the sign scores exactly 0.5)
    assert res.flags["low_score"] == 2
    assert review_priority(res.flags) > 0


def test_thresholds_per_source():
    cfg = AutoLabelConfig(min_score={"coco": 0.95, "openvocab": 0.95}, min_score_by_name={}, big_object_min_score=0.95)
    assert draft_labels(IMG, [("coco", coco_detector), ("openvocab", openvocab_detector)], cfg).instances == []


def test_per_class_and_big_vehicle_thresholds():
    far_moto = Detection("motorcycle", 0.13, rect(600, 300, 615, 330))          # small far motorbike: kept (0.12)
    big_truck = Detection("truck", 0.11, rect(0, 300, 300, 480))               # big truck at the edge: kept (0.10)
    big_person = Detection("person", 0.11, rect(400, 100, 500, 400))           # big but a person: still needs 0.15
    small_car = Detection("car", 0.2, rect(300, 50, 330, 70))                  # small car below 0.25: dropped
    res = draft_labels(IMG, [("coco", lambda im: [far_moto, big_truck, big_person, small_car])])
    assert sorted(i.cls for i in res.instances) == sorted([MOTORCYCLE, TRUCK])


def test_dedupe_keeps_highest_score():
    a = Detection("truck", 0.6, rect(300, 100, 440, 290))
    b = Detection("trailer", 0.45, rect(300, 120, 440, 290))
    c = Detection("car", 0.5, rect(0, 0, 50, 50))
    assert [d.name for d in dedupe([b, a, c], 0.6)] == ["truck", "car"]


def test_sam_refinement_replaces_masks():
    def sam(image, boxes):
        out = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros(image.shape[:2], bool)
            m[int(y0) + 2:int(y1) - 2, int(x0) + 2:int(x1) - 2] = True
            out.append(m)
        return out

    res = draft_labels(IMG, [("coco", lambda im: [Detection("car", 0.9, rect(20, 200, 260, 330))])], segmenter=sam)
    assert len(res.instances) == 1
    xs = res.instances[0].poly[:, 0] * W
    assert xs.min() >= 21  # shrunk by the fake SAM
