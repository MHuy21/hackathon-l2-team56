from robotaxi.classes import CAR, MOTORCYCLE, MOTORCYCLIST, PEDESTRIAN, SIGN
from robotaxi.sources.coco import CocoIndex, convert_image, eligible, select_images

H, W = 480, 640
CATS = [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "car"}, {"id": 4, "name": "motorcycle"},
        {"id": 7, "name": "train"}, {"id": 13, "name": "stop sign"}]


def box(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def ann(i, img, cat, seg, crowd=0):
    return {"id": i, "image_id": img, "category_id": cat, "segmentation": seg, "iscrowd": crowd}


def coco():
    images = [{"id": k, "width": W, "height": H, "file_name": f"{k}.jpg", "coco_url": f"http://x/{k}.jpg", "license": 4} for k in range(1, 6)]
    anns = [
        # image 1: car, a rider on a motorcycle, a pedestrian, a passenger inside the car, a stop sign
        ann(1, 1, 3, box(50, 200, 300, 330)),
        ann(2, 1, 4, box(400, 300, 480, 400)),
        ann(3, 1, 1, box(420, 220, 460, 385)),
        ann(4, 1, 1, box(550, 200, 590, 420)),
        ann(5, 1, 1, box(120, 215, 160, 270)),
        ann(6, 1, 13, box(560, 60, 600, 100)),
        ann(7, 2, 1, box(10, 10, 100, 300)),                       # image 2: only persons -> not a road scene
        ann(8, 3, 3, box(10, 10, 200, 200)), ann(9, 3, 7, box(300, 10, 600, 200)),  # image 3: has a train -> excluded
        ann(10, 4, 3, box(10, 10, 200, 200)), ann(11, 4, 1, {"counts": [0, 10], "size": [H, W]}, crowd=1),  # crowd -> excluded
        ann(12, 5, 2, box(100, 100, 200, 200)),                    # image 5: a bicycle
    ]
    return CocoIndex.from_json({"images": images, "annotations": anns, "categories": CATS, "licenses": [{"id": 4, "name": "CC BY"}]})


def test_eligibility_rules():
    idx = coco()
    assert [i for i in idx.images if eligible(idx, i)] == [1, 5]


def test_selection_is_seeded_and_bounded():
    idx = coco()
    assert select_images(idx, 10, seed=61) == [1, 5]
    one = select_images(idx, 1, seed=61)
    assert len(one) == 1 and one == select_images(idx, 1, seed=61)


def test_rider_candidate_skips_image_by_default():
    idx = coco()
    res = convert_image(idx.anns_by_image[1], idx.cat_name, H, W)
    assert res.instances is None and res.ambiguous == [(3, "motorcycle")]


def test_convert_image_with_decision_or_merge_policy():
    idx = coco()
    expected = sorted([CAR, MOTORCYCLIST, PEDESTRIAN, SIGN])  # passenger in the car is gone, rider merged
    for res in (convert_image(idx.anns_by_image[1], idx.cat_name, H, W, decisions={3: True}),
                convert_image(idx.anns_by_image[1], idx.cat_name, H, W, rider_policy="merge")):
        assert sorted(i.cls for i in res.instances) == expected
    res = convert_image(idx.anns_by_image[1], idx.cat_name, H, W, decisions={3: False})
    assert sorted(i.cls for i in res.instances) == sorted([CAR, MOTORCYCLE, PEDESTRIAN, PEDESTRIAN, SIGN])


def test_tiny_objects_removed():
    idx = CocoIndex.from_json({"images": [{"id": 1, "width": W, "height": H}], "categories": CATS,
                               "annotations": [ann(1, 1, 3, box(10, 10, 15, 15))]})
    assert convert_image(idx.anns_by_image[1], idx.cat_name, H, W).instances == []
