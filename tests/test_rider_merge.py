import cv2
import numpy as np

from robotaxi.classes import BICYCLE, CYCLIST, MOTORCYCLE, MOTORCYCLIST, PEDESTRIAN
from robotaxi.rider_merge import RiderMergeConfig, merge_riders, rider_score

H, W = 540, 960


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (x0, y0), (x1 - 1, y1 - 1), 1, -1)
    return m.astype(bool)


def person(x0, y0, x1, y1):
    """Body box: torso top half, legs bottom half (a filled rectangle is enough for the geometry)."""
    return rect(x0, y0, x1, y1)


MOTO = rect(200, 300, 320, 400)          # vehicle box 120 x 100, ground line y = 400
RIDER = person(235, 220, 285, 380)       # head above the bike, legs inside, feet above the ground line
PASSENGER = person(260, 230, 305, 385)
WALKER = person(330, 250, 370, 410)      # standing next to the bike, pushing it
NEAR_PED = person(240, 330, 290, 520)    # pedestrian in front of the bike (closer to camera, feet much lower)


def classes(out):
    return sorted(c for c, _ in out)


def test_rider_becomes_one_motorcyclist_instance():
    out = merge_riders([RIDER], [MOTO])
    assert classes(out) == [MOTORCYCLIST]
    union = out[0][1]
    assert union[RIDER].all() and union[MOTO].all()


def test_two_people_on_one_motorcycle():
    out = merge_riders([RIDER, PASSENGER], [MOTO])
    assert classes(out) == [MOTORCYCLIST]


def test_person_walking_bike_is_pedestrian_plus_motorcycle():
    out = merge_riders([WALKER], [MOTO])
    assert classes(out) == [MOTORCYCLE, PEDESTRIAN]


def test_pedestrian_in_front_of_parked_bike_is_not_merged():
    assert rider_score(NEAR_PED, MOTO) == 0.0
    assert classes(merge_riders([NEAR_PED], [MOTO])) == [MOTORCYCLE, PEDESTRIAN]


def test_cyclist_and_free_bicycle():
    bike = rect(500, 320, 600, 400)
    cyclist = person(525, 230, 570, 385)
    parked = rect(700, 330, 790, 400)
    out = merge_riders([cyclist], bicycles=[bike, parked])
    assert classes(out) == [BICYCLE, CYCLIST]


def test_person_assigned_to_best_vehicle_only():
    moto2 = rect(260, 300, 380, 400)  # overlaps MOTO; rider centre is closer to MOTO
    out = merge_riders([RIDER], [MOTO, moto2])
    assert classes(out) == [MOTORCYCLE, MOTORCYCLIST]


def test_no_vehicles_all_pedestrians():
    assert classes(merge_riders([RIDER, WALKER])) == [PEDESTRIAN, PEDESTRIAN]


def test_passenger_seen_through_car_window_is_dropped():
    car = rect(100, 200, 400, 360)
    passenger = person(180, 215, 230, 290)       # head + torso in the window, no feet on the road
    ped_front = person(250, 230, 290, 380)       # pedestrian in front of the car: feet below the car bottom
    ped_behind = person(320, 150, 350, 250)      # standing behind: head above the roof
    from robotaxi.rider_merge import drop_passengers, inside_vehicle
    assert inside_vehicle(passenger, car)
    assert drop_passengers([passenger, ped_front, ped_behind], [car]) == [1, 2]


def test_decisions_override_geometry():
    from robotaxi.rider_merge import rider_candidates
    assert rider_candidates([RIDER, WALKER], [MOTO]) == {0: 0}
    assert classes(merge_riders([RIDER], [MOTO], decisions=[False])) == [MOTORCYCLE, PEDESTRIAN]
    assert classes(merge_riders([RIDER], [MOTO], decisions=[True])) == [MOTORCYCLIST]
    assert classes(merge_riders([RIDER], [MOTO], decisions=[None])) == [MOTORCYCLIST]
    assert classes(merge_riders([WALKER], [MOTO], decisions=[True])) == [MOTORCYCLE, PEDESTRIAN]  # not a candidate


def test_thresholds_are_configurable():
    strict = RiderMergeConfig(min_lower_inside=1.01)
    assert classes(merge_riders([RIDER], [MOTO], cfg=strict)) == [MOTORCYCLE, PEDESTRIAN]
