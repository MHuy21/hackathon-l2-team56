"""Fixed class list of the competition (must match configs/classes.yaml)."""

CLASSES = (
    "Car",            # 0
    "Bus",            # 1
    "Truck",          # 2
    "Other vehicle",  # 3
    "Motorcycle",     # 4  motorcycle WITHOUT rider
    "Motorcyclist",   # 5  motorcycle + everyone on it, ONE instance
    "Bicycle",        # 6  bicycle WITHOUT rider
    "Cyclist",        # 7  bicycle + rider, ONE instance
    "Pedestrian",     # 8
    "Sign",           # 9  sign face only, no pole
    "Traffic light",  # 10 whole light housing
)
NC = len(CLASSES)
NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}

CAR, BUS, TRUCK, OTHER_VEHICLE, MOTORCYCLE, MOTORCYCLIST, BICYCLE, CYCLIST, PEDESTRIAN, SIGN, TRAFFIC_LIGHT = range(NC)

TEAM_ID = 61
IMGSZ = 640
MIN_GT_AREA_PX = 64.0          # at 640 letterbox scale; smaller GT is ignored by the scorer
MIN_GT_PER_CLASS_VAL = 10
MIN_SHORT_SIDE_RATIO = 0.0125  # guideline: short side of visible box < 1.25% of image long side -> do not label


def class_id(name: str) -> int:
    """Class id from its English name (case-insensitive)."""
    for i, n in enumerate(CLASSES):
        if n.lower() == name.strip().lower():
            return i
    raise KeyError(f"unknown class name: {name!r}")
