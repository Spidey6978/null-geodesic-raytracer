# doctor/modes/ergosphere_map.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_CAPTURED, IDX_ENTERED_ERGO, IDX_STEPS_IN_ERGO


def get_value(data: DoctorData) -> float:
    return float(data.steps_in_ergosphere)


def get_value_from_raw(stats) -> float:
    captured = stats[IDX_CAPTURED]
    entered = stats[IDX_ENTERED_ERGO]
    steps_in_ergo = stats[IDX_STEPS_IN_ERGO]

    if captured:
        return 2.0
    if steps_in_ergo > 0.0:
        return float(steps_in_ergo)
    if entered:
        return 1.0
    return 0.0

CONFIG = {
    "label": "Ergosphere vs Shadow",
    "colormap": {
        0: (0.0, 0.0, 0.1),   # never entered ergosphere — dark blue
        1: (1.0, 0.8, 0.0),   # entered ergosphere, escaped — gold
        2: (0.8, 0.0, 0.0),   # captured — red
    },
}