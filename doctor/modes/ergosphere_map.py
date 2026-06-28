# doctor/modes/ergosphere_map.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_ENTERED_ERGO, IDX_CAPTURED

def get_value(data: DoctorData) -> float:
    return float(data.entered_ergosphere)

def get_value_from_raw(stats) -> float:
    captured = stats[IDX_CAPTURED]
    entered  = stats[IDX_ENTERED_ERGO]
    if captured:
        return 2.0
    elif entered:
        return 1.0
    else:
        return 0.0

CONFIG = {
    "label": "Ergosphere vs Shadow",
    "colormap": {
        0: (0.0, 0.0, 0.1),   # never entered ergosphere — dark blue
        1: (1.0, 0.8, 0.0),   # entered ergosphere, escaped — gold
        2: (0.8, 0.0, 0.0),   # captured — red
    },
}