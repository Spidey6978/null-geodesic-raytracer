from doctor.diagnostics import DoctorData
from core.indices import IDX_CAPTURED

def get_value(data: DoctorData) -> float:
    return float(data.captured)

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_CAPTURED])

CONFIG = {
    "label":    "Capture Map",
    "colormap": {
        0: (1.0, 1.0, 1.0),  # not captured — white
        1: (0.0, 0.0, 0.0),  # captured — black
    },
}