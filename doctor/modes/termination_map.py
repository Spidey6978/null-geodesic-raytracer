from doctor.diagnostics import DoctorData
from core.indices import IDX_TERM_REASON

def get_value(data: DoctorData) -> float:
    return float(data.termination_reason)

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_TERM_REASON])

CONFIG = {
    "label":    "Termination Reason",
    "colormap": {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 0.0, 1.0),
        3: (0.0, 1.0, 0.0),
        4: (1.0, 1.0, 0.0),
    },
}