from doctor.diagnostics import DoctorData
from core.indices import IDX_HIT_COUNT

def get_value(data: DoctorData) -> float:
    return float(data.hit_count)

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_HIT_COUNT])

CONFIG = {
    "label":    "Disk Hit Count",
    "colormap": {
        0: (0.05, 0.05, 0.05),  # no disk crossing — near black
        1: (0.20, 0.40, 1.00),  # primary image — blue
        2: (0.20, 1.00, 0.40),  # secondary image — green
        3: (1.00, 0.80, 0.10),  # tertiary image — amber
        4: (1.00, 0.10, 0.10),  # quaternary+ — red
    },
}