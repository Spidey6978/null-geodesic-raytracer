from doctor.diagnostics import DoctorData
from core.indices import IDX_TERM_REASON

CONFIG = {
    "label": "Termination Reason",
    "colormap": {
        0: (0.0, 0.0, 0.0), # Unknown / Ongoing
        1: (1.0, 0.0, 0.0), # Captured (Event Horizon)
        2: (1.0, 0.0, 1.0), # Math Explosion / NaN
        3: (0.0, 1.0, 0.0), # Escaped / Bounds
        4: (1.0, 1.0, 0.0), # Orphan / Deep Space Budget Exhausted
        5: (1.0, 0.5, 0.0), # Photon Sphere Trapped (Orange)
    },
}

def get_value(data: DoctorData) -> float:
    return float(data.termination_reason)

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_TERM_REASON])