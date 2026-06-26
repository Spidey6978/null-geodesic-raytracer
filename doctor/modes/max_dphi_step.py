# doctor/modes/max_dphi_step.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_MAX_DPHI_STEP

def get_value(data: DoctorData) -> float:
    return data.max_dphi_step

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MAX_DPHI_STEP])

CONFIG = {
    "label": "Max Single-Step |dphi|",
    "colormap": "magma",
    "vmin": 1e-4,
    "vmax": 1.0,
    "scale": "log",
}