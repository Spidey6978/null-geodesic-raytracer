# doctor/modes/max_dphdlam.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_MAX_ABS_DPHDLAM

def get_value(data: DoctorData) -> float:
    return data.max_abs_dphdlam

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MAX_ABS_DPHDLAM])

CONFIG = {
    "label": "Max |dphi/dlambda| (raw derivative)",
    "colormap": "inferno",
    "vmin": 1e-2,
    "vmax": 1e4,
    "scale": "log",
}