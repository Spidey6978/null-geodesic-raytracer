# doctor/modes/max_inv_sin2.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_MAX_INV_SIN2

def get_value(data: DoctorData) -> float:
    return data.max_inv_sin2

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MAX_INV_SIN2])

CONFIG = {
    "label": "Max 1/(sin²θ + ε) — Polar Singularity Strength",
    "colormap": "inferno",
    "vmin": 1.0,
    "vmax": 1e7,    # matches your 1e-7 softening floor's reciprocal
    "scale": "log",
}