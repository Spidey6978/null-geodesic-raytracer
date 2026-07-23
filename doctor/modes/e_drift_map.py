# doctor/modes/e_drift_map.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_MAX_DE

def get_value(data: DoctorData) -> float:
    return data.max_dE

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MAX_DE])

CONFIG = {
    "label": "Energy Drift (|E_check - E|)",
    "colormap": "inferno",
    "vmin": 1e-8,
    "vmax": 1e-2,
    "scale": "log",
}
