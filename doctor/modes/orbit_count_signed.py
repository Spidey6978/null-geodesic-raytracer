# doctor/modes/orbit_count_signed.py
from doctor.diagnostics import DoctorData
from core.indices import IDX_ORBIT_COUNT_SIGNED

def get_value(data: DoctorData) -> float:
    return data.orbit_count_signed

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_ORBIT_COUNT_SIGNED])

CONFIG = {
    "label": "Signed Orbit Count (net winding)",
    "colormap": "coolwarm",   # centered at 0 — direction matters
    "vmin": -4.0,
    "vmax": 4.0,
    "scale": "linear",
}