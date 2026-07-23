from doctor.diagnostics import DoctorData
from core.indices import IDX_ORBIT_COUNT

def get_value(data: DoctorData) -> float:
    return data.orbit_count

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_ORBIT_COUNT])

CONFIG = {
    "label":    "Orbit Count",
    "colormap": "hot",
    "vmin":     0.0,
    "vmax":     4.0,
    "scale":    "linear",
}