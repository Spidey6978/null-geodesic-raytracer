from doctor.diagnostics import DoctorData
from core.indices import IDX_MIN_R

def get_value(data: DoctorData) -> float:
    return data.min_r

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MIN_R])

CONFIG = {
    "label":    "Minimum Approach Radius (min_r)",
    "colormap": "viridis",
    "vmin":     1.0,   # roughly r_horizon for typical spin values
    "vmax":     10.0,  # captures photon sphere through mid-disk region
    "scale":    "log",  # log scale spreads out the interesting near-BH detail
}