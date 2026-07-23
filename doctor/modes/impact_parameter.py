from doctor.diagnostics import DoctorData
from core.indices import IDX_IMPACT_PARAM

def get_value(data: DoctorData) -> float:
    return data.impact_parameter

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_IMPACT_PARAM])

CONFIG = {
    "label":    "Impact Parameter (b = L/E)",
    "colormap": "coolwarm",
    "vmin":     -10.0,
    "vmax":     10.0,
    "scale":    "linear",
}