from doctor.diagnostics import DoctorData
from core.indices import IDX_MIN_POLE_GAP


def get_value(data: DoctorData) -> float:
    return data.min_pole_gap


def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MIN_POLE_GAP])


CONFIG = {
    "label": "Minimum Pole Gap",
    "colormap": "viridis",
    "vmin": 0.0,
    "vmax": 1.0,
    "scale": "linear",
}
