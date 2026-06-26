from doctor.diagnostics import DoctorData
from core.indices import IDX_STEPS_IN_ERGO


def get_value(data: DoctorData) -> float:
    return float(data.steps_in_ergosphere)


def get_value_from_raw(stats) -> float:
    return float(stats[IDX_STEPS_IN_ERGO])


CONFIG = {
    "label": "Ergosphere Steps",
    "colormap": "plasma",
    "vmin": 0.0,
    "vmax": 100.0,
    "scale": "linear",
}
