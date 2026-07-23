from doctor.diagnostics import DoctorData
from core.indices import IDX_STEPS

def get_value(data: DoctorData) -> float:
    return data.steps_taken

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_STEPS])

CONFIG = {
    "label":    "Steps Taken (Integration Cost)",
    "colormap": "plasma",
    "vmin":     0.0,
    "vmax":     5000.0,  # match your max_steps
    "scale":    "linear",
}