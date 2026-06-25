from doctor.diagnostics import DoctorData
from core.indices import IDX_MAX_DH

def get_value(data: DoctorData) -> float:
    return data.max_dH

def get_value_from_raw(stats) -> float:
    return float(stats[IDX_MAX_DH])

CONFIG = {
    "label":    "Hamiltonian Drift (Numerical Error Map)",
    "colormap": "inferno",
    "vmin":     1e-6,
    "vmax":     1e-1,
    "scale":    "log",
}