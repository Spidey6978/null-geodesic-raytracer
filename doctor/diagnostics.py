from dataclasses import dataclass
import numpy as np
from core.indices import *

@dataclass
class DoctorData:
    # Termination
    captured:            bool
    termination_reason:  int
    steps_taken:         int

    # Radial & Geometric
    min_r:               float
    max_r:               float
    final_r:             float
    final_theta:         float
    min_delta:           float

    # Orbit Dynamics
    orbit_count:         float
    equatorial_crossings: int
    theta_turning_points: int
    min_pole_gap:        float

    # Disk
    hit_count:           int
    hit_radii:           list[float]   # list of floats, length = hit_count

    # Conservation Law Drift (Physics Validator)
    E:                   float
    L:                   float
    Q:                   float
    max_dH:              float

    # Derived Constants
    impact_parameter:    float  # b = L/E
    carter_constant:     float  # q = Q/E^2

    # Ergosphere
    steps_in_ergosphere: int
    entered_ergosphere:  bool


def collect_diagnostics(raw: np.ndarray) -> DoctorData:
    """Wraps the raw Numba float64 array into a human-readable DoctorData object."""
    hits = []
    if raw[9] > 0: hits.append(raw[21])
    if raw[9] > 1: hits.append(raw[22])
    if raw[9] > 2: hits.append(raw[23])

    return DoctorData(
        captured=bool(raw[0]),
        termination_reason=int(raw[1]),
        steps_taken=int(raw[2]),
        min_r=float(raw[3]),
        max_r=float(raw[4]),
        final_r=float(raw[5]),
        final_theta=float(raw[6]),
        min_delta=float(raw[20]),
        orbit_count=float(raw[7]),
        equatorial_crossings=int(raw[8]),
        theta_turning_points=int(raw[17]),
        min_pole_gap=float(raw[16]),
        hit_count=int(raw[9]),
        hit_radii=hits,
        E=float(raw[10]),
        L=float(raw[11]),
        Q=float(raw[12]),
        max_dH=float(raw[13]),
        impact_parameter=float(raw[14]),
        carter_constant=float(raw[15]),
        steps_in_ergosphere=int(raw[18]),
        entered_ergosphere=bool(raw[19]),
    )