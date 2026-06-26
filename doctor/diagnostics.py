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
    orbit_count_signed:  float

    # Disk
    hit_count:           int
    hit_radii:           list[float]   # list of floats, length = hit_count

    # Conservation Law Drift (Physics Validator)
    E:                   float
    L:                   float
    Q:                   float
    max_dH:              float
    max_dE:              float
    max_dL:              float
    max_dphi_step:       float

    # Derived Constants
    impact_parameter:    float  # b = L/E
    carter_constant:     float  # q = Q/E^2

    # Ergosphere
    steps_in_ergosphere: int
    entered_ergosphere:  bool


def collect_diagnostics(raw: np.ndarray) -> DoctorData:
    hits = []
    if raw[IDX_HIT_COUNT] > 0: hits.append(raw[IDX_HIT_R_1])
    if raw[IDX_HIT_COUNT] > 1: hits.append(raw[IDX_HIT_R_2])
    if raw[IDX_HIT_COUNT] > 2: hits.append(raw[IDX_HIT_R_3])

    return DoctorData(
        captured=bool(raw[IDX_CAPTURED]),
        termination_reason=int(raw[IDX_TERM_REASON]),
        steps_taken=int(raw[IDX_STEPS]),
        min_r=float(raw[IDX_MIN_R]),
        max_r=float(raw[IDX_MAX_R]),
        final_r=float(raw[IDX_FINAL_R]),
        final_theta=float(raw[IDX_FINAL_THETA]),
        min_delta=float(raw[IDX_MIN_DELTA]),
        orbit_count=float(raw[IDX_ORBIT_COUNT]),
        equatorial_crossings=int(raw[IDX_EQ_CROSSINGS]),
        theta_turning_points=int(raw[IDX_THETA_TURNS]),
        min_pole_gap=float(raw[IDX_MIN_POLE_GAP]),
        hit_count=int(raw[IDX_HIT_COUNT]),
        hit_radii=hits,
        E=float(raw[IDX_E]),
        L=float(raw[IDX_L]),
        Q=float(raw[IDX_Q]),
        max_dH=float(raw[IDX_MAX_DH]),
        impact_parameter=float(raw[IDX_IMPACT_PARAM]),
        carter_constant=float(raw[IDX_CARTER_CONST]),
        steps_in_ergosphere=int(raw[IDX_STEPS_IN_ERGO]),
        entered_ergosphere=bool(raw[IDX_ENTERED_ERGO]),
        max_dE=float(raw[IDX_MAX_DE]),
        max_dL=float(raw[IDX_MAX_DL]),
        orbit_count_signed=float(raw[IDX_ORBIT_COUNT_SIGNED]),
        max_dphi_step=float(raw[IDX_MAX_DPHI_STEP]),
    )