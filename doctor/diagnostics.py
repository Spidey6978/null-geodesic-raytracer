from dataclasses import dataclass, field
import numpy as np
from core.indices import *

@dataclass
class DoctorData:
    # Termination
    captured:            bool = False
    termination_reason:  int = 0
    steps_taken:         int = 0

    # Radial & Geometric
    min_r:               float = 0.0
    max_r:               float = 0.0
    final_r:             float = 0.0
    final_theta:         float = 0.0
    min_delta:           float = 0.0

    # Orbit Dynamics
    orbit_count:         float = 0.0
    equatorial_crossings: int = 0
    theta_turning_points: int = 0
    min_pole_gap:        float = 0.0
    orbit_count_signed:  float = 0.0

    # Disk
    hit_count:           int = 0
    hit_radii:           list[float] = field(default_factory=list)   # list of floats, length = hit_count

    # Conservation Law Drift (Physics Validator)
    E:                   float = 0.0
    L:                   float = 0.0
    Q:                   float = 0.0
    max_dH:              float = 0.0
    max_dE:              float = 0.0
    max_dL:              float = 0.0
    max_dphi_step:       float = 0.0
    max_abs_dphdlam:     float = 0.0
    max_inv_sin2:        float = 0.0

    # Derived Constants
    impact_parameter:    float = 0.0  # b = L/E
    carter_constant:     float = 0.0  # q = Q/E^2

    # Ergosphere
    steps_in_ergosphere: int = 0
    entered_ergosphere:  bool = False


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
        max_abs_dphdlam=float(raw[IDX_MAX_ABS_DPHDLAM]),
        max_inv_sin2=float(raw[IDX_MAX_INV_SIN2]),
    )