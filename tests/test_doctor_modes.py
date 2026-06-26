import numpy as np

from doctor.diagnostics import DoctorData
from doctor.modes import min_pole_gap
from doctor.modes import ergosphere_map
from doctor.modes import e_drift_map, l_drift_map, orbit_count_signed, max_dphi_step
from core.indices import (
    IDX_MIN_POLE_GAP,
    IDX_STEPS_IN_ERGO,
    IDX_MAX_DE,
    IDX_MAX_DL,
    IDX_ORBIT_COUNT_SIGNED,
    IDX_MAX_DPHI_STEP,
)


def test_min_pole_gap_mode_uses_diagnostics_field():
    data = DoctorData(
        captured=False,
        termination_reason=0,
        steps_taken=1,
        min_r=1.0,
        max_r=2.0,
        final_r=1.5,
        final_theta=0.2,
        min_delta=0.1,
        orbit_count=0.0,
        equatorial_crossings=0,
        theta_turning_points=0,
        min_pole_gap=0.35,
        hit_count=0,
        hit_radii=[],
        E=1.0,
        L=0.0,
        Q=0.0,
        max_dH=0.0,
        max_dE=0.0,
        max_dL=0.0,
        orbit_count_signed=0.0,
        max_dphi_step=0.0,
        impact_parameter=0.0,
        carter_constant=0.0,
        steps_in_ergosphere=0,
        entered_ergosphere=False,
    )

    assert min_pole_gap.get_value(data) == 0.35


def test_min_pole_gap_mode_reads_raw_stats_index():
    stats = np.zeros(25, dtype=np.float64)
    stats[IDX_MIN_POLE_GAP] = 0.77

    assert min_pole_gap.get_value_from_raw(stats) == 0.77


def test_ergosphere_mode_reads_raw_steps_index():
    stats = np.zeros(25, dtype=np.float64)
    stats[IDX_STEPS_IN_ERGO] = 12.0

    assert ergosphere_map.get_value_from_raw(stats) == 12.0


def test_drift_and_orbit_modes_read_their_indices():
    stats = np.zeros(29, dtype=np.float64)
    stats[IDX_MAX_DE] = 1e-5
    stats[IDX_MAX_DL] = 2e-5
    stats[IDX_ORBIT_COUNT_SIGNED] = -1.25
    stats[IDX_MAX_DPHI_STEP] = 0.33

    assert e_drift_map.get_value_from_raw(stats) == 1e-5
    assert l_drift_map.get_value_from_raw(stats) == 2e-5
    assert orbit_count_signed.get_value_from_raw(stats) == -1.25
    assert max_dphi_step.get_value_from_raw(stats) == 0.33


def test_new_modes_handle_short_stats_arrays_without_crashing():
    stats = np.zeros(25, dtype=np.float64)

    assert np.isnan(e_drift_map.get_value_from_raw(stats))
    assert np.isnan(l_drift_map.get_value_from_raw(stats))
    assert np.isnan(orbit_count_signed.get_value_from_raw(stats))
    assert np.isnan(max_dphi_step.get_value_from_raw(stats))
