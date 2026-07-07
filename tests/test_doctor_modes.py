import sys
from unittest.mock import patch

import numpy as np
import pytest

from core.geodesics import _compute_conserved_quantities
from doctor.diagnostics import DoctorData
from doctor.modes import min_pole_gap
from doctor.modes import ergosphere_map
from doctor.modes import e_drift_map, l_drift_map, orbit_count_signed, max_dphi_step
from core.indices import (
    IDX_TERM_REASON,
    IDX_MIN_POLE_GAP,
    IDX_STEPS_IN_ERGO,
    IDX_MAX_DE,
    IDX_MAX_DL,
    IDX_ORBIT_COUNT_SIGNED,
    IDX_MAX_DPHI_STEP,
    NUM_DOCTOR_METRICS,
)


def test_compute_conserved_quantities_scales_with_initial_momentum():
    r = 10.0
    theta = np.pi / 2.0
    a = 0.5
    mass = 1.0

    base = _compute_conserved_quantities(r, theta, 0.1, 0.05, 0.2, a, mass)
    scaled = _compute_conserved_quantities(r, theta, 0.2, 0.1, 0.4, a, mass)

    assert scaled[0] == pytest.approx(2.0 * base[0])
    assert scaled[1] == pytest.approx(2.0 * base[1])
    assert scaled[2] == pytest.approx(4.0 * base[2])
    assert scaled[3] == pytest.approx(2.0 * base[3])
    assert scaled[4] == pytest.approx(2.0 * base[4])


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


def test_new_modes_raise_for_short_stats_arrays():
    stats = np.zeros(25, dtype=np.float64)

    with np.testing.assert_raises(IndexError):
        e_drift_map.get_value_from_raw(stats)
    with np.testing.assert_raises(IndexError):
        l_drift_map.get_value_from_raw(stats)
    with np.testing.assert_raises(IndexError):
        orbit_count_signed.get_value_from_raw(stats)
    with np.testing.assert_raises(IndexError):
        max_dphi_step.get_value_from_raw(stats)


def test_pixel_probe_main_runs_with_sample_args():
    import doctor.tools.pixel_probe as pixel_probe

    sample_stats = np.zeros(NUM_DOCTOR_METRICS, dtype=np.float64)
    sample_stats[0] = 0.0
    sample_stats[1] = 4.0
    sample_stats[3] = 2.0
    sample_stats[13] = 0.1
    sample_stats[14] = 0.2
    sample_stats[17] = 0.3
    sample_stats[25] = 1e-4
    sample_stats[26] = 2e-4
    sample_stats[28] = 0.01
    sample_stats[29] = 0.02
    sample_stats[30] = 0.03

    with patch.object(pixel_probe, "generate_camera_rays", return_value=np.zeros((2, 2, 3), dtype=np.float64)), \
         patch.object(pixel_probe, "integrate_path_doctor", return_value=sample_stats):
        original_argv = sys.argv[:]
        sys.argv = [
            "pixel_probe.py",
            "--probe-x", "1",
            "--probe-y", "1",
            "--width", "2",
            "--height", "2",
            "--radius", "0",
        ]
        try:
            pixel_probe.main()
        finally:
            sys.argv = original_argv


def test_pixel_probe_does_not_swallow_integrator_errors():
    import doctor.tools.pixel_probe as pixel_probe

    with patch.object(pixel_probe, "generate_camera_rays", return_value=np.zeros((1, 1, 3), dtype=np.float64)), \
         patch.object(pixel_probe, "integrate_path_doctor", side_effect=IndexError("stale metric layout")):
        original_argv = sys.argv[:]
        sys.argv = [
            "pixel_probe.py",
            "--probe-x", "0",
            "--probe-y", "0",
            "--width", "1",
            "--height", "1",
            "--radius", "0",
        ]
        try:
            with np.testing.assert_raises_regex(IndexError, "stale metric layout"):
                pixel_probe.main()
        finally:
            sys.argv = original_argv


def test_pixel_probe_clips_window_at_image_edges():
    import doctor.tools.pixel_probe as pixel_probe

    sample_stats = np.zeros(NUM_DOCTOR_METRICS, dtype=np.float64)
    sample_stats[IDX_TERM_REASON] = 4.0
    ray_dirs = np.zeros((3, 3, 3), dtype=np.float64)

    with patch.object(pixel_probe, "generate_camera_rays", return_value=ray_dirs), \
         patch.object(pixel_probe, "integrate_path_doctor", return_value=sample_stats) as integrate:
        original_argv = sys.argv[:]
        sys.argv = [
            "pixel_probe.py",
            "--probe-x", "0",
            "--probe-y", "0",
            "--width", "3",
            "--height", "3",
            "--radius", "1",
        ]
        try:
            pixel_probe.main()
        finally:
            sys.argv = original_argv

    assert integrate.call_count == 4
