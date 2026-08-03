"""
Module: tests.test_camera_splines
Automated unit tests for 3D camera Bezier and Catmull-Rom spline trajectory interpolation.
"""

import numpy as np
from core.camera import (
    eval_bezier_3d,
    eval_catmull_rom_3d,
    generate_spline_camera_path,
)


def test_eval_bezier_3d():
    ctrl_pts = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 10.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    p_start = eval_bezier_3d(ctrl_pts, 0.0)
    p_mid = eval_bezier_3d(ctrl_pts, 0.5)
    p_end = eval_bezier_3d(ctrl_pts, 1.0)

    assert np.allclose(p_start, [0.0, 0.0, 0.0])
    assert np.allclose(p_end, [10.0, 0.0, 0.0])
    assert p_mid[1] > 0.0  # Curve bends upward toward y=10.0


def test_eval_catmull_rom_3d():
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 1.0],
        [6.0, 4.0, 2.0],
        [10.0, 0.0, 3.0]
    ])
    p_start = eval_catmull_rom_3d(waypoints, 0.0)
    p_end = eval_catmull_rom_3d(waypoints, 1.0)

    # Catmull-Rom spline passes directly through inner control points P1 and P2
    assert np.allclose(p_start, waypoints[1])
    assert np.allclose(p_end, waypoints[2])


def test_generate_spline_camera_path():
    waypoints = [
        [0.0, 5.0, 15.0],
        [5.0, 2.0, 10.0],
        [10.0, 0.0, 5.0]
    ]
    path = generate_spline_camera_path(waypoints, num_frames=10, fov=90.0)

    assert len(path) == 10
    assert path[0]["frame_index"] == 0
    assert path[9]["frame_index"] == 9
    assert len(path[0]["cam_pos"]) == 3
    assert path[0]["fov"] == 90.0
