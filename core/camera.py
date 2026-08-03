"""
Module: core.camera
Generates a grid of 3D ray directions for a pinhole camera model.
Utilizes NumPy vectorization for massive parallel math.
Includes 3D spline trajectory math (Bezier & Catmull-Rom) for camera flybys.
"""
import math
import numpy as np


def generate_camera_rays(width, height, fov_degrees, cam_pos, look_at, roll_degrees=0.0):
    """
    Generates a normalized direction vector for every pixel on the screen.
    Returns an array of shape (height, width, 3).
    """
    # 1. Camera Axis Vectors
    cam_pos = np.array(cam_pos, dtype=np.float64)
    look_at = np.array(look_at, dtype=np.float64)
    up_guide = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward = look_at - cam_pos
    norm_f = np.linalg.norm(forward)
    forward = forward / norm_f if norm_f > 1e-12 else np.array([0.0, 0.0, -1.0])

    right = np.cross(forward, up_guide)
    norm_r = np.linalg.norm(right)
    right = right / norm_r if norm_r > 1e-12 else np.array([1.0, 0.0, 0.0])

    up = np.cross(right, forward)

    # 2. Screen Dimensions
    aspect_ratio = width / height
    fov_radians = np.deg2rad(fov_degrees)

    viewport_height = 2.0 * np.tan(fov_radians / 2.0)
    viewport_width = aspect_ratio * viewport_height

    # 3. Vectorized Pixel Grid Generation
    x_coords = np.linspace(-viewport_width / 2, viewport_width / 2, width)
    y_coords = np.linspace(viewport_height / 2, -viewport_height / 2, height)

    xx, yy = np.meshgrid(x_coords, y_coords)

    # --- Apply Camera Roll (2D Rotation Matrix) ---
    roll_rad = np.deg2rad(roll_degrees)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)

    xx_rot = xx * cos_r - yy * sin_r
    yy_rot = xx * sin_r + yy * cos_r

    # 4. Calculate Ray Directions using the rotated grid
    ray_dirs = np.zeros((height, width, 3), dtype=np.float64)

    ray_dirs[..., 0] = forward[0] + xx_rot * right[0] + yy_rot * up[0]
    ray_dirs[..., 1] = forward[1] + xx_rot * right[1] + yy_rot * up[1]
    ray_dirs[..., 2] = forward[2] + xx_rot * right[2] + yy_rot * up[2]

    norms = np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    ray_dirs = ray_dirs / norms

    return ray_dirs


def eval_bezier_3d(control_points: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluates a 3D Bezier curve for parameter t in [0, 1].
    control_points: Array of shape (N, 3) where N >= 2.
    """
    pts = np.array(control_points, dtype=np.float64)
    n = len(pts) - 1
    if n == 0:
        return pts[0]

    t = float(np.clip(t, 0.0, 1.0))
    res = np.zeros(3, dtype=np.float64)
    for i in range(n + 1):
        comb = math.comb(n, i)
        b_in = comb * (t**i) * ((1.0 - t)**(n - i))
        res += b_in * pts[i]
    return res


def eval_catmull_rom_3d(waypoints: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluates a 3D Catmull-Rom spline passing through waypoints for t in [0, 1].
    waypoints: Array of shape (N, 3) where N >= 4.
    """
    pts = np.array(waypoints, dtype=np.float64)
    n_pts = len(pts)
    if n_pts < 4:
        raise ValueError("Catmull-Rom spline requires at least 4 waypoints.")

    t = float(np.clip(t, 0.0, 1.0))
    num_segments = n_pts - 3
    scaled_t = t * num_segments
    seg_idx = int(scaled_t)
    if seg_idx >= num_segments:
        seg_idx = num_segments - 1
    local_t = scaled_t - seg_idx

    p0 = pts[seg_idx]
    p1 = pts[seg_idx + 1]
    p2 = pts[seg_idx + 2]
    p3 = pts[seg_idx + 3]

    t2 = local_t * local_t
    t3 = t2 * local_t

    f0 = -0.5 * t3 + t2 - 0.5 * local_t
    f1 = 1.5 * t3 - 2.5 * t2 + 1.0
    f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * local_t
    f3 = 0.5 * t3 - 0.5 * t2

    return f0 * p0 + f1 * p1 + f2 * p2 + f3 * p3


def generate_spline_camera_path(waypoints: list, num_frames: int, look_at: list = None, fov: float = 100.0, roll: float = 0.0) -> list:
    """
    Generates an array of camera frame parameter dictionaries [(cam_pos, look_at, fov, roll)]
    interpolated smoothly along a 3D spline trajectory across num_frames.
    """
    pts = np.array(waypoints, dtype=np.float64)
    target_look = np.array(look_at, dtype=np.float64) if look_at is not None else np.array([0.0, 0.0, 0.0], dtype=np.float64)

    frames = []
    num_frames = max(num_frames, 1)

    if len(pts) < 4:
        if len(pts) == 1:
            pts = np.tile(pts, (4, 1))
        elif len(pts) == 2:
            p0 = pts[0] - (pts[1] - pts[0])
            p3 = pts[1] + (pts[1] - pts[0])
            pts = np.vstack([p0, pts, p3])
        elif len(pts) == 3:
            p0 = pts[0] - (pts[1] - pts[0])
            pts = np.vstack([p0, pts])

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        cam_pos = eval_catmull_rom_3d(pts, t)
        frames.append({
            "cam_pos": cam_pos.tolist(),
            "look_at": target_look.tolist(),
            "fov": fov,
            "roll": roll,
            "frame_index": i,
            "t": t
        })

    return frames