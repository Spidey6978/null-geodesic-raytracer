"""
Module: core.camera
Generates a grid of 3D ray directions for a pinhole camera model.
Utilizes NumPy vectorization for massive parallel math.
"""
import numpy as np

def generate_camera_rays(width, height, fov_degrees, cam_pos, look_at, roll_degrees=0.0):
    """
    Generates a normalized direction vector for every pixel on the screen.
    Returns an array of shape (height, width, 3).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not (0.0 < fov_degrees < 180.0):
        raise ValueError("fov_degrees must be between 0 and 180")

    # 1. Camera Axis Vectors
    cam_pos = np.array(cam_pos, dtype=np.float64)
    look_at = np.array(look_at, dtype=np.float64)
    up_guide = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward = look_at - cam_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("cam_pos and look_at must be different points")
    forward = forward / forward_norm

    right = np.cross(forward, up_guide)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        up_guide = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_guide)
        right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        raise ValueError("camera up vector is degenerate for this view direction")
    right = right / right_norm

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

    # --- NEW: Apply Camera Roll (2D Rotation Matrix) ---
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
    if np.any(norms < 1e-12):
        raise ValueError("generated a zero-length camera ray")
    ray_dirs = ray_dirs / norms

    return ray_dirs
