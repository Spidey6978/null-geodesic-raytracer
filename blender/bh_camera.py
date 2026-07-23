"""
blender/bh_camera.py
Converts a Blender camera object into the parameters needed by
generate_camera_rays(). Called once per frame before render_pixel_batch.
"""
import numpy as np


def blender_camera_to_ray_dirs(scene, width: int, height: int) -> tuple:
    """
    Reads the active camera from the Blender scene and returns
    (cam_pos, ray_dirs, fov_degrees) ready to pass to render_pixel_batch.

    Returns:
        cam_pos   : np.ndarray (3,) — camera position in BH coordinate space
        ray_dirs  : np.ndarray (H, W, 3) — ray directions per pixel
        fov_deg   : float — horizontal FOV in degrees
    """
    import bpy
    import mathutils

    cam_obj    = scene.camera
    cam_data   = cam_obj.data
    cam_matrix = cam_obj.matrix_world  # 4x4 world transform

    # ── Camera position ───────────────────────────────────────────────────────
    # Blender uses Z-up, Y-forward. Your raytracer uses Y-up.
    # The coordinate swap: Blender (X, Y, Z) → raytracer (X, Z, Y)
    bl_pos  = cam_matrix.translation
    cam_pos = np.array([bl_pos.x, bl_pos.z, bl_pos.y], dtype=np.float64)

    # ── Camera orientation ────────────────────────────────────────────────────
    # Extract forward (-Z in Blender camera space), up (Y), right (X)
    # from the rotation part of the matrix.
    rot      = cam_matrix.to_3x3()
    # Blender camera looks along -Z local axis
    forward_bl = -rot.col[2]   # -Z column
    up_bl      =  rot.col[1]   #  Y column

    # Apply the same coordinate swap
    forward = np.array([forward_bl.x, forward_bl.z, forward_bl.y], dtype=np.float64)
    up      = np.array([up_bl.x,      up_bl.z,      up_bl.y     ], dtype=np.float64)

    # Normalize (matrix should already be orthonormal but float safety)
    forward = forward / np.linalg.norm(forward)
    up      = up      / np.linalg.norm(up)
    right   = np.cross(forward, up)
    right   = right   / np.linalg.norm(right)
    up      = np.cross(right, forward)   # reorthogonalize

    # ── FOV ───────────────────────────────────────────────────────────────────
    # Blender stores lens as focal length (mm) relative to sensor size (mm).
    # angle_x is the horizontal FOV in radians.
    fov_rad = cam_data.angle_x   # Blender already gives horizontal FOV
    fov_deg = float(np.degrees(fov_rad))

    # ── Ray directions ────────────────────────────────────────────────────────
    # Same math as generate_camera_rays but using the Blender-derived axes
    # directly instead of computing them from cam_pos + look_at.
    aspect_ratio   = width / height
    viewport_h     = 2.0 * np.tan(fov_rad / 2.0)
    viewport_w     = aspect_ratio * viewport_h

    x_coords = np.linspace(-viewport_w / 2, viewport_w / 2, width)
    y_coords = np.linspace( viewport_h / 2, -viewport_h / 2, height)
    xx, yy   = np.meshgrid(x_coords, y_coords)

    ray_dirs = np.zeros((height, width, 3), dtype=np.float64)
    ray_dirs[..., 0] = forward[0] + xx * right[0] + yy * up[0]
    ray_dirs[..., 1] = forward[1] + xx * right[1] + yy * up[1]
    ray_dirs[..., 2] = forward[2] + xx * right[2] + yy * up[2]

    norms    = np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    ray_dirs = ray_dirs / norms

    return cam_pos, ray_dirs, fov_deg