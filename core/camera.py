"""
Module: core.camera
Generates a grid of 3D ray directions for a pinhole camera model.
Utilizes NumPy vectorization for massive parallel math.
"""
import numpy as np

def generate_camera_rays(width, height, fov_degrees, cam_pos, look_at):
    """
    Generates a normalized direction vector for every pixel on the screen.
    Returns an array of shape (height, width, 3).
    """
    # 1. Camera Axis Vectors (The local coordinate system of the camera)
    cam_pos = np.array(cam_pos, dtype=np.float64)
    look_at = np.array(look_at, dtype=np.float64)
    up_guide = np.array([0.0, 1.0, 0.0], dtype=np.float64) # Global Up

    # Forward direction (Z-axis)
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)

    # Right direction (X-axis)
    right = np.cross(forward, up_guide)
    right = right / np.linalg.norm(right)

    # True Up direction (Y-axis)
    up = np.cross(right, forward)

    # 2. Screen Dimensions (Field of View math)
    aspect_ratio = width / height
    fov_radians = np.deg2rad(fov_degrees)
    
    # Tan(fov/2) gives us the physical height of the screen in world units
    viewport_height = 2.0 * np.tan(fov_radians / 2.0)
    viewport_width = aspect_ratio * viewport_height

    # 3. Vectorized Pixel Grid Generation
    # We create a grid of X and Y coordinates for the screen
    x_coords = np.linspace(-viewport_width / 2, viewport_width / 2, width)
    # Y is flipped so +Y is up, -Y is down (standard graphics convention)
    y_coords = np.linspace(viewport_height / 2, -viewport_height / 2, height)
    
    xx, yy = np.meshgrid(x_coords, y_coords)

    # 4. Calculate Ray Directions
    # For every pixel, we start at the forward vector, and shift it left/right/up/down
    # using our Right and Up vectors.
    
    # We use np.dstack to combine the 2D grids into a 3D structure easily, 
    # then reshape it into a flat list of vectors.
    
    # The shape of `ray_dirs` will be (height, width, 3)
    ray_dirs = np.zeros((height, width, 3), dtype=np.float64)
    
    # Add the forward component to all rays
    ray_dirs[..., 0] = forward[0] + xx * right[0] + yy * up[0]
    ray_dirs[..., 1] = forward[1] + xx * right[1] + yy * up[1]
    ray_dirs[..., 2] = forward[2] + xx * right[2] + yy * up[2]

    # Normalize all vectors simultaneously
    # np.linalg.norm along axis=2 gets the length of each [x,y,z] vector
    norms = np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    ray_dirs = ray_dirs / norms

    return ray_dirs