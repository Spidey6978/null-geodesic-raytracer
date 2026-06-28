import os

import numpy as np
import cv2
from numba import njit, prange

from core.camera import generate_camera_rays
from core.geodesics import integrate_path_doctor
from core.indices import NUM_DOCTOR_METRICS

@njit(parallel=True, fastmath=True, cache=True)
def _generate_diagnostic_tensor(width, height, cam_pos, ray_dirs, dt, max_steps, mass, a, r_outer_horizon, disk_inner, disk_outer, sim_bounds, num_metrics):
    tensor = np.zeros((height, width, num_metrics), dtype=np.float64)
    for idx in prange(height * width):
        y = idx // width
        x = idx % width
        stats = integrate_path_doctor(
            cam_pos, ray_dirs[y, x], dt, max_steps, 
            mass, a, r_outer_horizon, disk_inner, disk_outer, sim_bounds
        )
        for k in range(num_metrics):
            tensor[y, x, k] = stats[k]
    return tensor

def compute_tensor(width, height, fov, cam_pos, look_at, dt=0.25, max_steps=1500, roll=0.0, spin=0.0, mass=1.0):
    # ── DYNAMIC PHYSICS BOUNDARIES ──
    a = spin * mass
    
    # 1. Event Horizon
    sqrt_term = (mass**2 - a**2)**0.5 if mass**2 >= a**2 else 0.0
    r_outer_horizon = mass + sqrt_term
    
    # 2. ISCO (Innermost Stable Circular Orbit)
    if abs(a) < 1e-10:
        disk_inner = 6.0 * mass
    else:
        Z1 = 1.0 + (1.0 - (a/mass)**2)**(1/3) * ((1.0 + a/mass)**(1/3) + (1.0 - a/mass)**(1/3))
        Z2 = (3.0 * (a/mass)**2 + Z1**2)**0.5
        disk_inner = mass * (3.0 + Z2 - ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**0.5)
        
    # 3. Outer Limits
    disk_outer = 18.0 * (2.0 * mass)
    sim_bounds = 200.0 * (2.0 * mass)
    
    ray_dirs = generate_camera_rays(width, height, fov, list(cam_pos), list(look_at), roll)
    
    return _generate_diagnostic_tensor(
        width, height, cam_pos, ray_dirs, dt, max_steps, 
        mass, a, r_outer_horizon, disk_inner, disk_outer, sim_bounds,
        NUM_DOCTOR_METRICS
    )

def save_diagnostic_image(img_float, filepath):
    import os
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)    
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    print(f"💾 Saved diagnostic map -> {filepath}")
