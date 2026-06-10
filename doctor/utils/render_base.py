import numpy as np
import cv2
from numba import njit, prange

from core.camera import generate_camera_rays
from core.geodesics import integrate_path_doctor
from doctor.utils.indices import NUM_DOCTOR_METRICS

@njit(parallel=True, fastmath=True, cache=True)
def _generate_diagnostic_tensor(width, height, cam_pos, ray_dirs, dt, max_steps):
    tensor = np.zeros((height, width, NUM_DOCTOR_METRICS), dtype=np.float64)
    for idx in prange(height * width):
        y = idx // width
        x = idx % width
        stats = integrate_path_doctor(cam_pos, ray_dirs[y, x], dt, max_steps)
        for k in range(NUM_DOCTOR_METRICS):
            tensor[y, x, k] = stats[k]
    return tensor

def compute_tensor(width, height, fov, cam_pos, look_at, dt=0.25, max_steps=1500, roll=0.0):
    ray_dirs = generate_camera_rays(width, height, fov, list(cam_pos), list(look_at), roll)
    return _generate_diagnostic_tensor(width, height, cam_pos, ray_dirs, dt, max_steps)

def save_diagnostic_image(img_float, filepath):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    print(f"💾 Saved diagnostic map -> {filepath}")