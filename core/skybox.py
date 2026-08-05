"""
Module: core.skybox
Equirectangular UV texture sampler for background skyboxes and celestial HDRI maps.
Numba-compiled for ultra-fast ray sampling upon photon escape (r > R_bounds).
"""

import numpy as np
from numba import njit
from pathlib import Path


@njit(fastmath=True)
def sample_skybox_equirectangular(vx: float, vy: float, vz: float, skybox_img: np.ndarray, sky_w: int, sky_h: int) -> np.ndarray:
    """
    Samples RGB color from an equirectangular skybox texture for an escaping ray vector (vx, vy, vz).
    vx, vy, vz: Cartesian direction components of escaping ray.
    skybox_img: float64 array of shape (height, width, 3).
    Returns RGB array [R, G, B].
    """
    # 1. Normalize direction vector
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    if norm > 1e-12:
        vx /= norm
        vy /= norm
        vz /= norm
    else:
        vx, vy, vz = 0.0, 0.0, 1.0

    # 2. Spherical coordinates
    # theta: polar angle from y-axis (0 at top pole, pi at bottom pole)
    cos_theta = min(max(vy, -1.0), 1.0)
    theta = np.arccos(cos_theta)

    # phi: azimuthal angle in x-z plane [-pi, pi]
    phi = np.arctan2(vz, vx)

    # 3. Map to UV coordinates in [0.0, 1.0]
    u = (phi + np.pi) / (2.0 * np.pi)
    v = theta / np.pi

    # 4. Convert UV to pixel coordinates
    px = int(u * (sky_w - 1))
    py = int(v * (sky_h - 1))

    px = min(max(px, 0), sky_w - 1)
    py = min(max(py, 0), sky_h - 1)

    out_rgb = np.zeros(3, dtype=np.float64)
    out_rgb[0] = skybox_img[py, px, 0]
    out_rgb[1] = skybox_img[py, px, 1]
    out_rgb[2] = skybox_img[py, px, 2]

    return out_rgb


def generate_procedural_skybox(width: int = 2048, height: int = 1024) -> np.ndarray:
    """
    Generates a fallback high-resolution deep-space procedural nebula skybox texture.
    Returns float64 array of shape (height, width, 3) in range [0.0, 1.0].
    """
    img = np.zeros((height, width, 3), dtype=np.float64)

    # Background space glow
    y_coords = np.linspace(0, np.pi, height)
    x_coords = np.linspace(-np.pi, np.pi, width)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Galactic plane band (y near np.pi/2)
    galactic_plane = np.exp(-12.0 * (yy - np.pi / 2.0)**2)
    img[..., 0] += 0.15 * galactic_plane + 0.05 * np.sin(xx * 2.0)**2 * galactic_plane
    img[..., 1] += 0.08 * galactic_plane
    img[..., 2] += 0.25 * galactic_plane + 0.1 * np.cos(xx * 3.0)**2 * galactic_plane

    # Scatter stars
    np.random.seed(42)
    num_stars = 3000
    star_x = np.random.randint(0, width, num_stars)
    star_y = np.random.randint(0, height, num_stars)
    star_bright = np.random.uniform(0.4, 1.0, num_stars)

    for i in range(num_stars):
        sx, sy, b = star_x[i], star_y[i], star_bright[i]
        color_choice = i % 3
        if color_choice == 0:
            img[sy, sx] += np.array([b, b * 0.8, b * 0.6])  # warm star
        elif color_choice == 1:
            img[sy, sx] += np.array([b * 0.7, b * 0.8, b])  # blue star
        else:
            img[sy, sx] += np.array([b, b, b])              # white star

    return np.clip(img, 0.0, 1.0)


def load_skybox_texture(filepath: str = None) -> tuple:
    """
    Loads an equirectangular image texture from file or generates a fallback procedural skybox.
    Returns (skybox_img, skybox_w, skybox_h).
    """
    if filepath and Path(filepath).exists():
        try:
            import matplotlib.pyplot as plt
            img = plt.imread(filepath)
            if img.dtype == np.uint8:
                img = img.astype(np.float64) / 255.0
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] > 3:
                img = img[..., :3]
            img = img.astype(np.float64)
            h, w = img.shape[:2]
            return img, w, h
        except Exception as e:
            print(f"⚠️ Failed to load skybox from {filepath}: {e}. Falling back to procedural skybox.")

    img = generate_procedural_skybox()
    h, w = img.shape[:2]
    return img, w, h
