"""
render_accretion_disk.py  —  v3: The High-Fidelity Parallel Numba Kernel

Physics implemented in the shader:
  1. Keplerian orbital velocity at the disk hit radius (capped at ISCO).
  2. Relativistic beta = v_orb / c and gamma = 1 / sqrt(1 - beta^2).
  3. Relativistic Doppler factor.
  4. Observed intensity scales as delta^4 (Luminet 1979).
  5. Novikov-Thorne temperature profile.
  6. Gravitational redshift tracking: nu_inf / nu_emit = sqrt(1 - Rs/r).
  7. Secondary (ghost) thin Einstein rings from looping null geodesics.
  8. Volumetric coronal gas glow on escaped light rays skimming the equator.
  9. Procedural lensed background star field.
  10. ACES Filmic Tone Mapping per-pixel.

Optimization:
  - @njit(parallel=True) multi-threading across all CPU cores.
  - No Python-to-C overhead; entire pipeline is compiled to C.
  - Optimized scalar math to eliminate dynamic array allocation in tight loops.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

# Clean, standard absolute imports relative to the project root
from core.camera    import generate_camera_rays
from core.geodesics import integrate_path
from core.constants import DISK_INNER, DISK_OUTER, RS, C

# ── Physical constants ────────────────────────────────────────────────────────
M = RS / 2.0
R_ISCO = 3.0 * RS

# ── Doppler & Gravitational Shaders (Compiled to C) ───────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def keplerian_beta(r):
    r_s = max(r, R_ISCO)
    beta = np.sqrt(M / r_s)
    return min(max(beta, 0.0), 0.999)

@njit(nopython=True, fastmath=True, cache=True)
def grav_redshift_factor(r):
    r_s = max(r, R_ISCO * 1.001)
    return np.sqrt(max(1.0 - RS / r_s, 0.0))

@njit(nopython=True, fastmath=True, cache=True)
def novikov_thorne_temperature(r):
    r_s = max(r, R_ISCO * 1.001)
    nt  = (r_s / R_ISCO) ** (-0.75) * max((1.0 - np.sqrt(R_ISCO / r_s)) ** 0.25, 0.0)
    return min(max(nt / 0.38, 0.0), 1.0) # 0.38 is the peak scale factor

@njit(nopython=True, fastmath=True, cache=True)
def blackbody_rgb(T_eff):
    t = min(max(T_eff, 0.0), 1.0)
    r = min(max(1.0 - 0.35 * (t - 1.0) ** 2, 0.0), 1.0)
    g = min(max(t ** 0.55, 0.0), 1.0)
    b = min(max((t - 0.82) * 5.5, 0.0), 1.0)
    return r, g, b

@njit(nopython=True, fastmath=True, cache=True)
def disk_colour(hit_radius, hit_phi, hit_vel_x, hit_vel_y, hit_vel_z, is_secondary):
    T_base = novikov_thorne_temperature(hit_radius)
    
    # Calculate Doppler Factor
    beta = keplerian_beta(hit_radius)
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    
    gas_dir_x = -np.sin(hit_phi)
    gas_dir_y = 0.0
    gas_dir_z = np.cos(hit_phi)
    
    cos_angle = -(gas_dir_x * hit_vel_x + gas_dir_y * hit_vel_y + gas_dir_z * hit_vel_z)
    delta = 1.0 / (gamma * (1.0 - beta * cos_angle))
    
    # Gravitational Redshift
    g_shift = grav_redshift_factor(hit_radius)
    
    # Combined Intensity
    combined = (delta * g_shift) ** 4
    T_eff = min(max(T_base * delta * g_shift, 0.0), 1.0)
    
    r_c, g_c, b_c = blackbody_rgb(T_eff)
    intensity = T_base * combined
    
    if is_secondary:
        intensity *= 0.18 # SECONDARY_SCALE
        
    intensity = min(max(intensity, 0.0), 1.8)
    return r_c * intensity, g_c * intensity, b_c * intensity

# ── Volumetric Coronal Glow Shader (Compiled to C) ────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def volumetric_glow(path):
    length = len(path)
    if length < 2:
        return 0.0, 0.0, 0.0

    glow_r = 0.0
    glow_g = 0.0
    glow_b = 0.0
    
    SCALE_HEIGHT = RS * 0.4
    GLOW_INNER = DISK_INNER * 0.8
    GLOW_OUTER = DISK_OUTER * 1.3

    for i in range(length):
        px = path[i, 0]
        py = path[i, 1]
        pz = path[i, 2]
        
        y = abs(py)
        r_c = np.sqrt(px*px + pz*pz)

        if r_c < GLOW_INNER or r_c > GLOW_OUTER:
            continue

        vert = np.exp(-0.5 * (y / SCALE_HEIGHT) ** 2)
        if vert < 0.005:
            continue

        T_base = novikov_thorne_temperature(r_c)
        phi = np.arctan2(pz, px)
        beta = keplerian_beta(r_c)
        gamma_lor = 1.0 / np.sqrt(1.0 - beta * beta)
        
        gas_dir_x = -np.sin(phi)
        gas_dir_y = 0.0
        gas_dir_z = np.cos(phi)

        if i + 1 < length:
            pdir_x = path[i+1, 0] - px
            pdir_y = path[i+1, 1] - py
            pdir_z = path[i+1, 2] - pz
        else:
            pdir_x = px - path[i-1, 0]
            pdir_y = py - path[i-1, 1]
            pdir_z = pz - path[i-1, 2]
            
        pnorm = np.sqrt(pdir_x*pdir_x + pdir_y*pdir_y + pdir_z*pdir_z)
        if pnorm > 0:
            pdir_x /= pnorm
            pdir_y /= pnorm
            pdir_z /= pnorm

        cos_ang = -(gas_dir_x * pdir_x + gas_dir_y * pdir_y + gas_dir_z * pdir_z)
        delta_vol = 1.0 / (gamma_lor * (1.0 - beta * cos_ang))
        g_shift = grav_redshift_factor(r_c)
        
        T_eff = min(max(T_base * delta_vol * g_shift, 0.0), 1.0)
        r_bb, g_bb, b_bb = blackbody_rgb(T_eff)

        step_i = T_base * (delta_vol * g_shift) ** 2 * vert
        weight = step_i * 0.012
        
        glow_r += r_bb * weight
        glow_g += g_bb * weight
        glow_b += b_bb * weight

    glow_r = min(max(glow_r, 0.0), 0.6)
    glow_g = min(max(glow_g, 0.0), 0.6)
    glow_b = min(max(glow_b, 0.0), 0.6)
    
    return glow_r, glow_g, glow_b

# ── Procedural Lensed Star Field Shader (Compiled to C) ───────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def star_field_colour(ray_dir, star_dirs, star_bright, star_cos_radii, star_colour, star_palettes):
    norm = np.sqrt(ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2) + 1e-12
    rx = ray_dir[0] / norm
    ry = ray_dir[1] / norm
    rz = ray_dir[2] / norm

    best_idx = -1
    max_bright = -1.0
    
    # Efficient scalar loop over all 2500 stars
    for i in range(2500):
        dot = star_dirs[i, 0] * rx + star_dirs[i, 1] * ry + star_dirs[i, 2] * rz
        if dot > star_cos_radii[i]:
            if star_bright[i] > max_bright:
                max_bright = star_bright[i]
                best_idx = i
                
    if best_idx != -1:
        col_idx = star_colour[best_idx]
        pal = star_palettes[col_idx]
        return pal[0] * max_bright * 0.9, pal[1] * max_bright * 0.9, pal[2] * max_bright * 0.9
    else:
        return 0.0, 0.0, 0.0

# ── ACES Filmic Tone Mapping Curve ────────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return min(max((x * (a * x + b)) / (x * (c * x + d) + e), 0.0), 1.0)

# ── The Parallel Render Kernel ────────────────────────────────────────────────

@njit(parallel=True, fastmath=True)
def render_kernel(width, height, cam_pos, ray_dirs,
                  star_dirs, star_bright, star_cos_radii, star_colour, star_palettes):
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    for y in prange(height):
        for x in range(width):
            start_vel = ray_dirs[y, x]

            # 1. Trace the Ray
            path, captured, hit_disk, hit_radius, hit_phi, hit_vel = integrate_path(
                cam_pos, start_vel, dt=0.5, max_steps=2000
            )

            # 2. Pixel Shader
            r_pixel = 0.0
            g_pixel = 0.0
            b_pixel = 0.0

            if captured:
                # Inside Event Horizon
                r_pixel = 0.0
                g_pixel = 0.0
                b_pixel = 0.0
            elif hit_disk:
                # Primary disk contribution
                r_p, g_p, b_p = disk_colour(hit_radius, hit_phi, hit_vel[0], hit_vel[1], hit_vel[2], False)
                r_pixel += r_p
                g_pixel += g_p
                b_pixel += b_p

                # Secondary ghost ring image calculation
                crossing_count = 0
                length = len(path)
                for k in range(1, length):
                    old_p = path[k-1]
                    new_p = path[k]
                    if old_p[1] * new_p[1] <= 0.0:
                        crossing_count += 1
                        if crossing_count == 1:
                            continue # skip the primary crossing
                        
                        dy = new_p[1] - old_p[1]
                        if dy == 0.0:
                            continue
                        t_f = -old_p[1] / dy
                        hx = old_p[0] + t_f * (new_p[0] - old_p[0])
                        hz = old_p[2] + t_f * (new_p[2] - old_p[2])
                        r_h = np.sqrt(hx*hx + hz*hz)
                        if DISK_INNER <= r_h <= DISK_OUTER:
                            phi_h = np.arctan2(hz, hx)
                            
                            # Local velocity vector direction
                            vh_x = new_p[0] - old_p[0]
                            vh_y = new_p[1] - old_p[1]
                            vh_z = new_p[2] - old_p[2]
                            vn = np.sqrt(vh_x*vh_x + vh_y*vh_y + vh_z*vh_z)
                            if vn > 0:
                                vh_x /= vn
                                vh_y /= vn
                                vh_z /= vn
                            
                            r_s, g_s, b_s = disk_colour(r_h, phi_h, vh_x, vh_y, vh_z, True)
                            r_pixel += r_s
                            g_pixel += g_s
                            b_pixel += b_s
                            break

                # Clip to HDR ceiling before tonemapping
                r_pixel = min(max(r_pixel, 0.0), 2.0)
                g_pixel = min(max(g_pixel, 0.0), 2.0)
                b_pixel = min(max(b_pixel, 0.0), 2.0)

            else:
                # Escaped: Procedural Star field + Volumetric gas glow
                final_dir_x = start_vel[0]
                final_dir_y = start_vel[1]
                final_dir_z = start_vel[2]
                
                if len(path) > 1:
                    final_dir_x = path[-1, 0] - path[-2, 0]
                    final_dir_y = path[-1, 1] - path[-2, 1]
                    final_dir_z = path[-1, 2] - path[-2, 2]

                final_dir = np.array([final_dir_x, final_dir_y, final_dir_z])
                
                # Sample star field
                s_r, s_g, s_b = star_field_colour(final_dir, star_dirs, star_bright, star_cos_radii, star_colour, star_palettes)
                
                # Volumetric coronal glow
                glow_r, glow_g, glow_b = volumetric_glow(path)

                r_pixel = s_r + glow_r
                g_pixel = s_g + glow_g
                b_pixel = s_b + glow_b

                # Clip escaped limits
                r_pixel = min(max(r_pixel, 0.0), 1.0)
                g_pixel = min(max(g_pixel, 0.0), 1.0)
                b_pixel = min(max(b_pixel, 0.0), 1.0)

            # Apply ACES filmic tonemap curve per-pixel
            image[y, x, 0] = aces_tonemap(r_pixel)
            image[y, x, 1] = aces_tonemap(g_pixel)
            image[y, x, 2] = aces_tonemap(b_pixel)

    return image

# ── Main Control ──────────────────────────────────────────────────────────────

def render():
    WIDTH   = 600
    HEIGHT  = 400
    FOV     = 60

    CAM_POS = np.array([0.0, 1.5, 15.0], dtype=np.float64)
    LOOK_AT = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Precalculate procedural star data structures
    _RNG         = np.random.default_rng(42)
    _N_STARS     = 2500
    _STAR_DIRS   = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
    _STAR_DIRS   /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)
    _STAR_BRIGHT = _RNG.power(0.25, _N_STARS).astype(np.float64)
    _STAR_RADII  = np.clip(_RNG.exponential(0.004, _N_STARS), 0.001, 0.025).astype(np.float64)
    _STAR_COS_RADII = np.cos(_STAR_RADII).astype(np.float64)
    _STAR_COLOUR = _RNG.integers(0, 3, _N_STARS).astype(np.int64)

    _STAR_PALETTES = np.array([
        [0.85, 0.90, 1.00],   # 0: blue-white
        [1.00, 0.95, 0.80],   # 1: yellow-white
        [1.00, 0.65, 0.40],   # 2: orange-red
    ], dtype=np.float64)

    print(f"📷  Initialising camera  ({WIDTH}×{HEIGHT})")
    ray_dirs = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)

    print("🚀  Compiling C-Kernel & Firing Photons... (This first compile takes ~3 seconds)")
    start_time = time.time()

    image = render_kernel(
        WIDTH, HEIGHT, CAM_POS, ray_dirs,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PALETTES
    )

    elapsed = time.time() - start_time
    print(f"✅  High-Fidelity Render complete in {elapsed:.3f}s")

    # ── Output ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.imshow(image, origin='upper')
    ax.axis('off')
    ax.set_title(
        f"Relativistic Accretion Disk (Parallel High-Fidelity Engine)\n"
        f"{WIDTH}x{HEIGHT} px | {elapsed:.2f}s render time",
        color='white', fontsize=11, pad=10
    )
    plt.tight_layout()
    plt.savefig("accretion_disk_parallel.png", bbox_inches='tight', dpi=200, facecolor='black')
    print("💾  Saved → accretion_disk_parallel.png")
    plt.show()

if __name__ == "__main__":
    render()