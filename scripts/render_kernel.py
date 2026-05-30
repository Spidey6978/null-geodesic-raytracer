"""
render_accretion_disk.py  —  v4c: Ultimate Performance & Saturation Tuning

Optimizations implemented:
  1. Trigonometric elimination: Star check uses dot-product comparison against precalculated 
     cos(radii), removing millions of np.arccos calculations per frame.
  2. Volumetric Stride: Skips redundant path points with a step stride of 4, cutting math overhead by 75%.
  3. Bounding Box Squared Check: Uses squared distance checks to skip costly np.sqrt calls for glow points.
  4. Hue-Preserving Clamping: Scales entire RGB vector uniformly to preserve vibrant oranges and reds.
  5. Saturated Novikov-Thorne Decay: Cools outer disk boundary beautifully to deep orange-reds.
  6. Radiative Transfer Blending: Blends stars, coronal glow, and disk emissions using absorption.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

from core.camera    import generate_camera_rays
from core.geodesics import integrate_path
from core.constants import DISK_INNER, DISK_OUTER, RS, C

M      = RS / 2.0
R_ISCO = 3.0 * RS

# ── Planck locus keyframes (Saturated and balanced for dynamic range) ─────────
_PLANCK_T = np.array([0.00, 0.20, 0.45, 0.65, 0.82, 1.00], dtype=np.float64)
_PLANCK_R = np.array([0.75, 1.00, 1.00, 1.00, 0.98, 0.70], dtype=np.float64)
_PLANCK_G = np.array([0.03, 0.32, 0.55, 0.82, 0.95, 0.82], dtype=np.float64)
_PLANCK_B = np.array([0.00, 0.00, 0.05, 0.25, 0.88, 1.00], dtype=np.float64)

_HIT_W = np.array([1.0, 0.18, 0.06, 0.02], dtype=np.float64)

# ── Point Source Star Field Pre-Calculations ──────────────────────────────────
_RNG         = np.random.default_rng(42)
_N_STARS     = 5000
_STAR_DIRS   = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
_STAR_DIRS   /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)
_STAR_BRIGHT = (_RNG.power(0.4, _N_STARS) * 0.90 + 0.10).astype(np.float64)
_STAR_RADII  = np.clip(_RNG.exponential(0.0004, _N_STARS), 0.0003, 0.0015).astype(np.float64)
_STAR_COS_RADII = np.cos(_STAR_RADII).astype(np.float64) # Precalculated for trig elimination

_STAR_COLOUR = _RNG.choice(
    np.array([0, 1, 2, 3], dtype=np.int64), size=_N_STARS,
    p=[0.05, 0.30, 0.35, 0.30]
).astype(np.int64)

_STAR_PAL = np.array([
    [0.70, 0.82, 1.00], # Blue-white
    [0.97, 0.97, 1.00], # Pure white
    [1.00, 0.78, 0.42], # Yellow-orange
    [1.00, 0.40, 0.15], # Deep red-orange
], dtype=np.float64)


# ── Optimized Numba Shaders ───────────────────────────────────────────────────

@njit(cache=True)
def _keplerian_beta(r):
    r_s  = r if r > R_ISCO else R_ISCO
    beta = (M / r_s) ** 0.5
    return min(beta, 0.999)


@njit(cache=True)
def _doppler_factor(hit_radius, hit_phi, hv0, hv1, hv2):
    beta      = _keplerian_beta(hit_radius)
    gamma     = 1.0 / (1.0 - beta * beta) ** 0.5
    gx        = -np.sin(hit_phi)
    gz        =  np.cos(hit_phi)
    cos_angle = gx * (-hv0) + gz * (-hv2)
    denom     = gamma * (1.0 - beta * cos_angle)
    if abs(denom) < 1e-9:
        return 1.0
    return 1.0 / denom


@njit(cache=True)
def _grav_redshift(r):
    r_s = r if r > R_ISCO * 1.001 else R_ISCO * 1.001
    val = 1.0 - RS / r_s
    return val ** 0.5 if val > 0.0 else 0.0


@njit(cache=True)
def _novikov_thorne(r):
    r_s = r if r > R_ISCO * 1.001 else R_ISCO * 1.001
    # Standard Novikov-Thorne profile
    nt  = (r_s / R_ISCO) ** (-0.75) * max((1.0 - (R_ISCO / r_s) ** 0.5) ** 0.25, 0.0)
    # Apply cooling decay factor to the outer regions so they fade gracefully to orange-reds
    decay = (R_ISCO / r_s) ** 0.5
    raw = (nt / 0.38) * decay
    return raw if raw < 1.0 else 1.0


@njit(cache=True)
def _blackbody_rgb(T_eff, out):
    t = T_eff
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0

    idx = 0
    if t >= _PLANCK_T[4]: idx = 4
    elif t >= _PLANCK_T[3]: idx = 3
    elif t >= _PLANCK_T[2]: idx = 2
    elif t >= _PLANCK_T[1]: idx = 1
    else: idx = 0

    t0    = _PLANCK_T[idx]
    t1    = _PLANCK_T[idx + 1]
    alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0.0 else 0.0

    r = _PLANCK_R[idx] + alpha * (_PLANCK_R[idx + 1] - _PLANCK_R[idx])
    g = _PLANCK_G[idx] + alpha * (_PLANCK_G[idx + 1] - _PLANCK_G[idx])
    b = _PLANCK_B[idx] + alpha * (_PLANCK_B[idx + 1] - _PLANCK_B[idx])

    out[0] = r if r < 1.0 else 1.0
    out[1] = g if g < 1.0 else 1.0
    out[2] = b if b < 1.0 else 1.0


@njit(cache=True)
def _disk_colour(hit_radius, hit_phi, hv0, hv1, hv2, weight, out):
    T_base  = _novikov_thorne(hit_radius)
    delta   = _doppler_factor(hit_radius, hit_phi, hv0, hv1, hv2)
    g_shift = _grav_redshift(hit_radius)

    T_eff = T_base * delta * g_shift
    if T_eff > 1.0: T_eff = 1.0
    if T_eff < 0.0: T_eff = 0.0

    combined = (delta * g_shift) ** 4
    if combined > 16.0: combined = 16.0

    # Smooth physical intensity remapping
    intensity = T_base * (combined ** 0.5) * 0.95 * weight

    _blackbody_rgb(T_eff, out)
    cap = 2.0
    out[0] *= intensity if intensity < cap else cap
    out[1] *= intensity if intensity < cap else cap
    out[2] *= intensity if intensity < cap else cap


@njit(cache=True)
def _star_colour(ray_dir, star_dirs, star_bright, star_cos_radii, star_colour, star_pal, out):
    rn = (ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2) ** 0.5
    if rn < 1e-12:
        out[0] = out[1] = out[2] = 0.0
        return
    rx = ray_dir[0] / rn
    ry = ray_dir[1] / rn
    rz = ray_dir[2] / rn

    best_b   = -1.0
    best_idx = -1
    n = star_dirs.shape[0]
    
    # TRIG ELIMINATION: Fully vectorized scalar comparisons instead of np.arccos
    for i in range(n):
        dot = star_dirs[i, 0]*rx + star_dirs[i, 1]*ry + star_dirs[i, 2]*rz
        if dot > star_cos_radii[i] and star_bright[i] > best_b:
            best_b   = star_bright[i]
            best_idx = i

    if best_idx < 0:
        out[0] = out[1] = out[2] = 0.0
    else:
        sc = star_colour[best_idx]
        out[0] = star_pal[sc, 0] * best_b
        out[1] = star_pal[sc, 1] * best_b
        out[2] = star_pal[sc, 2] * best_b


@njit(cache=True)
def _volumetric_glow(path, n_steps, out):
    SCALE_H       = RS * 0.35
    GLOW_INNER_SQ = (DISK_INNER * 0.85) ** 2
    GLOW_OUTER_SQ = (DISK_OUTER * 1.2) ** 2
    tmp = np.zeros(3)

    for i in range(0, n_steps, 4):
        p   = path[i]
        
        r_c_sq = p[0]**2 + p[2]**2
        if r_c_sq < GLOW_INNER_SQ or r_c_sq > GLOW_OUTER_SQ:
            continue

        r_c = r_c_sq ** 0.5
        y   = abs(p[1])
        vert = np.exp(-0.5 * (y / SCALE_H) ** 2)
        if vert < 0.008:
            continue

        T_base    = _novikov_thorne(r_c)
        phi       = np.arctan2(p[2], p[0])
        beta      = _keplerian_beta(r_c)
        gamma_lor = 1.0 / (1.0 - beta * beta) ** 0.5
        gx        = -np.sin(phi)
        gz        =  np.cos(phi)

        if i + 1 < n_steps:
            px = path[i+1, 0] - p[0]
            py = path[i+1, 1] - p[1]
            pz = path[i+1, 2] - p[2]
        else:
            px = p[0] - path[i-1, 0]
            py = p[1] - path[i-1, 1]
            pz = p[2] - path[i-1, 2]

        pnorm = (px**2 + py**2 + pz**2) ** 0.5
        if pnorm > 0:
            px /= pnorm; py /= pnorm; pz /= pnorm

        cos_ang   = gx * (-px) + gz * (-pz)
        delta_vol = 1.0 / (gamma_lor * (1.0 - beta * cos_ang))
        g_shift   = _grav_redshift(r_c)

        T_eff = T_base * delta_vol * g_shift
        if T_eff > 1.0: T_eff = 1.0
        if T_eff < 0.0: T_eff = 0.0

        step_i = T_base * (delta_vol * g_shift) ** 2 * vert
        _blackbody_rgb(T_eff, tmp)
        
        # Compensate for striding stride factor (multiply by 4)
        out[0] += tmp[0] * step_i * 0.008 * 4.0
        out[1] += tmp[1] * step_i * 0.008 * 4.0
        out[2] += tmp[2] * step_i * 0.008 * 4.0

    # HUE-PRESERVING CLAMPING: Prevents channels from flat-clipping into dead yellow-greys
    glow_max = max(out[0], max(out[1], out[2]))
    if glow_max > 0.45:
        scale = 0.45 / glow_max
        out[0] *= scale
        out[1] *= scale
        out[2] *= scale


# ── Parallel Pixel Renderer with Absorption Blending ─────────────────────────

@njit(parallel=True, cache=True)
def render_pixel_batch(ray_dirs, cam_pos,
                       star_dirs, star_bright, star_cos_radii, star_colour, star_pal,
                       image, width, height):
    for idx in prange(height * width):
        y = idx // width
        x = idx  %  width

        pos0 = cam_pos.copy()
        vel0 = ray_dirs[y, x].copy()

        # Run Physics Geodesics Integrator
        path, steps_taken, captured, hit_count, hit_radii, hit_phis, hit_vels, _ = integrate_path(
    cam_pos, ray_dirs[y, x], dt=0.1, max_steps=1500
        )

        pixel = np.zeros(3)
        tmp   = np.zeros(3)

        if captured:
            if hit_count > 0:
                for k in range(int(hit_count)):
                    w  = _HIT_W[k] if k < 4 else 0.01
                    hv = hit_vels[k]
                    _disk_colour(hit_radii[k], hit_phis[k], hv[0], hv[1], hv[2], w, tmp)
                    cap = 3.0
                    pixel[0] += tmp[0] if tmp[0] < cap else cap
                    pixel[1] += tmp[1] if tmp[1] < cap else cap
                    pixel[2] += tmp[2] if tmp[2] < cap else cap

        else: # Escaped Rays
            # 1. Accumulate Disk Emission
            if hit_count > 0:
                for k in range(int(hit_count)):
                    w  = _HIT_W[k] if k < 4 else 0.01
                    hv = hit_vels[k]
                    _disk_colour(hit_radii[k], hit_phis[k], hv[0], hv[1], hv[2], w, tmp)
                    cap = 3.0
                    pixel[0] += tmp[0] if tmp[0] < cap else cap
                    pixel[1] += tmp[1] if tmp[1] < cap else cap
                    pixel[2] += tmp[2] if tmp[2] < cap else cap

            # 2. Get Background Features (Stars + Atmospheric Coronal Glow)
            bg_color = np.zeros(3)
            n_steps = path.shape[0]
            if n_steps > 1:
                fd = path[n_steps-1] - path[n_steps-2]
            else:
                fd = vel0.copy()

            _star_colour(fd, star_dirs, star_bright, star_cos_radii, star_colour, star_pal, tmp)
            bg_color[0] = tmp[0]
            bg_color[1] = tmp[1]
            bg_color[2] = tmp[2]

            _volumetric_glow(path, steps_taken, bg_color)

            # Absorption Blending: The disk attenuates background light
            transmission = 0.10 ** hit_count if hit_count > 0 else 1.0
            
            pixel[0] += bg_color[0] * transmission
            pixel[1] += bg_color[1] * transmission
            pixel[2] += bg_color[2] * transmission

        image[y, x, 0] = pixel[0]
        image[y, x, 1] = pixel[1]
        image[y, x, 2] = pixel[2]


# ── Tone Mapping ──────────────────────────────────────────────────────────────

def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)


# ── Main Control ──────────────────────────────────────────────────────────────

def render():
    WIDTH   = 960
    HEIGHT  = 540
    FOV     = 100
    ROLL   = -14.0
    CAM_POS = np.array([37.5, 0.4, 18.00], dtype=np.float64)
    LOOK_AT = [-3.0, -1.0, 0.0]

    print(f"📷  Camera {WIDTH}×{HEIGHT}  |  Rs={RS:.4f}  R_ISCO={R_ISCO:.4f}")
    ray_dirs = generate_camera_rays(WIDTH, HEIGHT, FOV, list(CAM_POS), LOOK_AT, roll_degrees=ROLL)
    image    = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    print("🔥  Warming up Numba JIT …")
    _d_img = np.zeros((2, 2, 3), dtype=np.float64)
    _d_ray = ray_dirs[:2, :2, :].copy()
    render_pixel_batch(
        _d_ray, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _d_img, 2, 2
    )
    print("✅  JIT warm. Rendering …")

    t0 = time.time()
    render_pixel_batch(
        ray_dirs, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        image, WIDTH, HEIGHT
    )
    elapsed = time.time() - t0
    print(f"✅  Done in {elapsed:.1f}s")

    image = aces_tonemap(image)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.imshow(image, origin='upper')
    ax.axis('off')
    ax.set_title(
        f"Relativistic Accretion Disk (Parallel Multi-Hit Engine)\n"
        f"{WIDTH}×{HEIGHT} px | {elapsed:.1f}s render time",
        color='white', fontsize=11, pad=10
    )
    plt.tight_layout()
    out = "accretion_disk_v6.8.12 (a=0.998).png"
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor='black')
    print(f"💾  Saved → {out}")
    plt.show()


if __name__ == "__main__":
    render()