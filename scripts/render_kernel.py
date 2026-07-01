"""
render_kernel.py — Production Kerr Black Hole Renderer
Parallel multi-hit engine with physically accurate Kerr geodesics.

Uses integrate_path_lean for zero-path-allocation parallel rendering.
termination_reason is now used to correctly handle photon-sphere-trapped
rays (reason 5) — these are NOT sampled from the star field.
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "12"  # set to your actual core count

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import json

from core.camera    import generate_camera_rays
from core.geodesics import integrate_path_lean
from core.constants import (MASS, SPIN, R_OUTER_HORIZON,
                             DISK_INNER, DISK_OUTER, SIM_BOUNDS, RS, C)

# ── Derived constants (Python level — fed as runtime args to Numba) ───────────
M      = MASS
R_ISCO = DISK_INNER   # already correctly computed for Kerr in constants.py

# ── Planck locus keyframes ────────────────────────────────────────────────────
_PT = np.array([0.00, 0.20, 0.45, 0.65, 0.82, 1.00], dtype=np.float64)
_PR = np.array([0.75, 1.00, 1.00, 1.00, 0.98, 0.70], dtype=np.float64)
_PG = np.array([0.03, 0.32, 0.55, 0.82, 0.95, 0.82], dtype=np.float64)
_PB = np.array([0.00, 0.00, 0.05, 0.25, 0.88, 1.00], dtype=np.float64)

# ── Hit weights (primary, secondary, tertiary, quaternary) ────────────────────
_HIT_W = np.array([1.0, 0.22, 0.07, 0.02], dtype=np.float64)

# ── Star field ────────────────────────────────────────────────────────────────
_RNG         = np.random.default_rng(42)
_N_STARS     = 3000
_STAR_DIRS   = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
_STAR_DIRS  /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)
_STAR_BRIGHT = (_RNG.power(0.3, _N_STARS) * 0.85 + 0.15).astype(np.float64)
_STAR_RADII  = np.clip(_RNG.exponential(0.0015, _N_STARS), 0.0008, 0.006).astype(np.float64)
_STAR_COS_RADII = np.cos(_STAR_RADII).astype(np.float64)
_STAR_COLOUR = _RNG.choice(
    np.array([0, 1, 2, 3], dtype=np.int64), size=_N_STARS,
    p=[0.05, 0.30, 0.35, 0.30]
).astype(np.int64)
_STAR_PAL = np.array([
    [0.70, 0.82, 1.00],   # O/B blue-white
    [0.97, 0.97, 1.00],   # F/G near-white
    [1.00, 0.78, 0.42],   # K   orange
    [1.00, 0.40, 0.18],   # M   red-orange
], dtype=np.float64)


# ── Numba shader helpers ──────────────────────────────────────────────────────

@njit(cache=True)
def _keplerian_beta(r, mass, r_isco):
    r_s  = r if r > r_isco else r_isco
    beta = (mass / r_s) ** 0.5
    return beta if beta < 0.999 else 0.999


@njit(cache=True)
def _doppler_factor(r, phi, hv0, hv1, hv2, mass, r_isco):
    beta      = _keplerian_beta(r, mass, r_isco)
    gamma     = 1.0 / (1.0 - beta * beta) ** 0.5
    gx        = -np.sin(phi)
    gz        =  np.cos(phi)
    cos_angle = gx * (-hv0) + gz * (-hv2)
    denom     = gamma * (1.0 - beta * cos_angle)
    if abs(denom) < 1e-9:
        return 1.0
    return 1.0 / denom


@njit(cache=True)
def _grav_redshift(r, r_isco, rs):
    r_s = r if r > r_isco * 1.001 else r_isco * 1.001
    val = 1.0 - rs / r_s
    return val ** 0.5 if val > 0.0 else 0.0


@njit(cache=True)
def _novikov_thorne(r, r_isco):
    r_s   = r if r > r_isco * 1.001 else r_isco * 1.001
    nt    = (r_s / r_isco) ** (-0.75) * max((1.0 - (r_isco / r_s) ** 0.5) ** 0.25, 0.0)
    decay = (r_isco / r_s) ** 0.5
    raw   = (nt / 0.38) * decay
    return raw if raw < 1.0 else 1.0


@njit(cache=True)
def _blackbody_rgb(T_eff, out, PT, PR, PG, PB):
    t = T_eff
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    idx = 0
    if   t >= PT[4]: idx = 4
    elif t >= PT[3]: idx = 3
    elif t >= PT[2]: idx = 2
    elif t >= PT[1]: idx = 1
    t0    = PT[idx];  t1 = PT[idx + 1]
    alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0.0 else 0.0
    out[0] = PR[idx] + alpha * (PR[idx+1] - PR[idx])
    out[1] = PG[idx] + alpha * (PG[idx+1] - PG[idx])
    out[2] = PB[idx] + alpha * (PB[idx+1] - PB[idx])
    if out[0] > 1.0: out[0] = 1.0
    if out[1] > 1.0: out[1] = 1.0
    if out[2] > 1.0: out[2] = 1.0


@njit(cache=True)
def _disk_colour(hit_radius, hit_phi, hv0, hv1, hv2, weight,
                 mass, r_isco, rs, out, PT, PR, PG, PB):
    T_base  = _novikov_thorne(hit_radius, r_isco)
    delta   = _doppler_factor(hit_radius, hit_phi, hv0, hv1, hv2, mass, r_isco)
    g_shift = _grav_redshift(hit_radius, r_isco, rs)

    T_eff = T_base * delta * g_shift
    if T_eff > 1.0: T_eff = 1.0
    if T_eff < 0.0: T_eff = 0.0

    combined = (delta * g_shift) ** 4
    if combined > 16.0: combined = 16.0

    intensity = T_base * (combined ** 0.5) * 0.65 * weight

    _blackbody_rgb(T_eff, out, PT, PR, PG, PB)
    cap = 2.0
    out[0] *= intensity if intensity < cap else cap
    out[1] *= intensity if intensity < cap else cap
    out[2] *= intensity if intensity < cap else cap


@njit(cache=True)
def _star_colour(ray_dir, star_dirs, star_bright, star_cos_radii,
                 star_colour, star_pal, out):
    rn = (ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2) ** 0.5
    if rn < 1e-12:
        out[0] = out[1] = out[2] = 0.0
        return
    rx = ray_dir[0]/rn;  ry = ray_dir[1]/rn;  rz = ray_dir[2]/rn
    best_b = -1.0;  best_idx = -1
    for i in range(star_dirs.shape[0]):
        dot = star_dirs[i,0]*rx + star_dirs[i,1]*ry + star_dirs[i,2]*rz
        if dot > star_cos_radii[i] and star_bright[i] > best_b:
            best_b = star_bright[i];  best_idx = i
    if best_idx < 0:
        out[0] = out[1] = out[2] = 0.0
    else:
        sc = star_colour[best_idx]
        out[0] = star_pal[sc, 0] * best_b
        out[1] = star_pal[sc, 1] * best_b
        out[2] = star_pal[sc, 2] * best_b


@njit(cache=True)
def _volumetric_glow(path, steps_taken, out,
                     mass, r_isco, rs, disk_inner, disk_outer,
                     PT, PR, PG, PB):
    SCALE_H    = rs * 0.35
    GLOW_INNER = disk_inner * 0.85
    GLOW_OUTER = disk_outer * 1.2
    tmp = np.zeros(3)

    for i in range(0, steps_taken, 4):   # stride-4 for performance
        p   = path[i]
        y   = abs(p[1])
        r_c = (p[0]**2 + p[2]**2) ** 0.5

        if r_c < GLOW_INNER or r_c > GLOW_OUTER:
            continue

        vert = np.exp(-0.5 * (y / SCALE_H) ** 2)
        if vert < 0.008:
            continue

        T_base    = _novikov_thorne(r_c, r_isco)
        phi       = np.arctan2(p[2], p[0])
        beta      = _keplerian_beta(r_c, mass, r_isco)
        gamma_lor = 1.0 / (1.0 - beta * beta) ** 0.5
        gx        = -np.sin(phi)
        gz        =  np.cos(phi)

        if i + 1 < steps_taken:
            px_ = path[i+1, 0] - p[0]
            py_ = path[i+1, 1] - p[1]
            pz_ = path[i+1, 2] - p[2]
        else:
            px_ = p[0] - path[i-1, 0]
            py_ = p[1] - path[i-1, 1]
            pz_ = p[2] - path[i-1, 2]

        pnorm = (px_**2 + py_**2 + pz_**2) ** 0.5
        if pnorm > 0:
            px_ /= pnorm; py_ /= pnorm; pz_ /= pnorm

        cos_ang   = gx * (-px_) + gz * (-pz_)
        delta_vol = 1.0 / (gamma_lor * (1.0 - beta * cos_ang))
        g_shift   = _grav_redshift(r_c, r_isco, rs)

        T_eff = T_base * delta_vol * g_shift
        if T_eff > 1.0: T_eff = 1.0
        if T_eff < 0.0: T_eff = 0.0

        step_i = T_base * (delta_vol * g_shift) ** 2 * vert
        _blackbody_rgb(T_eff, tmp, PT, PR, PG, PB)
        out[0] += tmp[0] * step_i * 0.010
        out[1] += tmp[1] * step_i * 0.010
        out[2] += tmp[2] * step_i * 0.010

    if out[0] > 0.5: out[0] = 0.5
    if out[1] > 0.5: out[1] = 0.5
    if out[2] > 0.5: out[2] = 0.5


def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)


# ── Parallel pixel batch ──────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def render_pixel_batch(ray_dirs, cam_pos,
                       star_dirs, star_bright, star_cos_radii, star_colour, star_pal,
                       PT, PR, PG, PB,
                       image, width, height,
                       mass, spin, r_outer_horizon,
                       disk_inner, disk_outer, sim_bounds,
                       rs, r_isco, hit_w):
    for idx in prange(height * width):
        y = idx // width
        x = idx  %  width

        pos0 = cam_pos.copy()
        vel0 = ray_dirs[y, x].copy()

        final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, term_reason = \
            integrate_path_lean(
                pos0, vel0, 0.1, 1500,
                mass, spin, r_outer_horizon,
                disk_inner, disk_outer, sim_bounds
            )

        pixel = np.zeros(3)
        tmp   = np.zeros(3)

        if captured:
            # ── Event horizon — black ─────────────────────────────────────
            # Still render any disk hits that occurred before capture
            for k in range(int(hit_count)):
                w  = hit_w[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w,
                             mass, r_isco, rs, tmp, PT, PR, PG, PB)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

        elif term_reason == 5:
            # ── Photon-sphere trapped — near-black, no star sampling ──────
            # These rays orbit the photon sphere until budget exhaustion.
            # They are NOT escaped — sampling the star field from their
            # last velocity direction produces spurious stars in the shadow.
            # Render as a very faint reddish tint (unresolved photon ring
            # contribution) rather than pure black or full star brightness.
            pixel[0] = 0.008
            pixel[1] = 0.002
            pixel[2] = 0.002

        else:
            # ── Genuine escape (reason 3) or far-out budget (reason 4) ───
            # Disk emission
            transmission = 1.0
            for k in range(int(hit_count)):
                w  = hit_w[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w,
                             mass, r_isco, rs, tmp, PT, PR, PG, PB)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

                # Optically thick attenuation
                transmission *= 0.40

            # Star field — only sampled for genuine escaped rays
            _star_colour(final_dir, star_dirs, star_bright, star_cos_radii,
                         star_colour, star_pal, tmp)
            pixel[0] += tmp[0] * transmission
            pixel[1] += tmp[1] * transmission
            pixel[2] += tmp[2] * transmission

        image[y, x, 0] = pixel[0]
        image[y, x, 1] = pixel[1]
        image[y, x, 2] = pixel[2]


# ── Main ──────────────────────────────────────────────────────────────────────

def render():
    WIDTH   = 960
    HEIGHT  = 540
    FOV     = 100.0
    ROLL    = -14.0
    CAM_POS = np.array([6.5, 0.4, 18.0], dtype=np.float64)
    LOOK_AT = [-3.0, -1.0, 0.0]

    print(f"📷  Camera {WIDTH}×{HEIGHT}  |  Rs={RS:.4f}  R_ISCO={R_ISCO:.4f}")
    print(f"    SPIN={SPIN:.4f}  R_horizon={R_OUTER_HORIZON:.4f}")
    print(f"    Disk [{DISK_INNER:.4f} → {DISK_OUTER:.4f}]")

    ray_dirs = generate_camera_rays(
        WIDTH, HEIGHT, FOV, list(CAM_POS), LOOK_AT, roll_degrees=ROLL
    )
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    print("🔥  Warming up Numba JIT …")
    _d_img = np.zeros((2, 2, 3), dtype=np.float64)
    _d_ray = ray_dirs[:2, :2, :].copy()
    render_pixel_batch(
        _d_ray, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB,
        _d_img, 2, 2,
        MASS, SPIN, R_OUTER_HORIZON,
        DISK_INNER, DISK_OUTER, SIM_BOUNDS,
        RS, R_ISCO, _HIT_W
    )
    print("✅  JIT warm. Rendering …")

    t0 = time.time()
    render_pixel_batch(
        ray_dirs, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB,
        image, WIDTH, HEIGHT,
        MASS, SPIN, R_OUTER_HORIZON,
        DISK_INNER, DISK_OUTER, SIM_BOUNDS,
        RS, R_ISCO, _HIT_W
    )
    elapsed = time.time() - t0
    print(f"✅  Done in {elapsed:.1f}s")

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = aces_tonemap(image)

    from datetime import datetime
    from pathlib import Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("accretion_disk_*.png"))
    serial   = len(existing) + 1
    filename = f"accretion_disk_{serial:04d}_a{SPIN:.3f}_{WIDTH}x{HEIGHT}_{timestamp}.png"
    out      = str(output_dir / filename)

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor='black')
    ax.imshow(image, origin='upper')
    ax.axis('off')
    ax.set_title(
        f"Relativistic Accretion Disk (Parallel Multi-Hit Engine)\n"
        f"{WIDTH}×{HEIGHT} px | {elapsed:.1f}s render time",
        color='white', fontsize=11, pad=10
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor='black')
    print(f"💾  Saved → {out}")
    plt.show()
    
    meta = {
    "serial":     serial,
    "timestamp":  timestamp,
    "spin":       float(SPIN),
    "mass":       float(MASS),
    "disk_inner": float(R_ISCO),
    "disk_outer": float(DISK_OUTER),
    "cam_pos":    list(CAM_POS),
    "look_at":    LOOK_AT,
    "fov":        FOV,
    "dt":         0.1,        # whatever you're using
    "max_steps":  1500,
    "width":      WIDTH,
    "height":     HEIGHT,
    "render_time_s": elapsed,
}
    meta_path = out.replace(".png", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📋  Metadata → {meta_path}")


if __name__ == "__main__":
    render()