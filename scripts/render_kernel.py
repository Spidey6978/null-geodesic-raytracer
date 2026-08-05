"""
render_kernel.py — Production Kerr Black Hole Renderer
Parallel multi-hit engine with physically accurate Kerr geodesics.

Uses integrate_path_lean for zero-path-allocation parallel rendering.
termination_reason is now used to correctly handle photon-sphere-trapped
rays (reason 5) — these are NOT sampled from the star field.
"""

import os
import sys
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--spin",       type=float, default=None)
_pre.add_argument("--mass",       type=float, default=None)
_pre.add_argument("--disk-inner", type=float, default=None)
_pre.add_argument("--disk-outer", type=float, default=None)
_pre_args, _ = _pre.parse_known_args()

if _pre_args.spin       is not None: os.environ["BH_SPIN"]       = str(_pre_args.spin)
if _pre_args.mass       is not None: os.environ["BH_MASS"]       = str(_pre_args.mass)
if _pre_args.disk_inner is not None: os.environ["BH_DISK_INNER"] = str(_pre_args.disk_inner)
if _pre_args.disk_outer is not None: os.environ["BH_DISK_OUTER"] = str(_pre_args.disk_outer)

os.environ["NUMBA_NUM_THREADS"] = "12"  # set to your actual core count

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from numba import njit, prange
import json
from pathlib import Path

from core.camera    import generate_camera_rays
from core.geodesics import integrate_path_lean
from core.constants import (MASS, SPIN, R_OUTER_HORIZON,
                             DISK_INNER, DISK_OUTER, SIM_BOUNDS, RS, C)
from scripts.cam_presets import CAMERA_PRESETS

# ── Derived constants (Python level — fed as runtime args to Numba) ───────────
M      = MASS
R_ISCO = DISK_INNER   # already correctly computed for Kerr in constants.py

DEFAULT_RENDER_WIDTH = 960
DEFAULT_RENDER_HEIGHT = 540
DEFAULT_RENDER_DT = 0.2
DEFAULT_RENDER_MAX_STEPS = 5000


def resolve_render_args(args):
    presets = {
        "preview":    dict(width=600,  height=400,  dt=0.1, max_steps=1500),
        "quality":    dict(width=960,  height=540,  dt=0.1, max_steps=5000),
        "production": dict(width=1920, height=1080, dt=0.2, max_steps=8000),
    }

    if args.mode:
        p = presets[args.mode]
        args.width = args.width if args.width is not None else p["width"]
        args.height = args.height if args.height is not None else p["height"]
        args.dt = args.dt if args.dt is not None else p["dt"]
        args.max_steps = args.max_steps if args.max_steps is not None else p["max_steps"]
    else:
        args.width = DEFAULT_RENDER_WIDTH if args.width is None else args.width
        args.height = DEFAULT_RENDER_HEIGHT if args.height is None else args.height
        args.dt = DEFAULT_RENDER_DT if args.dt is None else args.dt
        args.max_steps = DEFAULT_RENDER_MAX_STEPS if args.max_steps is None else args.max_steps

    return args

# ── Planck locus keyframes ────────────────────────────────────────────────────
_PT = np.array([0.00, 0.20, 0.45, 0.65, 0.82, 1.00], dtype=np.float64)
_PR = np.array([0.75, 1.00, 1.00, 1.00, 0.98, 0.70], dtype=np.float64)
_PG = np.array([0.03, 0.32, 0.55, 0.82, 0.95, 0.82], dtype=np.float64)
_PB = np.array([0.00, 0.00, 0.05, 0.25, 0.88, 1.00], dtype=np.float64)

# ── Hit weights (primary, secondary, tertiary, quaternary) ────────────────────
_HIT_W = np.array([1.0, 0.22, 0.07, 0.02], dtype=np.float64)

# ── Star field ────────────────────────────────────────────────────────────────
# Physically motivated stellar population:
#   - Galactic plane concentration (higher density toward a chosen plane)
#   - IMF-weighted spectral types (mostly white/yellow, few blue, rare red)
#   - Logarithmic brightness distribution (many dim, few bright)
#   - Fixed sub-pixel angular size regardless of FOV

_RNG = np.random.default_rng(42)
_N_STARS = 8000

# ── Spectral type palette (O/B, A/F, G, K, M) ────────────────────────────────
# Colors tuned to actual stellar chromaticity, not artistic approximation.
# O/B: ~15000-30000K  A/F: ~7000-10000K  G: ~5500K  K: ~4000K  M: ~3000K
_STAR_PAL = np.array([
    [0.64, 0.75, 1.00],   # O/B  blue-white (hot)
    [0.95, 0.97, 1.00],   # A/F  near-white with faint blue cast
    [1.00, 0.97, 0.88],   # G    warm white (sun-like)
    [1.00, 0.82, 0.60],   # K    pale orange
    [1.00, 0.54, 0.30],   # M    deep orange-red (rare, dim)
], dtype=np.float64)

# Spectral type probabilities weighted by visual contribution to night sky
# (not raw stellar counts — M dwarfs dominate by number but not by visibility)
_STAR_COLOUR = _RNG.choice(
    np.array([0, 1, 2, 3, 4], dtype=np.int64), size=_N_STARS,
    p=[0.06, 0.22, 0.38, 0.24, 0.10]
).astype(np.int64)

# ── Directional distribution with galactic plane concentration ────────────────
# Generate base uniform sphere directions, then weight toward a galactic plane.
# Galactic plane normal chosen arbitrarily — rotate to taste.
_raw_dirs = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
_raw_dirs /= np.linalg.norm(_raw_dirs, axis=1, keepdims=True)

# Galactic plane normal (points "up" out of the plane)
# Tilt it ~60 degrees from world-y so it cuts diagonally across most views
_GAL_NORMAL = np.array([0.5, 0.866, 0.0], dtype=np.float64)
_GAL_NORMAL /= np.linalg.norm(_GAL_NORMAL)

# Galactic latitude of each raw direction (0 = in plane, pi/2 = pole)
_gal_lat = np.abs(np.dot(_raw_dirs, _GAL_NORMAL))  # 0=plane, 1=pole

# Acceptance probability: stars near the plane (low _gal_lat) are more likely
# Use a von Mises-Fisher-like weighting: p = exp(-gal_lat^2 / sigma^2)
_GAL_SIGMA = 0.45   # controls width of galactic band; larger = broader band
_accept_prob = np.exp(-(_gal_lat ** 2) / (_GAL_SIGMA ** 2))
# Also add a uniform floor so off-plane stars still exist
_accept_prob = 0.25 + 0.75 * _accept_prob
_accept_prob /= _accept_prob.max()

# Rejection sample to thin out polar regions
_keep = _RNG.random(_N_STARS) < _accept_prob
# Pad with uniform stars if rejection removed too many
_n_kept = _keep.sum()
if _n_kept < _N_STARS:
    _extra = _RNG.normal(size=(_N_STARS - _n_kept, 3)).astype(np.float64)
    _extra /= np.linalg.norm(_extra, axis=1, keepdims=True)
    _STAR_DIRS = np.vstack([_raw_dirs[_keep], _extra]).astype(np.float64)
else:
    _STAR_DIRS = _raw_dirs[:_N_STARS].astype(np.float64)

# Re-normalize after stacking
_STAR_DIRS /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)

# ── Brightness: logarithmic distribution ─────────────────────────────────────
# power(0.12) gives heavy weight to dim stars with rare bright outliers.
# Floor at 0.04 so dimmest stars are still technically visible.
# Spectral type modulates brightness: hot stars are intrinsically brighter.
_base_bright = (_RNG.power(0.12, _N_STARS) * 0.92 + 0.04).astype(np.float64)

# Spectral brightness modifier: O/B stars intrinsically brighter, M stars dimmer
_SPEC_BRIGHT_MOD = np.array([1.4, 1.1, 1.0, 0.75, 0.45], dtype=np.float64)
_bright_mod = np.array([_SPEC_BRIGHT_MOD[c] for c in _STAR_COLOUR])
_STAR_BRIGHT = np.clip(_base_bright * _bright_mod, 0.0, 1.0).astype(np.float64)

# ── Angular size: fixed sub-pixel, FOV-independent ───────────────────────────
# All stars are point sources at 0.00035 radians (~0.02 degrees).
# This is below 1 pixel at any practical FOV — stars render as single pixels.
# DO NOT scale this by FOV. Lensing distortion comes from position shift
# (via final_dir), not from size inflation.
_STAR_RADIUS   = 0.002  # radians — fixed, FOV-independent
_STAR_COS_RADII = np.full(_N_STARS, np.cos(_STAR_RADIUS), dtype=np.float64)

# ── Large-scale density modulation (breaks uniformity, adds depth feel) ───────
# Low-frequency brightness variation across the sky simulates unresolved
# background structure (distant galaxy clusters, nebulae, dust lanes).
# Applied as a per-star brightness multiplier based on sky position.
_gal_density = 0.7 + 0.3 * np.exp(-(_gal_lat ** 2) / (0.3 ** 2))
_STAR_BRIGHT *= _gal_density
_STAR_BRIGHT = np.clip(_STAR_BRIGHT, 0.0, 1.0).astype(np.float64)

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
def _hash21(x, y):
    n = np.int64(x * 127.1 + y * 311.7)
    n = (n << 13) ^ n
    n = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff
    return float(n) / 2147483647.0


@njit(cache=True)
def _smooth_noise2d(x, y):
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    fx = x - float(xi)
    fy = y - float(yi)

    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    v00 = _hash21(xi, yi)
    v10 = _hash21(xi + 1, yi)
    v01 = _hash21(xi, yi + 1)
    v11 = _hash21(xi + 1, yi + 1)

    v0 = v00 + ux * (v10 - v00)
    v1 = v01 + ux * (v11 - v01)
    return v0 + uy * (v1 - v0)


@njit(cache=True)
def _fbm2d(x, y, octaves=3):
    val = 0.0
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        val += amp * _smooth_noise2d(x * freq, y * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return val / norm if norm > 0.0 else 0.0


@njit(cache=True)
def _kerr_keplerian_omega(r, mass, spin):
    r_safe = max(r, 1e-5)
    sqrt_M = np.sqrt(max(mass, 1e-10))
    denom = r_safe ** 1.5 + spin * sqrt_M
    if abs(denom) < 1e-10:
        return 0.0
    return sqrt_M / denom


@njit(cache=True)
def _disk_colour(hit_radius, hit_phi, hv0, hv1, hv2, weight,
                 mass, r_isco, rs, out, PT, PR, PG, PB,
                 spin=0.998, frame_time=0.0, noise_scale=0.8, contrast=0.65):
    T_base  = _novikov_thorne(hit_radius, r_isco)

    # ── Procedural Plasma Turbulence & Keplerian Advection ─────────
    omega = _kerr_keplerian_omega(hit_radius, mass, spin)
    # Advect angle over time: inner disk rotates faster than outer disk
    phi_adv = hit_phi - omega * frame_time * 8.0

    x_adv = hit_radius * np.cos(phi_adv) * noise_scale
    y_adv = hit_radius * np.sin(phi_adv) * noise_scale

    turb = _fbm2d(x_adv, y_adv, 3)

    # Modulate base Novikov-Thorne temperature profile with turbulence
    T_turb = T_base * (1.0 - contrast + contrast * turb * 2.0)
    if T_turb < 0.0: T_turb = 0.0

    delta   = _doppler_factor(hit_radius, hit_phi, hv0, hv1, hv2, mass, r_isco)
    g_shift = _grav_redshift(hit_radius, r_isco, rs)

    T_eff = T_turb * delta * g_shift
    if T_eff > 1.0: T_eff = 1.0
    if T_eff < 0.0: T_eff = 0.0

    combined = (delta * g_shift) ** 4
    if combined > 16.0: combined = 16.0

    intensity = T_turb * (combined ** 0.5) * 0.65 * weight

    _blackbody_rgb(T_eff, out, PT, PR, PG, PB)
    cap = 2.0
    out[0] *= intensity if intensity < cap else cap
    out[1] *= intensity if intensity < cap else cap
    out[2] *= intensity if intensity < cap else cap



@njit(cache=True)
def _star_colour(ray_dir, star_dirs, star_bright, star_cos_radii,
                 star_colour, star_pal, out):
    # Normalize ray direction
    rn = (ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2) ** 0.5
    if rn < 1e-12:
        out[0] = out[1] = out[2] = 0.0
        return
    rx = ray_dir[0]/rn;  ry = ray_dir[1]/rn;  rz = ray_dir[2]/rn

    # Find the single brightest star within angular radius
    # (point source model — no disc rendering, no size accumulation)
    best_b   = -1.0
    best_idx = -1
    for i in range(star_dirs.shape[0]):
        dot = star_dirs[i,0]*rx + star_dirs[i,1]*ry + star_dirs[i,2]*rz
        if dot < 0.0:
            continue
        if dot > star_cos_radii[i] and star_bright[i] > best_b:
            best_b   = star_bright[i]
            best_idx = i

    if best_idx < 0:
        out[0] = out[1] = out[2] = 0.0
        return

    sc = star_colour[best_idx]

    # Apply brightness with a mild gamma lift so dim stars
    # aren't completely invisible but bright stars still pop
    # gamma = 0.7 compresses the range: dim stars get a slight lift
    b = best_b ** 0.7

    out[0] = star_pal[sc, 0] * b
    out[1] = star_pal[sc, 1] * b
    out[2] = star_pal[sc, 2] * b


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


from core.skybox import sample_skybox_equirectangular

# ── Parallel pixel batch ──────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def render_pixel_batch(ray_dirs, cam_pos,
                       star_dirs, star_bright, star_cos_radii, star_colour, star_pal,
                       PT, PR, PG, PB,
                       image, ray_debug, width, height,
                       mass, spin, r_outer_horizon,
                       disk_inner, disk_outer, sim_bounds,
                       rs, r_isco, hit_w, dt, max_steps,
                       frame_time=0.0,
                       skybox_img=None, sky_w=0, sky_h=0):
    for idx in prange(height * width):
        y = idx // width
        x = idx  %  width

        final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, term_reason, initial_state_code = \
            integrate_path_lean(
                cam_pos, ray_dirs[y, x], dt, max_steps,
                mass, spin, r_outer_horizon,
                disk_inner, disk_outer, sim_bounds
            )

        pixel = np.zeros(3)
        tmp   = np.zeros(3)

        if initial_state_code == 1:
            ray_debug[y, x] = 1
        elif initial_state_code == 2:
            ray_debug[y, x] = 2
        elif initial_state_code == 3:
            ray_debug[y, x] = 3
        elif initial_state_code == 4:
            ray_debug[y, x] = 4
        elif initial_state_code == 5:
            ray_debug[y, x] = 5
        elif initial_state_code == 6:
            ray_debug[y, x] = 6

        if captured:
            # ── Event horizon — black ─────────────────────────────────────
            for k in range(int(hit_count)):
                w  = hit_w[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w,
                             mass, r_isco, rs, tmp, PT, PR, PG, PB,
                             spin, frame_time)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

        elif term_reason == 5:
            # ── Photon-sphere trapped — near-black, no star sampling ──────
            pixel[0] = 0.008
            pixel[1] = 0.002
            pixel[2] = 0.002

        else:
            # ── Genuine escape (reason 3) or far-out budget (reason 4) ───
            transmission = 1.0
            for k in range(int(hit_count)):
                w  = hit_w[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w,
                             mass, r_isco, rs, tmp, PT, PR, PG, PB,
                             spin, frame_time)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

                transmission *= 0.40

            # Skybox / Star field — only sampled for genuine escaped rays
            if sky_w > 0 and skybox_img is not None:
                sky_rgb = sample_skybox_equirectangular(final_dir[0], final_dir[1], final_dir[2], skybox_img, sky_w, sky_h)
                pixel[0] += sky_rgb[0] * transmission
                pixel[1] += sky_rgb[1] * transmission
                pixel[2] += sky_rgb[2] * transmission
            else:
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
    parser = argparse.ArgumentParser(
        description="Production Kerr black hole renderer",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Physics (pre-parsed above, shown here for --help)
    parser.add_argument("--spin",       type=float, default=SPIN,
                        help=f"BH spin parameter (default: {SPIN})")
    parser.add_argument("--mass",       type=float, default=MASS,
                        help=f"BH mass (default: {MASS})")
    parser.add_argument("--disk-inner", type=float, default=None,
                        help="Disk inner edge in units of RS (e.g. 3.0 = 3xRS = 6.0 geom units)")
    parser.add_argument("--disk-outer", type=float, default=None,
                        help="Disk outer edge in units of RS (e.g. 10.0 = 10xRS = 20.0 geom units)")
    parser.add_argument("--preset", type=str, default=None,
                    choices=list(CAMERA_PRESETS.keys()),
                    help="Named camera preset (overrides --cam-pos, --look-at, --fov)")

    # Integration
    parser.add_argument("--dt",        type=float, default=None,
                        help="Base step size (defaults to 0.2; overridden by --mode unless you pass --dt explicitly)")
    parser.add_argument("--max-steps", type=int,   default=None,
                        help="Max integration steps per ray (defaults to 5000; overridden by --mode unless you pass --max-steps explicitly)")

    # Camera
    parser.add_argument("--width",  type=int,   default=None,
                        help="Render width (defaults to 960; overridden by --mode unless you pass --width explicitly)")
    parser.add_argument("--height", type=int,   default=None,
                        help="Render height (defaults to 540; overridden by --mode unless you pass --height explicitly)")
    parser.add_argument("--fov",    type=float, default=100.0)
    parser.add_argument("--roll",   type=float, default=0.0)
    parser.add_argument("--cam-pos", nargs=3, type=float,
                        default=[6.5, 0.4, 18.0], metavar=("X","Y","Z"))
    parser.add_argument("--look-at", nargs=3, type=float,
                        default=[-3.0, -1.0, 0.0], metavar=("X","Y","Z"))

    # Output
    parser.add_argument("--out",  type=str, default=None,
                        help="Output filename (auto-generated if omitted)")
    parser.add_argument("--show", action="store_true",
                        help="Show image interactively after saving")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Frame timestamp for plasma advection & Keplerian rotation")

    # Render mode shortcut
    parser.add_argument("--mode", type=str, default=None,
                        choices=["preview", "quality", "production"],
                        help=(
                            "Preset mode (fills in width/height/dt/max-steps when you don't pass them explicitly):\n"
                            "  preview    : 600x400,  dt=0.1, steps=1500  (~5s)\n"
                            "  quality    : 960x540,  dt=0.1, steps=5000  (~30s)\n"
                            "  production : 1920x1080, dt=0.2, steps=8000 (~4min)"
                        ))

    args = parser.parse_args()
        
    if args.preset:
        p = CAMERA_PRESETS[args.preset]
        args.cam_pos = p["cam_pos"]
        args.look_at = p["look_at"]
        args.fov     = p["fov"]
        if "roll" in p:
            args.roll = p["roll"]
        print(f"📍  Preset '{args.preset}': {p['note']}")

    args = resolve_render_args(args)

    WIDTH     = args.width
    HEIGHT    = args.height
    DT        = args.dt
    MAX_STEPS = args.max_steps
    CAMERA_NAME = args.preset or "custom_camera"
    CAM_POS   = np.array(args.cam_pos, dtype=np.float64)
    LOOK_AT   = args.look_at

    print(f"📷  {WIDTH}×{HEIGHT}  dt={DT}  max_steps={MAX_STEPS}  t={args.time:.2f}")
    print(f"    spin={SPIN:.4f}  M={MASS:.4f}  R_horizon={R_OUTER_HORIZON:.4f}")
    print(f"    R_ISCO={DISK_INNER:.4f}  disk_outer={DISK_OUTER:.4f}")
    print(f"    cam={list(CAM_POS)}  look_at={LOOK_AT}")

    ray_dirs = generate_camera_rays(
        WIDTH, HEIGHT, args.fov, list(CAM_POS), LOOK_AT,
        roll_degrees=args.roll
    )
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    ray_debug = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    print("🔥  Warming up JIT …")
    _d_img = np.zeros((2, 2, 3), dtype=np.float64)
    _d_ray = ray_dirs[:2, :2, :].copy()
    _d_debug = np.zeros((2, 2), dtype=np.uint8)
    render_pixel_batch(
        _d_ray, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB,
        _d_img, _d_debug, 2, 2,
        MASS, SPIN, R_OUTER_HORIZON,
        DISK_INNER, DISK_OUTER, SIM_BOUNDS,
        RS, DISK_INNER, _HIT_W,
        DT, MAX_STEPS,
        args.time
    )
    print("✅  JIT warm. Rendering …")

    t0 = time.time()
    render_pixel_batch(
        ray_dirs, CAM_POS,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB,
        image, ray_debug, WIDTH, HEIGHT,
        MASS, SPIN, R_OUTER_HORIZON,
        DISK_INNER, DISK_OUTER, SIM_BOUNDS,
        RS, DISK_INNER, _HIT_W,
        DT, MAX_STEPS,
        args.time
    )
    elapsed = time.time() - t0
    print(f"✅  Done in {elapsed:.1f}s")

    unique, counts = np.unique(ray_debug, return_counts=True)
    print("🔎  Ray diagnostic counts:")
    print(f"    total rays:       {HEIGHT * WIDTH}")
    print(f"    initial invalid:  {int(counts[unique == 1].sum() if np.any(unique == 1) else 0)}")
    print(f"    initial captured: {int(counts[unique == 2].sum() if np.any(unique == 2) else 0)}")
    print(f"    initial escaped:  {int(counts[unique == 3].sum() if np.any(unique == 3) else 0)}")
    entered_rk4 = (HEIGHT * WIDTH) - (int(counts[unique == 1].sum() if np.any(unique == 1) else 0) + int(counts[unique == 2].sum() if np.any(unique == 2) else 0) + int(counts[unique == 3].sum() if np.any(unique == 3) else 0))
    print(f"    entered RK4:      {entered_rk4}")
    print(f"    captured in RK4:  {int(counts[unique == 4].sum() if np.any(unique == 4) else 0)}")
    print(f"    escaped in RK4:   {int(counts[unique == 5].sum() if np.any(unique == 5) else 0)}")
    print(f"    trapped / budget: {int(counts[unique == 6].sum() if np.any(unique == 6) else 0)}")

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = aces_tonemap(image)

    # ── Output with timestamp + serial + sidecar JSON ────────────────────────
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    serial    = len(list(output_dir.glob("production_render_*"))) + 1
    stem      = f"production_render_{CAMERA_NAME}_{serial:04d}_a{SPIN:.3f}_{WIDTH}x{HEIGHT}_{timestamp}"
    out       = str(output_dir / f"{stem}.png") if args.out is None else args.out

    meta = {
        "serial":        serial,
        "timestamp":     timestamp,
        "spin":          float(SPIN),
        "mass":          float(MASS),
        "r_horizon":     float(R_OUTER_HORIZON),
        "disk_inner":    float(DISK_INNER),
        "disk_outer":    float(DISK_OUTER),
        "cam_pos":       list(CAM_POS),
        "look_at":       LOOK_AT,
        "fov":           args.fov,
        "roll":          args.roll,
        "width":         WIDTH,
        "height":        HEIGHT,
        "dt":            DT,
        "max_steps":     MAX_STEPS,
        "mode":          args.mode,
        "camera_preset": args.preset,
        "render_time_s": elapsed,
        "cli":           " ".join(sys.argv),
    }
    with open(out.replace(".png", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    fig, ax = plt.subplots(figsize=(WIDTH/80, HEIGHT/80), facecolor="black")
    ax.imshow(image, origin="upper")
    ax.axis("off")
    ax.set_title(
        f"Relativistic Accretion Disk — Camera: {CAMERA_NAME}\n"
        f"{WIDTH}×{HEIGHT} | {elapsed:.1f}s | a={SPIN:.3f} | "
        f"dt={DT} | steps={MAX_STEPS}",
        color="white", fontsize=10, pad=8
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150, facecolor="black")
    print(f"💾  Saved → {out}")
    print(f"📋  Metadata → {out.replace('.png', '.json')}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    render()
