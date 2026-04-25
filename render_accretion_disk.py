"""
render_accretion_disk.py  —  v3: Visual Upgrade

New in v3:
  - Secondary (ghost) image: photons that loop ~270 deg around the BH and hit
    the disk from below are rendered at reduced intensity, producing the thin
    bright ring that hugs the shadow — the most iconic feature of the
    Interstellar render.
  - Volumetric disk glow: escaped photons that passed close to the disk plane
    accumulate a soft emission halo proportional to how near they got.
  - Gravitational redshift: photons climbing out of the BH's potential well
    lose energy. Combined with Doppler, the inner disk is redder on the
    far side and blue-white on the near side — physically correct.
  - Procedural star field: escaped photons are mapped to a deterministic star
    distribution so the gravitational lensing of background stars is visible.
  - ACES filmic tone mapping: prevents hard clipping, bright regions bloom
    rather than clip flat white.

Physics references:
  Luminet (1979)  A&A 75, 228
  James et al. (2015)  CQG 32 065001  (the Interstellar paper)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from core.camera    import generate_camera_rays
from core.geodesics import integrate_path
from core.constants import DISK_INNER, DISK_OUTER, RS, C

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

M      = RS / 2.0       # BH mass (geometrized units, G = c = 1)
R_ISCO = 3.0 * RS       # Innermost Stable Circular Orbit

# ─────────────────────────────────────────────────────────────────────────────
# Doppler + gravitational redshift
# ─────────────────────────────────────────────────────────────────────────────

def keplerian_beta(r):
    """Orbital speed as fraction of c at radius r (clamped to ISCO)."""
    r_s  = max(r, R_ISCO)
    beta = np.sqrt(M / r_s)
    return np.clip(beta, 0.0, 0.999)


def doppler_factor(hit_radius, hit_phi, hit_vel):
    """
    Relativistic Doppler beaming factor delta.
    delta > 1 : blueshift (approaching gas, brighter)
    delta < 1 : redshift  (receding gas,   dimmer)
    """
    beta      = keplerian_beta(hit_radius)
    gamma     = 1.0 / np.sqrt(1.0 - beta * beta)
    gas_dir   = np.array([-np.sin(hit_phi), 0.0, np.cos(hit_phi)])
    cos_angle = np.dot(gas_dir, -hit_vel)
    return 1.0 / (gamma * (1.0 - beta * cos_angle))


def grav_redshift_factor(r):
    """
    Gravitational redshift: photon emitted at r arrives at infinity with
    frequency ratio  nu_inf / nu_emit = sqrt(1 - Rs/r).
    """
    r_s = max(r, R_ISCO * 1.001)
    return np.sqrt(max(1.0 - RS / r_s, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Temperature profile + colour mapping
# ─────────────────────────────────────────────────────────────────────────────

def novikov_thorne_temperature(r):
    """
    Novikov-Thorne radial temperature profile (normalised).
    T(r) ∝ r^{-3/4} * (1 - sqrt(R_ISCO/r))^{1/4}
    Returns value in [0, 1] with peak near 1.4 * R_ISCO.
    """
    r_s = max(r, R_ISCO * 1.001)
    nt  = (r_s / R_ISCO) ** (-0.75) * max((1.0 - np.sqrt(R_ISCO / r_s)) ** 0.25, 0.0)
    return np.clip(nt / 0.38, 0.0, 1.0)   # 0.38 ≈ peak value


def blackbody_rgb(T_eff):
    """
    Map effective temperature proxy [0, 1] to RGB.
    0.0  → deep red      (cool outer disk)
    0.5  → orange-gold   (mid disk)
    0.85 → yellow-white  (hot inner disk)
    1.0  → blue-white    (ISCO / extreme blueshift)
    """
    t = np.clip(T_eff, 0.0, 1.0)
    r = np.clip(1.0 - 0.35 * (t - 1.0) ** 2,  0.0, 1.0)
    g = np.clip(t ** 0.55,                      0.0, 1.0)
    b = np.clip((t - 0.82) * 5.5,               0.0, 1.0)
    return np.array([r, g, b])


# ─────────────────────────────────────────────────────────────────────────────
# Procedural star field
# ─────────────────────────────────────────────────────────────────────────────

_RNG         = np.random.default_rng(42)
_N_STARS     = 2500
_STAR_DIRS   = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
_STAR_DIRS  /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)
_STAR_BRIGHT = _RNG.power(0.25, _N_STARS).astype(np.float64)
_STAR_RADII  = np.clip(_RNG.exponential(0.004, _N_STARS), 0.001, 0.025).astype(np.float64)
_STAR_COLOUR = _RNG.integers(0, 3, _N_STARS)

_STAR_PALETTES = np.array([
    [0.85, 0.90, 1.00],   # 0: blue-white
    [1.00, 0.95, 0.80],   # 1: yellow-white
    [1.00, 0.65, 0.40],   # 2: orange-red
], dtype=np.float64)


def star_field_colour(ray_dir):
    """Sample the procedural star field; returns RGB or zeros for empty sky."""
    ray_dir  = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)
    dots     = _STAR_DIRS @ ray_dir
    ang_dist = np.arccos(np.clip(dots, -1.0, 1.0))
    inside   = ang_dist < _STAR_RADII
    if not np.any(inside):
        return np.zeros(3)
    idx  = np.where(inside)[0]
    best = idx[np.argmax(_STAR_BRIGHT[idx])]
    return _STAR_PALETTES[_STAR_COLOUR[best]] * _STAR_BRIGHT[best] * 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Disk pixel shader
# ─────────────────────────────────────────────────────────────────────────────

SECONDARY_SCALE = 0.18   # ghost image fraction of primary intensity


def disk_colour(hit_radius, hit_phi, hit_vel, is_secondary=False):
    """
    Full physically-motivated disk shader including Doppler beaming,
    gravitational redshift, and Novikov-Thorne temperature profile.
    """
    T_base  = novikov_thorne_temperature(hit_radius)
    delta   = doppler_factor(hit_radius, hit_phi, hit_vel)
    g_shift = grav_redshift_factor(hit_radius)

    # Observed intensity: I_obs = (delta * g)^4 * I_emit
    combined  = (delta * g_shift) ** 4
    T_eff     = np.clip(T_base * delta * g_shift, 0.0, 1.0)
    colour    = blackbody_rgb(T_eff)
    intensity = T_base * combined

    if is_secondary:
        intensity *= SECONDARY_SCALE

    return colour * np.clip(intensity, 0.0, 1.8)


# ─────────────────────────────────────────────────────────────────────────────
# Volumetric disk glow (escaped rays that skimmed the disk plane)
# ─────────────────────────────────────────────────────────────────────────────

def volumetric_glow(path):
    """
    Walk the escaped ray's path; accumulate soft emission from gas near the
    equatorial plane, simulating the corona / atmosphere of the disk.
    """
    if len(path) < 2:
        return np.zeros(3)

    glow         = np.zeros(3)
    SCALE_HEIGHT = RS * 0.4
    GLOW_INNER   = DISK_INNER * 0.8
    GLOW_OUTER   = DISK_OUTER * 1.3

    for i in range(len(path)):
        p   = path[i]
        y   = abs(p[1])
        r_c = np.sqrt(p[0]**2 + p[2]**2)

        if r_c < GLOW_INNER or r_c > GLOW_OUTER:
            continue

        vert = np.exp(-0.5 * (y / SCALE_HEIGHT) ** 2)
        if vert < 0.005:
            continue

        T_base    = novikov_thorne_temperature(r_c)
        phi       = np.arctan2(p[2], p[0])
        beta      = keplerian_beta(r_c)
        gamma_lor = 1.0 / np.sqrt(1.0 - beta * beta)
        gas_dir   = np.array([-np.sin(phi), 0.0, np.cos(phi)])

        if i + 1 < len(path):
            pdir = path[i+1] - p
        else:
            pdir = p - path[i-1]
        pnorm = np.linalg.norm(pdir)
        if pnorm > 0:
            pdir /= pnorm

        cos_ang   = np.dot(gas_dir, -pdir)
        delta_vol = 1.0 / (gamma_lor * (1.0 - beta * cos_ang))
        g_shift   = grav_redshift_factor(r_c)
        T_eff     = np.clip(T_base * delta_vol * g_shift, 0.0, 1.0)

        step_i = T_base * (delta_vol * g_shift) ** 2 * vert
        glow  += blackbody_rgb(T_eff) * step_i * 0.012

    return np.clip(glow, 0.0, 0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Tone mapping
# ─────────────────────────────────────────────────────────────────────────────

def aces_tonemap(x):
    """ACES filmic approximation — bright regions bloom rather than clip."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render():
    WIDTH   = 600
    HEIGHT  = 400
    FOV     = 60

    # Low camera angle — shows near disk + lensed far disk simultaneously
    CAM_POS = [0.0, 1.5, 15.0]
    LOOK_AT = [0.0,  0.0,  0.0]

    print(f"📷  Camera {WIDTH}×{HEIGHT}  |  Rs={RS:.4f}  R_ISCO={R_ISCO:.4f}")
    ray_dirs = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    image    = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    print("🚀  Rendering …")
    t0 = time.time()

    for y in range(HEIGHT):
        if y % 40 == 0:
            print(f"    row {y:4d}/{HEIGHT}  ({y/HEIGHT*100:5.1f}%)")

        for x in range(WIDTH):
            pos0 = np.array(CAM_POS, dtype=np.float64)
            vel0 = ray_dirs[y, x]

            path, captured, hit_disk, hit_radius, hit_phi, hit_vel = integrate_path(
                pos0, vel0, dt=0.5, max_steps=2000
            )

            if captured:
                image[y, x] = [0.0, 0.0, 0.0]

            elif hit_disk:
                # Primary disk contribution
                pixel = disk_colour(hit_radius, hit_phi, hit_vel, is_secondary=False)

                # Secondary image: scan the path for a second equatorial
                # crossing that lands inside the disk annulus.
                crossing_count = 0
                for k in range(1, len(path)):
                    old_p = path[k-1]
                    new_p = path[k]
                    if old_p[1] * new_p[1] <= 0.0:
                        crossing_count += 1
                        if crossing_count == 1:
                            continue          # skip the primary crossing
                        dy = new_p[1] - old_p[1]
                        if dy == 0.0:
                            continue
                        t_f = -old_p[1] / dy
                        hx  = old_p[0] + t_f * (new_p[0] - old_p[0])
                        hz  = old_p[2] + t_f * (new_p[2] - old_p[2])
                        r_h = np.sqrt(hx*hx + hz*hz)
                        if DISK_INNER <= r_h <= DISK_OUTER:
                            phi_h = np.arctan2(hz, hx)
                            vh    = new_p - old_p
                            vn    = np.linalg.norm(vh)
                            if vn > 0:
                                vh /= vn
                            pixel += disk_colour(r_h, phi_h, vh, is_secondary=True)
                            break

                image[y, x] = np.clip(pixel, 0.0, 2.0)   # allow HDR before tonemap

            else:
                # Escaped: star field + volumetric glow
                final_dir    = path[-1] - path[-2] if len(path) > 1 else vel0
                image[y, x]  = np.clip(
                    star_field_colour(final_dir) + volumetric_glow(path),
                    0.0, 1.0
                )

    elapsed = time.time() - t0
    print(f"✅  Done in {elapsed:.1f}s")

    # Tone map the full image
    image = aces_tonemap(image)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.imshow(image, origin='upper')
    ax.axis('off')
    ax.set_title(
        f"Relativistic Accretion Disk (Doppler Beamed)\n"
        f"{WIDTH}x{HEIGHT} px | {elapsed:.1f}s render time",
        color='white', fontsize=11, pad=10
    )
    plt.tight_layout()
    out = "accretion_disk_v3.png"
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor='black')
    print(f"💾  Saved → {out}")
    plt.show()


if __name__ == "__main__":
    render()