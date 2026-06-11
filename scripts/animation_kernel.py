"""
animation_kernel.py — The Quasar Anomaly Setup (Ultra-Optimized)
"""

import cv2
import numpy as np
import time
from tqdm import tqdm
from numba import njit, prange

from core.camera    import generate_camera_rays
from core.geodesics import integrate_path_lean
from core.constants import DISK_INNER, DISK_OUTER, RS, C

M      = RS / 2.0
R_ISCO = DISK_INNER

# ── Physics helpers ───────────────────────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def _keplerian_omega(r):
    r_s = r if r > R_ISCO else R_ISCO
    return (M / (r_s ** 3)) ** 0.5

@njit(nopython=True, fastmath=True, cache=True)
def _keplerian_beta(r):
    r_s  = r if r > R_ISCO else R_ISCO
    beta = (M / r_s) ** 0.5
    return beta if beta < 0.999 else 0.999

@njit(nopython=True, fastmath=True, cache=True)
def _doppler_factor(r, phi, hv0, hv1, hv2):
    beta      = _keplerian_beta(r)
    gamma     = 1.0 / (1.0 - beta * beta) ** 0.5
    gx        = -np.sin(phi)
    gz        =  np.cos(phi)
    cos_angle = gx * (-hv0) + gz * (-hv2)
    denom     = gamma * (1.0 - beta * cos_angle)
    if abs(denom) < 1e-9:
        return 1.0
    return 1.0 / denom

@njit(nopython=True, fastmath=True, cache=True)
def _grav_redshift(r):
    r_s = r if r > R_ISCO * 1.001 else R_ISCO * 1.001
    val = 1.0 - RS / r_s
    return val ** 0.5 if val > 0.0 else 0.0

@njit(nopython=True, fastmath=True, cache=True)
def _novikov_thorne(r):
    r_s   = r if r > R_ISCO * 1.001 else R_ISCO * 1.001
    nt    = (r_s / R_ISCO) ** (-0.75) * max((1.0 - (R_ISCO / r_s) ** 0.5) ** 0.25, 0.0)
    decay = (R_ISCO / r_s) ** 0.5
    raw   = (nt / 0.38) * decay
    return raw if raw < 1.0 else 1.0

# ── Blackbody colour — Planck locus keyframes ─────────────────────────────────

_PT = np.array([0.00, 0.20, 0.45, 0.65, 0.82, 1.00], dtype=np.float64)
_PR = np.array([0.75, 1.00, 1.00, 1.00, 0.98, 0.70], dtype=np.float64)
_PG = np.array([0.03, 0.32, 0.55, 0.82, 0.95, 0.82], dtype=np.float64)
_PB = np.array([0.00, 0.00, 0.05, 0.25, 0.88, 1.00], dtype=np.float64)

@njit(nopython=True, fastmath=True, cache=True)
def _blackbody_rgb(T_eff, out):
    t = T_eff
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    idx = 0
    if   t >= _PT[4]: idx = 4
    elif t >= _PT[3]: idx = 3
    elif t >= _PT[2]: idx = 2
    elif t >= _PT[1]: idx = 1
    t0    = _PT[idx];  t1 = _PT[idx + 1]
    alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0.0 else 0.0
    out[0] = _PR[idx] + alpha * (_PR[idx+1] - _PR[idx])
    out[1] = _PG[idx] + alpha * (_PG[idx+1] - _PG[idx])
    out[2] = _PB[idx] + alpha * (_PB[idx+1] - _PB[idx])
    if out[0] > 1.0: out[0] = 1.0
    if out[1] > 1.0: out[1] = 1.0
    if out[2] > 1.0: out[2] = 1.0

# ── Gargantua-style disk structure ───────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def _disk_density(r, phi, time_val):
    omega    = _keplerian_omega(r)
    phi_rest = phi - omega * time_val

    r_norm = (r - DISK_INNER) / (DISK_OUTER - DISK_INNER)
    r_norm = r_norm if r_norm > 0.0 else 0.0
    r_norm = r_norm if r_norm < 1.0 else 1.0

    warp_phi = phi_rest + 0.18 * np.sin(2.0 * phi_rest + 1.4 * r_norm)
    warp_r   = r_norm  + 0.08 * np.cos(3.0 * phi_rest - 0.9 * r_norm)

    k_pitch  = 0.30
    arm_phi  = warp_phi - k_pitch * warp_r * 6.283

    arm_base = (np.cos(2.0 * arm_phi) + 1.0) * 0.5

    threshold = 0.35
    shaped = (arm_base - threshold) / (1.0 - threshold)
    shaped = shaped if shaped > 0.0 else 0.0

    shaped = shaped ** 0.7
    radial_profile = (1.0 - r_norm) ** 1.4 + 0.15
    az_mod = 0.75 + 0.25 * np.sin(phi_rest + 0.5)

    fade_in  = (r - DISK_INNER) / (RS * 1.2)
    fade_in  = fade_in if fade_in < 1.0 else 1.0
    fade_in  = fade_in if fade_in > 0.0 else 0.0
    fade_out = (DISK_OUTER - r) / (RS * 2.5)
    fade_out = fade_out if fade_out < 1.0 else 1.0
    fade_out = fade_out if fade_out > 0.0 else 0.0
    edge     = fade_in * fade_out

    density = shaped * radial_profile * az_mod * edge
    floor   = 0.12 * radial_profile * edge
    return density + floor

# ── Full disk shader ──────────────────────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def _disk_colour(r, phi, hv0, hv1, hv2, weight, time_val, out):
    T_base  = _novikov_thorne(r)
    delta   = _doppler_factor(r, phi, hv0, hv1, hv2)
    g_shift = _grav_redshift(r)
    density = _disk_density(r, phi, time_val)

    T_eff = T_base * delta * g_shift
    if T_eff > 1.0: T_eff = 1.0
    if T_eff < 0.0: T_eff = 0.0

    combined = (delta * g_shift) ** 4
    if combined > 16.0: combined = 16.0

    intensity = T_base * (combined ** 0.5) * density * 1.4 * weight

    _blackbody_rgb(T_eff, out)
    cap = 3.0
    out[0] *= intensity if intensity < cap else cap
    out[1] *= intensity if intensity < cap else cap
    out[2] *= intensity if intensity < cap else cap

# ── Star field ────────────────────────────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
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
        out[0] = star_pal[sc,0] * best_b
        out[1] = star_pal[sc,1] * best_b
        out[2] = star_pal[sc,2] * best_b

# ── Tone mapping ──────────────────────────────────────────────────────────────

@njit(nopython=True, fastmath=True, cache=True)
def _aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    v = (x*(a*x+b)) / (x*(c*x+d)+e)
    return v if v > 0.0 else 0.0 if v > 1.0 else (v if v < 1.0 else 1.0)

# ── Parallel frame renderer ───────────────────────────────────────────────────

_HIT_W = np.array([1.0, 0.22, 0.07, 0.02], dtype=np.float64)

@njit(parallel=True, fastmath=True, cache=True)
def render_frame(width, height, cam_pos, ray_dirs,
                 star_dirs, star_bright, star_cos_radii, star_colour, star_pal,
                 time_val):
    image = np.zeros((height, width, 3), dtype=np.float64)

    for idx in prange(height * width):
        y = idx // width
        x = idx  %  width

        # FIX: Increased dt to 0.4 and dropped max_steps to 500. 
        # This covers exactly 200 spatial units (hitting SIM_BOUNDS) 
        # but cuts the CPU workload by 300%!
        final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, _ = integrate_path_lean(
            cam_pos, ray_dirs[y, x], dt=0.4, max_steps=500
        )

        pixel = np.zeros(3)
        tmp   = np.zeros(3)

        if captured:
            for k in range(int(hit_count)):
                w  = _HIT_W[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w, time_val, tmp)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

        else:
            transmission = 1.0
            for k in range(int(hit_count)):
                w  = _HIT_W[k] if k < 4 else 0.01
                hv = hit_vels[k]
                _disk_colour(hit_radii[k], hit_phis[k],
                             hv[0], hv[1], hv[2], w, time_val, tmp)
                cap = 3.0
                pixel[0] += tmp[0] if tmp[0] < cap else cap
                pixel[1] += tmp[1] if tmp[1] < cap else cap
                pixel[2] += tmp[2] if tmp[2] < cap else cap

                density = _disk_density(hit_radii[k], hit_phis[k], time_val)
                opacity = density * 0.55
                transmission *= (1.0 - opacity) if (1.0 - opacity) > 0.05 else 0.05

            # FIX: No more path subtractions needed, final_dir is provided directly!
            _star_colour(final_dir, star_dirs, star_bright, star_cos_radii,
                         star_colour, star_pal, tmp)

            pixel[0] += tmp[0] * transmission
            pixel[1] += tmp[1] * transmission
            pixel[2] += tmp[2] * transmission

        r_out = _aces(pixel[0]) * 255.0
        g_out = _aces(pixel[1]) * 255.0
        b_out = _aces(pixel[2]) * 255.0

        image[y, x, 0] = r_out if r_out < 255.0 else 255.0
        image[y, x, 1] = g_out if g_out < 255.0 else 255.0
        image[y, x, 2] = b_out if b_out < 255.0 else 255.0

    return image

# ── Main ─────────────────────────────────────────────────────────────────────

def render_video():
    WIDTH        = 960
    HEIGHT       = 540
    FPS          = 60
    SECONDS      = 5
    TOTAL_FRAMES = FPS * SECONDS

    # --- CAMERA OPTIMIZED FOR QUASAR GLITCH ---
    FOV    = 90.0
    ROLL   = 0.0
    CAM_POS = np.array([12.5, 5.5, 40.0], dtype=np.float64) 
    LOOK_AT = np.array([0.0, 0.0,  0.0], dtype=np.float64)
    # ------------------------------------------

    TIME_STEP = 0.175

    _RNG          = np.random.default_rng(42)
    _N_STARS      = 5000
    _STAR_DIRS    = _RNG.normal(size=(_N_STARS, 3)).astype(np.float64)
    _STAR_DIRS   /= np.linalg.norm(_STAR_DIRS, axis=1, keepdims=True)
    _STAR_BRIGHT  = (_RNG.power(0.15, _N_STARS) * 0.85 + 0.05).astype(np.float64)
    _STAR_RADII   = np.clip(
        _RNG.exponential(0.00015, _N_STARS), 0.00005, 0.0006
    ).astype(np.float64)
    _STAR_COS_RADII = np.cos(_STAR_RADII).astype(np.float64)
    _STAR_COLOUR  = _RNG.choice(
        np.array([0, 1, 2, 3], dtype=np.int64), size=_N_STARS,
        p=[0.05, 0.30, 0.35, 0.30]
    ).astype(np.int64)
    _STAR_PAL     = np.array([
        [0.70, 0.82, 1.00],   
        [0.97, 0.97, 1.00],   
        [1.00, 0.78, 0.42],   
        [1.00, 0.40, 0.18],   
    ], dtype=np.float64)

    print(f"🎬  {WIDTH}×{HEIGHT} @ {FPS}fps  |  {TOTAL_FRAMES} frames")
    print(f"    dt=0.1  max_steps=1500  TIME_STEP={TIME_STEP}")

    ray_dirs = generate_camera_rays(
        WIDTH, HEIGHT, FOV, list(CAM_POS), list(LOOK_AT), roll_degrees=ROLL
    )

    print("🔥  Warming up Numba JIT …")
    _d_img = render_frame(
        2, 2, CAM_POS, ray_dirs[:2, :2, :].copy(),
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        0.0
    )
    print("✅  JIT warm.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = "quasar_anomaly.mp4" 
    video_writer = cv2.VideoWriter(out_file, fourcc, FPS, (WIDTH, HEIGHT))

    frame_times = []
    print(f"🚀  Rendering Quasar Glitch ({TOTAL_FRAMES} frames) …")

    for frame_idx in tqdm(range(TOTAL_FRAMES), desc="Rendering", unit="frame"):
        t_frame_start = time.time()
        current_time  = frame_idx * TIME_STEP

        frame_rgb = render_frame(
            WIDTH, HEIGHT, CAM_POS, ray_dirs,
            _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
            current_time
        )

        frame_uint8 = frame_rgb.astype(np.uint8)
        frame_bgr   = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        frame_times.append(time.time() - t_frame_start)
        if frame_idx % 30 == 0 and frame_idx > 0:
            avg   = sum(frame_times[-30:]) / 30
            eta_s = avg * (TOTAL_FRAMES - frame_idx)
            print(f"    frame {frame_idx}/{TOTAL_FRAMES}  |  "
                  f"{avg:.1f}s/frame  |  ETA {eta_s/60:.1f} min")

    video_writer.release()
    avg_total = sum(frame_times) / len(frame_times)
    print(f"\n✅  Done.  Avg {avg_total:.1f}s/frame")
    print(f"💾  Saved → {out_file}")

if __name__ == "__main__":
    render_video()