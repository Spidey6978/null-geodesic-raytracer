"""
Module: core.geodesics
Calculates the path of a photon through curved spacetime.
Optimized with Numba for C-speed execution.

v4: Multi-hit equatorial plane intersection tracking for arbitrary-order lensed rings.
"""
import numpy as np
from numba import njit
from .constants import RS, C, SIM_BOUNDS, DISK_INNER, DISK_OUTER

@njit(nopython=True, cache=True)
def get_acceleration(pos, vel):
    """
    Calculates the relativistic acceleration for a photon in Schwarzschild spacetime.
    Derived from the effective potential: U_eff = h^2 / (2r^2) * (1 - Rs/r)
    The acceleration is: a = -1.5 * Rs * h^2 / r^5 * pos
    """
    r_sq = np.dot(pos, pos)
    if r_sq < RS * RS:
        return np.zeros_like(pos)

    h = np.cross(pos, vel)
    if h.ndim == 0:
        h_val = h.item()
        h2 = h_val * h_val
    else:
        h2 = np.dot(h, h)

    prefactor = -1.5 * RS * h2 / (r_sq * r_sq * r_sq)
    return prefactor * pos


@njit(nopython=True, cache=True)
def integrate_path(start_pos, start_vel, dt=0.5, max_steps=5000):
    """
    Traces a single photon ray using RK4 integration through Schwarzschild spacetime.
    Tracks up to 4 separate equatorial disk crossings (multi-hit tracing).

    Returns:
        path         : np.ndarray, shape (steps, 3) — the full trajectory
        captured     : bool — True if the photon fell inside the event horizon
        hit_count    : int — Number of times the ray crossed the disk plane
        hit_radii    : np.ndarray (shape 4) — cylindrical radius of each crossing
        hit_phis     : np.ndarray (shape 4) — azimuthal angle of each crossing (radians)
        hit_vels     : np.ndarray (shape 4, 3) — photon velocity vector at each crossing
    """
    pos = start_pos.astype(np.float64)
    vel = start_vel.astype(np.float64)

    path = np.zeros((max_steps + 1, len(pos)), dtype=np.float64)
    path[0] = pos

    steps_taken = 0
    captured = False

    # --- 1. CHANGE: Pre-allocate arrays for up to 4 crossings to satisfy Numba typing ---
    hit_count = 0
    hit_radii = np.zeros(4, dtype=np.float64)
    hit_phis = np.zeros(4, dtype=np.float64)
    hit_vels = np.zeros((4, 3), dtype=np.float64)

    for i in range(max_steps):
        steps_taken += 1
        old_pos = pos.copy()
        old_vel = vel.copy()

        # --- Runge-Kutta 4 Integration ---
        k1_v = get_acceleration(pos, vel)
        k1_p = vel

        k2_v = get_acceleration(pos + k1_p * dt * 0.5, vel + k1_v * dt * 0.5)
        k2_p = vel + k1_v * dt * 0.5

        k3_v = get_acceleration(pos + k2_p * dt * 0.5, vel + k2_v * dt * 0.5)
        k3_p = vel + k2_v * dt * 0.5

        k4_v = get_acceleration(pos + k3_p * dt, vel + k3_v * dt)
        k4_p = vel + k3_v * dt

        vel += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        pos += (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

        # Re-normalise to speed of light (keeps the photon null)
        speed = np.sqrt(np.dot(vel, vel))
        if speed > 0:
            vel = (vel / speed) * C

        path[i+1] = pos

        # --- Accretion Disk Collision (equatorial plane y = 0) ---
        if old_pos[1] * pos[1] <= 0.0:
            dy = pos[1] - old_pos[1]
            if dy != 0.0:
                t_frac = -old_pos[1] / dy

                # Exact intersection point via linear interpolation
                hit_x = old_pos[0] + t_frac * (pos[0] - old_pos[0])
                hit_z = old_pos[2] + t_frac * (pos[2] - old_pos[2])

                r_hit_sq = hit_x * hit_x + hit_z * hit_z

                if (DISK_INNER * DISK_INNER) <= r_hit_sq <= (DISK_OUTER * DISK_OUTER):
                    # --- 2. CHANGE: Append to arrays instead of assigning to scalars & breaking ---
                    if hit_count < 4:
                        hit_radii[hit_count] = np.sqrt(r_hit_sq)
                        hit_phis[hit_count] = np.arctan2(hit_z, hit_x)
                        
                        # Interpolate photon velocity at the crossing point
                        h_vel = old_vel + t_frac * (vel - old_vel)
                        h_speed = np.sqrt(h_vel[0]**2 + h_vel[1]**2 + h_vel[2]**2)
                        if h_speed > 0:
                            hit_vels[hit_count, 0] = h_vel[0] / h_speed
                            hit_vels[hit_count, 1] = h_vel[1] / h_speed
                            hit_vels[hit_count, 2] = h_vel[2] / h_speed
                        else:
                            hit_vels[hit_count] = h_vel
                            
                        hit_count += 1
                        # DO NOT break here anymore! The ray passes through the gas disk.

        # --- Termination: captured or escaped ---
        dist_sq = np.dot(pos, pos)

        if dist_sq < (RS * 1.01) ** 2:
            captured = True
            break

        if dist_sq > SIM_BOUNDS ** 2:
            break

    # --- 3. CHANGE: Return arrays and crossing counts ---
    return path[:steps_taken + 1], captured, hit_count, hit_radii, hit_phis, hit_vels