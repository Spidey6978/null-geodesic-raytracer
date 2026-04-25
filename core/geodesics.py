"""
Module: core.geodesics
Calculates the path of a photon through curved spacetime.
Optimized with Numba for C-speed execution.

v2: Returns hit_pos and hit_vel at disk intersection for Doppler beaming.
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

    Returns:
        path         : np.ndarray, shape (steps, 3) — the full trajectory
        captured     : bool — True if the photon fell inside the event horizon
        hit_disk     : bool — True if the photon intersected the accretion disk
        hit_radius   : float — cylindrical radius of disk hit (in geometrized units)
        hit_phi      : float — azimuthal angle of disk hit (radians, in XZ plane)
                               0 = +X axis, pi/2 = +Z axis. Used for Doppler beaming.
        hit_vel      : np.ndarray, shape (3,) — photon velocity direction at disk hit.
                               Used to compute the angle between photon and disk gas motion.
    """
    pos = start_pos.astype(np.float64)
    vel = start_vel.astype(np.float64)

    path = np.zeros((max_steps + 1, len(pos)), dtype=np.float64)
    path[0] = pos

    steps_taken = 0
    captured = False

    hit_disk   = False
    hit_radius = 0.0
    hit_phi    = 0.0
    hit_vel    = np.zeros(3, dtype=np.float64)

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
                    hit_disk   = True
                    hit_radius = np.sqrt(r_hit_sq)
                    # atan2(z, x): phi=0 along +X, phi=pi/2 along +Z
                    hit_phi  = np.arctan2(hit_z, hit_x)
                    # Interpolated photon velocity at the crossing point
                    hit_vel  = old_vel + t_frac * (vel - old_vel)
                    speed2   = np.sqrt(np.dot(hit_vel, hit_vel))
                    if speed2 > 0:
                        hit_vel = hit_vel / speed2
                    break

        # --- Termination: captured or escaped ---
        dist_sq = np.dot(pos, pos)

        if dist_sq < (RS * 1.01) ** 2:
            captured = True
            break

        if dist_sq > SIM_BOUNDS ** 2:
            break

    return path[:steps_taken + 1], captured, hit_disk, hit_radius, hit_phi, hit_vel