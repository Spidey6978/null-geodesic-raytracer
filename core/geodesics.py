"""
Module: core.geodesics
Calculates the path of a photon through curved spacetime.
Optimized with Numba for C-speed execution.

v5.1: Zero-Allocation Pure Scalar RK4 Solver with Static Returns.
"""
import numpy as np
from numba import njit
from .constants import RS, C, SIM_BOUNDS, DISK_INNER, DISK_OUTER

@njit(nopython=True, cache=True, fastmath=True)
def _get_acceleration_scalar(px, py, pz, vx, vy, vz):
    """
    Calculates Schwarzschild gravitational acceleration using pure scalar math.
    Zero heap allocations.
    """
    r_sq = px*px + py*py + pz*pz
    if r_sq < RS * RS:
        return 0.0, 0.0, 0.0

    # Cross product of position and velocity: pos x vel
    hx = py * vz - pz * vy
    hy = pz * vx - px * vz
    hz = px * vy - py * vx
    h2 = hx*hx + hy*hy + hz*hz

    prefactor = -1.5 * RS * h2 / (r_sq * r_sq * r_sq)
    return prefactor * px, prefactor * py, prefactor * pz


@njit(nopython=True, cache=True, fastmath=True)
def integrate_path(start_pos, start_vel, dt=0.5, max_steps=5000):
    """
    Traces a single photon ray using a zero-allocation, register-optimized
    scalar RK4 integrator through Schwarzschild spacetime.
    Tracks up to 4 separate equatorial disk crossings.
    """
    # Unpack vectors into fast CPU register scalars
    px = start_pos[0]
    py = start_pos[1]
    pz = start_pos[2]

    vx = start_vel[0]
    vy = start_vel[1]
    vz = start_vel[2]

    # Pre-normalize velocity to C using scalar operations
    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C
        vy = (vy / speed) * C
        vz = (vz / speed) * C

    path = np.zeros((max_steps + 1, 3), dtype=np.float64)
    path[0, 0] = px
    path[0, 1] = py
    path[0, 2] = pz

    steps_taken = 0
    captured = False

    # Pre-allocate fixed arrays for crossings
    hit_count = 0
    hit_radii = np.zeros(4, dtype=np.float64)
    hit_phis = np.zeros(4, dtype=np.float64)
    hit_vels = np.zeros((4, 3), dtype=np.float64)

    dt_half = dt * 0.5

    for i in range(max_steps):
        steps_taken += 1
        
        # Track previous coordinates as scalars (replacing .copy() overhead!)
        old_px, old_py, old_pz = px, py, pz
        old_vx, old_vy, old_vz = vx, vy, vz

        # --- RK4 Scalar Integration Step ---
        
        # k1
        k1_vx, k1_vy, k1_vz = _get_acceleration_scalar(px, py, pz, vx, vy, vz)
        k1_px, k1_py, k1_pz = vx, vy, vz

        # k2
        p2_x = px + k1_px * dt_half
        p2_y = py + k1_py * dt_half
        p2_z = pz + k1_pz * dt_half
        v2_x = vx + k1_vx * dt_half
        v2_y = vy + k1_vy * dt_half
        v2_z = vz + k1_vz * dt_half
        k2_vx, k2_vy, k2_vz = _get_acceleration_scalar(p2_x, p2_y, p2_z, v2_x, v2_y, v2_z)
        k2_px, k2_py, k2_pz = v2_x, v2_y, v2_z

        # k3
        p3_x = px + k2_px * dt_half
        p3_y = py + k2_py * dt_half
        p3_z = pz + k2_pz * dt_half
        v3_x = vx + k2_vx * dt_half
        v3_y = vy + k2_vy * dt_half
        v3_z = vz + k2_vz * dt_half
        k3_vx, k3_vy, k3_vz = _get_acceleration_scalar(p3_x, p3_y, p3_z, v3_x, v3_y, v3_z)
        k3_px, k3_py, k3_pz = v3_x, v3_y, v3_z

        # k4
        p4_x = px + k3_px * dt
        p4_y = py + k3_py * dt
        p4_z = pz + k3_pz * dt
        v4_x = vx + k3_vx * dt
        v4_y = vy + k3_vy * dt
        v4_z = vz + k3_vz * dt
        k4_vx, k4_vy, k4_vz = _get_acceleration_scalar(p4_x, p4_y, p4_z, v4_x, v4_y, v4_z)
        k4_px, k4_py, k4_pz = v4_x, v4_y, v4_z

        # Update scalar coordinates
        vx += (dt / 6.0) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        vy += (dt / 6.0) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)
        vz += (dt / 6.0) * (k1_vz + 2*k2_vz + 2*k3_vz + k4_vz)
        
        px += (dt / 6.0) * (k1_px + 2*k2_px + 2*k3_px + k4_px)
        py += (dt / 6.0) * (k1_py + 2*k2_py + 2*k3_py + k4_py)
        pz += (dt / 6.0) * (k1_pz + 2*k2_pz + 2*k3_pz + k4_pz)

        # Normalize velocity components
        v_speed = (vx*vx + vy*vy + vz*vz)**0.5
        if v_speed > 0:
            vx = (vx / v_speed) * C
            vy = (vy / v_speed) * C
            vz = (vz / v_speed) * C

        # Log position to trajectory array
        path[steps_taken, 0] = px
        path[steps_taken, 1] = py
        path[steps_taken, 2] = pz

        # --- Accretion Disk Collision (equatorial plane y = 0) ---
        if old_py * py <= 0.0:
            dy = py - old_py
            if dy != 0.0:
                t_frac = -old_py / dy

                # Exact intersection coordinates via linear interpolation
                hit_x = old_px + t_frac * (px - old_px)
                hit_z = old_pz + t_frac * (pz - old_pz)
                r_hit_sq = hit_x * hit_x + hit_z * hit_z

                # If crossing lands inside disk annulus
                if (DISK_INNER * DISK_INNER) <= r_hit_sq <= (DISK_OUTER * DISK_OUTER):
                    if hit_count < 4:
                        hit_radii[hit_count] = r_hit_sq ** 0.5
                        hit_phis[hit_count] = np.arctan2(hit_z, hit_x)
                        
                        # Interpolate velocity direction vectors
                        h_vx = old_vx + t_frac * (vx - old_vx)
                        h_vy = old_vy + t_frac * (vy - old_vy)
                        h_vz = old_vz + t_frac * (vz - old_vz)
                        h_speed = (h_vx*h_vx + h_vy*h_vy + h_vz*h_vz)**0.5
                        
                        if h_speed > 0:
                            hit_vels[hit_count, 0] = h_vx / h_speed
                            hit_vels[hit_count, 1] = h_vy / h_speed
                            hit_vels[hit_count, 2] = h_vz / h_speed
                        else:
                            hit_vels[hit_count, 0] = h_vx
                            hit_vels[hit_count, 1] = h_vy
                            hit_vels[hit_count, 2] = h_vz
                            
                        hit_count += 1

        # --- Termination ---
        dist_sq = px*px + py*py + pz*pz
        if dist_sq < (RS * 1.01) ** 2:
            captured = True
            break
        if dist_sq > SIM_BOUNDS ** 2:
            break

    # --- PERFORMANCE FIX: Return unsliced static path array + steps_taken scalar ---
    return path, steps_taken, captured, hit_count, hit_radii, hit_phis, hit_vels