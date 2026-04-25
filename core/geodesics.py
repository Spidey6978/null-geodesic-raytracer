"""
Module: core.geodesics
Calculates the path of a photon through curved spacetime.
Optimized with Numba for C-speed execution.
"""
import numpy as np
from numba import njit
from .constants import RS, C, SIM_BOUNDS, DISK_INNER, DISK_OUTER

@njit(nopython=True, cache=True)
def get_acceleration(pos, vel):
    """Calculates the relativistic acceleration for a photon."""
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
    Traces a single ray using RK4 integration.
    """
    pos = start_pos.astype(np.float64)
    vel = start_vel.astype(np.float64)
    
    path = np.zeros((max_steps + 1, len(pos)), dtype=np.float64)
    path[0] = pos
    
    steps_taken = 0
    captured = False
    
    # --- HIT TRACKING ---
    hit_disk = False
    hit_radius = 0.0
    hit_pos = np.zeros(3, dtype=np.float64)
    hit_vel = np.zeros(3, dtype=np.float64)
    
    for i in range(max_steps):
        steps_taken += 1
        old_pos = pos.copy()
        
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

        speed = np.sqrt(np.dot(vel, vel))
        if speed > 0:
            vel = (vel / speed) * C
        
        path[i+1] = pos
        
        # --- Accretion Disk Collision Logic ---
        if old_pos[1] * pos[1] <= 0.0:
            dy = pos[1] - old_pos[1]
            if dy != 0.0:
                t = -old_pos[1] / dy
                hit_x = old_pos[0] + t * (pos[0] - old_pos[0])
                hit_z = old_pos[2] + t * (pos[2] - old_pos[2])
                r_hit_sq = hit_x*hit_x + hit_z*hit_z
                
                if (DISK_INNER*DISK_INNER) <= r_hit_sq <= (DISK_OUTER*DISK_OUTER):
                    hit_disk = True
                    hit_radius = np.sqrt(r_hit_sq)
                    hit_pos[0] = hit_x
                    hit_pos[1] = 0.0
                    hit_pos[2] = hit_z
                    hit_vel[0] = vel[0]
                    hit_vel[1] = vel[1]
                    hit_vel[2] = vel[2]
                    break 
        
        # --- Termination Conditions ---
        dist_sq = np.dot(pos, pos)
        if dist_sq < (RS * 1.01)**2: 
            captured = True
            break
        if dist_sq > SIM_BOUNDS**2:
            break
            
    # We now return 6 specific variables for advanced physics shaders
    return path[:steps_taken+1], captured, hit_disk, hit_radius, hit_pos, hit_vel