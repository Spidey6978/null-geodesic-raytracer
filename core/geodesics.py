"""
Module: core.geodesics
Calculates the path of a photon through curved spacetime.
Optimized with Numba for C-speed execution.

v6.8: Analytical Softening & Post-Step Normalization
Resolves the RK4 Mid-Step discontinuity and prevents the Stiff ODE 
polar explosion using mathematical softening.
"""
import numpy as np
from numba import njit
from .constants import RS, C, SIM_BOUNDS, DISK_INNER, DISK_OUTER, MASS, SPIN, R_OUTER_HORIZON

# ── 1. Coordinate Translators ─────────────────────────────────────────────────

@njit(nopython=True, cache=True)
def _cartesian_to_bl(x, y, z, vx, vy, vz, a):
    bl_x, bl_y, bl_z = x, z, y
    bl_vx, bl_vy, bl_vz = vx, vz, vy

    R_sq = bl_x*bl_x + bl_y*bl_y + bl_z*bl_z
    r_sq = 0.5 * ((R_sq - a*a) + np.sqrt((R_sq - a*a)**2 + 4.0 * a*a * bl_z*bl_z))
    r = np.sqrt(r_sq)

    theta = np.arccos(bl_z / r) if r > 1e-12 else np.pi / 2.0
    if theta < 1e-9: theta = 1e-9
    if theta > np.pi - 1e-9: theta = np.pi - 1e-9

    phi = np.arctan2(bl_y, bl_x)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    sqrt_ra = np.sqrt(r*r + a*a)
    
    J = np.zeros((3, 3), dtype=np.float64)
    J[0, 0] = (r / sqrt_ra) * sin_t * cos_p
    J[0, 1] = sqrt_ra * cos_t * cos_p
    J[0, 2] = -sqrt_ra * sin_t * sin_p
    
    J[1, 0] = (r / sqrt_ra) * sin_t * sin_p
    J[1, 1] = sqrt_ra * cos_t * sin_p
    J[1, 2] = sqrt_ra * sin_t * cos_p
    
    J[2, 0] = cos_t
    J[2, 1] = -r * sin_t
    J[2, 2] = 0.0
    
    v_vec = np.array([bl_vx, bl_vy, bl_vz], dtype=np.float64)
    bl_dots = np.linalg.solve(J, v_vec)
    
    return r, theta, phi, bl_dots[0], bl_dots[1], bl_dots[2]

@njit(nopython=True, cache=True)
def _bl_to_cartesian_pos(r, theta, phi, a):
    sqrt_ra = np.sqrt(r*r + a*a)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    
    bl_x = sqrt_ra * sin_t * cos_p
    bl_y = sqrt_ra * sin_t * sin_p
    bl_z = r * cos_t
    
    return bl_x, bl_z, bl_y

@njit(nopython=True, cache=True)
def _bl_to_cartesian_vel(r, theta, phi, dr, dtheta, dphi, a):
    sqrt_ra = np.sqrt(r*r + a*a)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    
    dx_dr = (r / sqrt_ra) * sin_t * cos_p
    dx_dt = sqrt_ra * cos_t * cos_p
    dx_dp = -sqrt_ra * sin_t * sin_p
    
    dy_dr = (r / sqrt_ra) * sin_t * sin_p
    dy_dt = sqrt_ra * cos_t * sin_p
    dy_dp = sqrt_ra * sin_t * cos_p
    
    dz_dr = cos_t
    dz_dt = -r * sin_t
    dz_dp = 0.0
    
    bl_vx = dx_dr * dr + dx_dt * dtheta + dx_dp * dphi
    bl_vy = dy_dr * dr + dy_dt * dtheta + dy_dp * dphi
    bl_vz = dz_dr * dr + dz_dt * dtheta + dz_dp * dphi
    
    return bl_vx, bl_vz, bl_vy

# ── 2. Kerr Constants & Hamiltonian ───────────────────────────────────────────

@njit(nopython=True, cache=True)
def _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, a, M):
    Sigma = r*r + a*a * np.cos(theta)**2
    Delta = r*r - 2.0*M*r + a*a
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    
    g_tt = -(1.0 - 2.0*M*r / Sigma)
    g_tphi = -2.0*M*a*r * sin_t**2 / Sigma
    g_phiphi = (r*r + a*a + 2.0*M*a*a*r * sin_t**2 / Sigma) * sin_t**2
    g_rr = Sigma / Delta
    g_thetatheta = Sigma
    
    A = g_tt
    B = 2.0 * g_tphi * dot_phi
    C = g_phiphi * dot_phi**2 + g_rr * dot_r**2 + g_thetatheta * dot_theta**2
    
    discriminant = B*B - 4.0*A*C
    if discriminant < 0.0: discriminant = 0.0
        
    t_root1 = (-B + np.sqrt(discriminant)) / (2.0 * A)
    t_root2 = (-B - np.sqrt(discriminant)) / (2.0 * A)
    dot_t = max(t_root1, t_root2)
    
    p_t = g_tt * dot_t + g_tphi * dot_phi
    p_phi = g_tphi * dot_t + g_phiphi * dot_phi
    p_r = g_rr * dot_r
    p_theta = g_thetatheta * dot_theta
    
    E = -p_t
    L = p_phi
    
    sin_t_safe = max(abs(sin_t), 1e-3)
    Q = p_theta**2 + cos_t**2 * ( (L**2 / sin_t_safe**2) - a**2 * E**2 )
    
    return 1.0, L / E, Q / (E**2), p_r / E, p_theta / E

@njit(nopython=True, cache=True)
def _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, a, M):
    # No coordinate bounces inside the derivatives! RK4 remains perfectly smooth.
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    Sigma = r*r + a*a * cos_t*cos_t
    Delta = r*r - 2.0*M*r + a*a
    
    # Secure Delta to prevent Mid-Step zero-division near the horizon
    inv_Sigma = 1.0 / Sigma
    Delta_safe = Delta

    if Delta_safe > 0.0:
        Delta_safe = max(Delta_safe, 1e-6)
    else:
        Delta_safe = min(Delta_safe, -1e-6)

    inv_Delta = 1.0 / Delta_safe    
    # ── ANALYTICAL SOFTENING (The Quasar Killer) ──
    # By adding 1e-7 to the sin powers, we prevent the infinite singularity at the poles
    # while maintaining a perfectly smooth, differentiable curve for the RK4 integrator.
    # No min/max caps required!
    sin2 = sin_t * sin_t
    inv_sin2 = 1.0 / (sin2 + 1e-7)
    soft_sin4 = sin2 * sin2 + 1e-7
    
    P = E * (r*r + a*a) - a * L
    
    dr_dlam  = Delta * inv_Sigma * pr
    dth_dlam = ptheta * inv_Sigma
    dph_dlam = inv_Sigma * ( (L * inv_sin2) - a*E + (a*P * inv_Delta) )
    dt_dlam  = inv_Sigma * ( -a * (a*E * sin2 - L) + ((r*r + a*a)*P * inv_Delta) )
    
    K = Q + (a*E - L)**2
    
    # PURE GRAVITY: The true Kerr gravitational pull
    dpr_dlam = inv_Sigma * ( ((2.0*r*E*P - (r - M)*K) * inv_Delta) - 2.0*(r - M)*pr*pr )
    
    # SMOOTH CENTRIFUGAL BARRIER: Perfectly balanced repulsion at the poles
    dptheta_dlam = (cos_t * sin_t * inv_Sigma) * ( (L*L / soft_sin4) - a*a * E*E )
    
    return dr_dlam, dth_dlam, dph_dlam, dpr_dlam, dptheta_dlam, dt_dlam

# ── 3. The Runge-Kutta 4 Integrators ──────────────────────────────────────────

@njit(nopython=True, cache=True)
def integrate_path(start_pos, start_vel, dt=0.5, max_steps=5000):
    px, py, pz = start_pos[0], start_pos[1], start_pos[2]
    vx, vy, vz = start_vel[0], start_vel[1], start_vel[2]

    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C
        vy = (vy / speed) * C
        vz = (vz / speed) * C

    r, theta, phi, dot_r, dot_theta, dot_phi = _cartesian_to_bl(px, py, pz, vx, vy, vz, SPIN)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, SPIN, MASS)
    t = 0.0

    path = np.zeros((max_steps + 1, 3), dtype=np.float64)
    path[0, 0] = px; path[0, 1] = py; path[0, 2] = pz

    steps_taken = 0
    captured = False
    termination_reason = 0
    hit_count = 0
    hit_radii = np.zeros(4, dtype=np.float64)
    hit_phis = np.zeros(4, dtype=np.float64)
    hit_vels = np.zeros((4, 3), dtype=np.float64)

    dt_local = dt
    if r < 5.0:
        dt_local = dt * 0.5
    if r < 3.0:
        dt_local = dt * 0.25
    if r < 2.0:
        dt_local = dt * 0.1
        
    dt_half = dt_local * 0.5
    pi_2 = np.pi / 2.0
    
    capture_radius = R_OUTER_HORIZON + 0.05

    for i in range(max_steps):
        steps_taken += 1
        
        old_r, old_theta, old_phi = r, theta, phi
        old_pr, old_ptheta = pr, ptheta

        dr1, dth1, dph1, dpr1, dpth1, dt1 = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, SPIN, MASS)
        
        r2 = r + dr1*dt_half; th2 = theta + dth1*dt_half; ph2 = phi + dph1*dt_half
        pr2 = pr + dpr1*dt_half; pth2 = ptheta + dpth1*dt_half
        dr2, dth2, dph2, dpr2, dpth2, dt2 = _kerr_derivatives(r2, th2, ph2, pr2, pth2, E, L, Q, SPIN, MASS)
        
        r3 = r + dr2*dt_half; th3 = theta + dth2*dt_half; ph3 = phi + dph2*dt_half
        pr3 = pr + dpr2*dt_half; pth3 = ptheta + dpth2*dt_half
        dr3, dth3, dph3, dpr3, dpth3, dt3 = _kerr_derivatives(r3, th3, ph3, pr3, pth3, E, L, Q, SPIN, MASS)
        
        r4 = r + dr3*dt_local; th4 = theta + dth3*dt_local; ph4 = phi + dph3*dt_local
        pr4 = pr + dpr3*dt_local; pth4 = ptheta + dpth3*dt_local
        dr4, dth4, dph4, dpr4, dpth4, dt4 = _kerr_derivatives(r4, th4, ph4, pr4, pth4, E, L, Q, SPIN, MASS)

        r      += (dt_local / 6.0) * (dr1 + 2*dr2 + 2*dr3 + dr4)
        theta  += (dt_local / 6.0) * (dth1 + 2*dth2 + 2*dth3 + dth4)
        phi    += (dt_local / 6.0) * (dph1 + 2*dph2 + 2*dph3 + dph4)
        pr     += (dt_local / 6.0) * (dpr1 + 2*dpr2 + 2*dpr3 + dpr4)
        ptheta += (dt_local / 6.0) * (dpth1 + 2*dpth2 + 2*dpth3 + dpth4)
        t      += (dt_local / 6.0) * (dt1 + 2*dt2 + 2*dt3 + dt4)

        if (
            not np.isfinite(r)
            or not np.isfinite(theta)
            or not np.isfinite(phi)
            or not np.isfinite(pr)
            or not np.isfinite(ptheta)
            ):
            captured = True
            termination_reason = 2
            break
        
        # ── POST-STEP NORMALIZATION (The Ghost Killer) ──
        # Normalizing coordinates outside the derivative function ensures RK4 
        # stays continuous, while preparing correct values for disk collision.
        while theta < 0.0 or theta > np.pi:
            if theta < 0.0:
                theta = -theta
                ptheta = -ptheta
                phi += np.pi
            elif theta > np.pi:
                theta = 2.0 * np.pi - theta
                ptheta = -ptheta
                phi += np.pi

        px, py, pz = _bl_to_cartesian_pos(r, theta, phi, SPIN)
        path[steps_taken, 0] = px
        path[steps_taken, 1] = py
        path[steps_taken, 2] = pz

        if (old_theta - pi_2) * (theta - pi_2) <= 0.0:
            d_theta = theta - old_theta
            if d_theta != 0.0:
                t_frac = (pi_2 - old_theta) / d_theta
                hit_r = old_r + t_frac * (r - old_r)
                
                if (DISK_INNER) <= hit_r <= (DISK_OUTER):
                    if hit_count < 4:
                        hit_phi = old_phi + t_frac * (phi - old_phi)
                        hit_radii[hit_count] = hit_r
                        hit_phis[hit_count]  = hit_phi
                        
                        hit_pr = old_pr + t_frac * (pr - old_pr)
                        hit_ptheta = old_ptheta + t_frac * (ptheta - old_ptheta)
                        
                        hdr, hdth, hdph, _, _, _ = _kerr_derivatives(hit_r, pi_2, hit_phi, hit_pr, hit_ptheta, E, L, Q, SPIN, MASS)
                        h_vx, h_vy, h_vz = _bl_to_cartesian_vel(hit_r, pi_2, hit_phi, hdr, hdth, hdph, SPIN)
                        
                        h_spd = (h_vx*h_vx + h_vy*h_vy + h_vz*h_vz)**0.5
                        if h_spd > 0:
                            hit_vels[hit_count, 0] = h_vx / h_spd
                            hit_vels[hit_count, 1] = h_vy / h_spd
                            hit_vels[hit_count, 2] = h_vz / h_spd
                        else:
                            hit_vels[hit_count, 0] = h_vx
                            hit_vels[hit_count, 1] = h_vy
                            hit_vels[hit_count, 2] = h_vz
                            
                        hit_count += 1

        if r < capture_radius:
            captured = True
            termination_reason = 1
            break
        if not (r <= SIM_BOUNDS):
            termination_reason = 3 
            break

    # If we exited the loop without hitting any break condition above,
    # it means the integrator ran to completion (natural termination).
    if termination_reason == 0:
        termination_reason = 4

    return path, steps_taken, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason

@njit(nopython=True, cache=True)
def integrate_path_lean(start_pos, start_vel, dt=0.5, max_steps=2000):
    px, py, pz = start_pos[0], start_pos[1], start_pos[2]
    vx, vy, vz = start_vel[0], start_vel[1], start_vel[2]

    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C; vy = (vy / speed) * C; vz = (vz / speed) * C

    r, theta, phi, dot_r, dot_theta, dot_phi = _cartesian_to_bl(px, py, pz, vx, vy, vz, SPIN)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, SPIN, MASS)
    
    captured = False
    hit_count = 0
    termination_reason = 0
    hit_radii = np.zeros(4, dtype=np.float64)
    hit_phis = np.zeros(4, dtype=np.float64)
    hit_vels = np.zeros((4, 3), dtype=np.float64)

    dt_local = dt
    if r < 5.0:
        dt_local *= 0.5

    if r < 3.0:
        dt_local *= 0.25

    if r < 2.0:
        dt_local *= 0.1
        
    dt_half = dt_local * 0.5
    pi_2 = np.pi / 2.0
    capture_radius = R_OUTER_HORIZON + 0.05

    for i in range(max_steps):
        old_r, old_theta, old_phi = r, theta, phi
        old_pr, old_ptheta = pr, ptheta

        dr1, dth1, dph1, dpr1, dpth1, dt1 = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, SPIN, MASS)
        
        r2 = r + dr1*dt_half; th2 = theta + dth1*dt_half; ph2 = phi + dph1*dt_half
        pr2 = pr + dpr1*dt_half; pth2 = ptheta + dpth1*dt_half
        dr2, dth2, dph2, dpr2, dpth2, dt2 = _kerr_derivatives(r2, th2, ph2, pr2, pth2, E, L, Q, SPIN, MASS)
        
        r3 = r + dr2*dt_half; th3 = theta + dth2*dt_half; ph3 = phi + dph2*dt_half
        pr3 = pr + dpr2*dt_half; pth3 = ptheta + dpth2*dt_half
        dr3, dth3, dph3, dpr3, dpth3, dt3 = _kerr_derivatives(r3, th3, ph3, pr3, pth3, E, L, Q, SPIN, MASS)
        
        r4 = r + dr3*dt_local; th4 = theta + dth3*dt_local; ph4 = phi + dph3*dt_local
        pr4 = pr + dpr3*dt_local; pth4 = ptheta + dpth3*dt_local
        dr4, dth4, dph4, dpr4, dpth4, dt4 = _kerr_derivatives(r4, th4, ph4, pr4, pth4, E, L, Q, SPIN, MASS)

        r      += (dt_local / 6.0) * (dr1 + 2*dr2 + 2*dr3 + dr4)
        theta  += (dt_local / 6.0) * (dth1 + 2*dth2 + 2*dth3 + dth4)
        phi    += (dt_local / 6.0) * (dph1 + 2*dph2 + 2*dph3 + dph4)
        pr     += (dt_local / 6.0) * (dpr1 + 2*dpr2 + 2*dpr3 + dpr4)
        ptheta += (dt_local / 6.0) * (dpth1 + 2*dpth2 + 2*dpth3 + dpth4)
        
        if (
            not np.isfinite(r)
            or not np.isfinite(theta)
            or not np.isfinite(phi)
            or not np.isfinite(pr)
            or not np.isfinite(ptheta)
            ):
            captured = True
            termination_reason = 2
            break

        while theta < 0.0 or theta > np.pi:
            if theta < 0.0:
                theta = -theta
                ptheta = -ptheta
                phi += np.pi
            elif theta > np.pi:
                theta = 2.0 * np.pi - theta
                ptheta = -ptheta
                phi += np.pi

        if (old_theta - pi_2) * (theta - pi_2) <= 0.0:
            d_theta = theta - old_theta
            if d_theta != 0.0:
                t_frac = (pi_2 - old_theta) / d_theta
                hit_r = old_r + t_frac * (r - old_r)
                
                if (DISK_INNER) <= hit_r <= (DISK_OUTER):
                    if hit_count < 4:
                        hit_phi = old_phi + t_frac * (phi - old_phi)
                        hit_radii[hit_count] = hit_r
                        hit_phis[hit_count]  = hit_phi
                        
                        hit_pr = old_pr + t_frac * (pr - old_pr)
                        hit_ptheta = old_ptheta + t_frac * (ptheta - old_ptheta)
                        
                        hdr, hdth, hdph, _, _, _ = _kerr_derivatives(hit_r, pi_2, hit_phi, hit_pr, hit_ptheta, E, L, Q, SPIN, MASS)
                        h_vx, h_vy, h_vz = _bl_to_cartesian_vel(hit_r, pi_2, hit_phi, hdr, hdth, hdph, SPIN)
                        
                        h_spd = (h_vx*h_vx + h_vy*h_vy + h_vz*h_vz)**0.5
                        if h_spd > 0:
                            hit_vels[hit_count, 0] = h_vx / h_spd
                            hit_vels[hit_count, 1] = h_vy / h_spd
                            hit_vels[hit_count, 2] = h_vz / h_spd
                        else:
                            hit_vels[hit_count, 0] = h_vx
                            hit_vels[hit_count, 1] = h_vy
                            hit_vels[hit_count, 2] = h_vz
                            
                        hit_count += 1

        if r < capture_radius:
            captured = True
            termination_reason = 1
            break
        if not (r <= SIM_BOUNDS):
            termination_reason = 3 
            break

    # If we exited the loop without setting a reason, mark natural termination
    if termination_reason == 0:
        termination_reason = 4

    fdr, fdth, fdph, _, _, _ = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, SPIN, MASS)
    f_vx, f_vy, f_vz = _bl_to_cartesian_vel(r, theta, phi, fdr, fdth, fdph, SPIN)
    
    final_dir = np.empty(3, dtype=np.float64)
    final_dir[0] = f_vx; final_dir[1] = f_vy; final_dir[2] = f_vz
    
    return final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason