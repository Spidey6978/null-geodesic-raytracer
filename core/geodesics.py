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
from core.indices import *

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
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    Sigma = r*r + a*a * cos_t*cos_t
    Delta = r*r - 2.0*M*r + a*a

    inv_Sigma = 1.0 / max(Sigma, 1e-10)

    # Horizon-aware softening — floor scales with M² not an absolute
    Delta_floor = 1e-4 * M * M
    if Delta > 0.0:
        Delta_safe = max(Delta, Delta_floor)
    else:
        Delta_safe = min(Delta, -Delta_floor)
    inv_Delta = 1.0 / Delta_safe

    # Polar softening — unchanged
    sin2     = sin_t * sin_t
    inv_sin2 = 1.0 / (sin2 + 1e-7)
    soft_sin4 = sin2 * sin2 + 1e-7

    P = E * (r*r + a*a) - a * L

    # Cap P/Delta and K/Delta ratios to prevent near-horizon explosion
    _cap = 1e4 / max(M, 1e-10)

    P_over_D = P * inv_Delta
    if   P_over_D >  _cap: P_over_D =  _cap
    elif P_over_D < -_cap: P_over_D = -_cap

    K = Q + (a*E - L)**2
    K_over_D = K * inv_Delta
    if   K_over_D >  _cap: K_over_D =  _cap
    elif K_over_D < -_cap: K_over_D = -_cap

    dr_dlam      = Delta    * inv_Sigma * pr
    dth_dlam     = ptheta   * inv_Sigma
    dph_dlam     = inv_Sigma * (L * inv_sin2  - a*E + a*P_over_D)
    dt_dlam      = inv_Sigma * (-a*(a*E*sin2 - L) + (r*r + a*a)*P_over_D)
    dpr_dlam     = inv_Sigma * (2.0*r*E*P_over_D - (r-M)*K_over_D
                                - 2.0*(r-M)*pr*pr)
    dptheta_dlam = (cos_t * sin_t * inv_Sigma) * (L*L/soft_sin4 - a*a*E*E)

    return dr_dlam, dth_dlam, dph_dlam, dpr_dlam, dptheta_dlam, dt_dlam

@njit(nopython=True, cache=True)
def integrate_path(start_pos, start_vel, dt, max_steps,
                   mass, spin, r_outer_horizon,
                   disk_inner, disk_outer, sim_bounds):
    px, py, pz = start_pos[0], start_pos[1], start_pos[2]
    vx, vy, vz = start_vel[0], start_vel[1], start_vel[2]

    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C
        vy = (vy / speed) * C
        vz = (vz / speed) * C

    r, theta, phi, dot_r, dot_theta, dot_phi = _cartesian_to_bl(
    px, py, pz, vx, vy, vz, spin
)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, spin, mass)
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

    
    pi_2 = np.pi / 2.0
    
    # Capture radius: 5% outside the event horizon.
    # Boyer-Lindquist coordinates become numerically degenerate as Delta->0
    # at the horizon. Since no photon at r < r_horizon can physically escape,
    # stopping integration here hides no observable physics — only the
    # unobservable final infall. The buffer prevents inv_Delta from spiking
    # into the capped softening regime, keeping dpr/dphi derivatives clean.
    # KS coordinates would eliminate the need for this buffer entirely.
    capture_radius = r_outer_horizon * 1.05

    if (
        not np.isfinite(r)
        or not np.isfinite(theta)
        or not np.isfinite(phi)
        or not np.isfinite(E)
        or not np.isfinite(L)
        or not np.isfinite(Q)
        or not np.isfinite(pr)
        or not np.isfinite(ptheta)
    ):
        captured = True
        termination_reason = 2
    elif r <= capture_radius:
        captured = True
        termination_reason = 1
    elif r >= sim_bounds:
        termination_reason = 3

    if termination_reason != 0:
        final_dir = np.empty(3, dtype=np.float64)
        final_dir[0] = 0.0
        final_dir[1] = 0.0
        final_dir[2] = 0.0
        return final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason

    for i in range(max_steps):
        # ── PRE-STEP CAPTURE CHECK ─────────────────────────────────────────
        # Avoids evaluating derivatives at r < capture_radius where
        # Delta is negative and inv_Delta would be large even with softening.
        if r < capture_radius:
            captured = True
            termination_reason = 1
            break
        steps_taken += 1
        
        # ── Smooth adaptive dt ────────────────────────────────────────────────────
        # Three independent danger penalties compound multiplicatively:
        #   r_factor   → reduces dt near the event horizon (Delta → 0)
        #   theta_factor → reduces dt near the spin axis (sin²θ → 0)
        #   pole_factor  → reduces dt when the path approaches the polar axis.
        # Combined floor of 0.01 prevents dt from going to zero.

        r_norm    = (r - r_outer_horizon) / (r_outer_horizon * 4.0)
        r_factor  = r_norm if r_norm < 1.0 else 1.0
        r_factor  = r_factor if r_factor > 0.0 else 0.0

        sin2      = np.sin(theta) ** 2
        gap       = theta if theta < pi_2 else np.pi - theta
        theta_factor = sin2 / (sin2 + 0.05)
        pole_factor = gap / (gap + 0.05)

        far_factor = 1.0 + 3.0 * max(r / (sim_bounds * 0.1) - 1.0, 0.0)
        far_factor = far_factor if far_factor < 4.0 else 4.0

        dphi_scale = abs(L) / (sin2 + 1e-6)
        polar_cap = 0.15 / dphi_scale if dphi_scale > 0.15 else 1.0
        dt_local = dt * min(max(r_factor * theta_factor, 1e-4), polar_cap) * far_factor
        dt_half   = dt_local * 0.5
        
        old_r, old_theta, old_phi = r, theta, phi
        old_pr, old_ptheta = pr, ptheta

        dr1, dth1, dph1, dpr1, dpth1, dt1 = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, spin, mass)
        
        r2 = r + dr1*dt_half; th2 = theta + dth1*dt_half; ph2 = phi + dph1*dt_half
        pr2 = pr + dpr1*dt_half; pth2 = ptheta + dpth1*dt_half
        dr2, dth2, dph2, dpr2, dpth2, dt2 = _kerr_derivatives(r2, th2, ph2, pr2, pth2, E, L, Q, spin, mass)
        
        r3 = r + dr2*dt_half; th3 = theta + dth2*dt_half; ph3 = phi + dph2*dt_half
        pr3 = pr + dpr2*dt_half; pth3 = ptheta + dpth2*dt_half
        dr3, dth3, dph3, dpr3, dpth3, dt3 = _kerr_derivatives(r3, th3, ph3, pr3, pth3, E, L, Q, spin, mass)
        
        r4 = r + dr3*dt_local; th4 = theta + dth3*dt_local; ph4 = phi + dph3*dt_local
        pr4 = pr + dpr3*dt_local; pth4 = ptheta + dpth3*dt_local
        dr4, dth4, dph4, dpr4, dpth4, dt4 = _kerr_derivatives(r4, th4, ph4, pr4, pth4, E, L, Q, spin, mass)

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

        px, py, pz = _bl_to_cartesian_pos(r, theta, phi, spin)
        path[steps_taken, 0] = px
        path[steps_taken, 1] = py
        path[steps_taken, 2] = pz

        if hit_count < 4:
            if (old_theta - pi_2) * (theta - pi_2) <= 0.0:
                d_theta = theta - old_theta
                if d_theta != 0.0:
                    t_frac = (pi_2 - old_theta) / d_theta
                    hit_r = old_r + t_frac * (r - old_r)
                    
                    old_gap = old_theta if old_theta < pi_2 else np.pi - old_theta
                    if min(old_gap, gap) > 1e-3 and (disk_inner) <= hit_r <= (disk_outer):
                        hit_phi = old_phi + t_frac * (phi - old_phi)
                        hit_radii[hit_count] = hit_r
                        hit_phis[hit_count]  = hit_phi
                        
                        hit_pr = old_pr + t_frac * (pr - old_pr)
                        hit_ptheta = old_ptheta + t_frac * (ptheta - old_ptheta)
                        
                        hdr, hdth, hdph, _, _, _ = _kerr_derivatives(hit_r, pi_2, hit_phi, hit_pr, hit_ptheta, E, L, Q, spin, mass)
                        h_vx, h_vy, h_vz = _bl_to_cartesian_vel(hit_r, pi_2, hit_phi, hdr, hdth, hdph, spin)
                        
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
        if not (r <= sim_bounds):
            termination_reason = 3 
            break
        
    # If we exited the loop without hitting any break condition above,
    # it means the integrator ran to completion (natural termination).
    if termination_reason == 0:
        termination_reason = 4
    if termination_reason == 4:
       if r<5.0*r_outer_horizon:
          termination_reason = 5

    return path, steps_taken, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason

@njit(nopython=True, cache=True, fastmath=True)
def integrate_path_lean(start_pos, start_vel, dt, max_steps,
                        mass, spin, r_outer_horizon,
                        disk_inner, disk_outer, sim_bounds):
    px, py, pz = start_pos[0], start_pos[1], start_pos[2]
    vx, vy, vz = start_vel[0], start_vel[1], start_vel[2]

    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C; vy = (vy / speed) * C; vz = (vz / speed) * C

    r, theta, phi, dot_r, dot_theta, dot_phi = _cartesian_to_bl(px, py, pz, vx, vy, vz, spin)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, spin, mass)

    captured = False
    hit_count = 0
    termination_reason = 0
    initial_state_code = 0
    hit_radii = np.zeros(4, dtype=np.float64)
    hit_phis = np.zeros(4, dtype=np.float64)
    hit_vels = np.zeros((4, 3), dtype=np.float64)

    pi_2 = np.pi / 2.0
    capture_radius = r_outer_horizon * 1.05

    if (
        not np.isfinite(r)
        or not np.isfinite(theta)
        or not np.isfinite(phi)
        or not np.isfinite(E)
        or not np.isfinite(L)
        or not np.isfinite(Q)
        or not np.isfinite(pr)
        or not np.isfinite(ptheta)
    ):
        captured = True
        termination_reason = 2
        initial_state_code = 1
    elif r <= capture_radius:
        captured = True
        termination_reason = 1
        initial_state_code = 2
    elif r >= sim_bounds:
        termination_reason = 3
        initial_state_code = 3

    if termination_reason != 0:
        final_dir = np.empty(3, dtype=np.float64)
        final_dir[0] = 0.0
        final_dir[1] = 0.0
        final_dir[2] = 0.0
        return final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason, initial_state_code

    for i in range(max_steps):
        # ── PRE-STEP CAPTURE CHECK ─────────────────────────────────────────
        # Avoids evaluating derivatives at r < capture_radius where
        # Delta is negative and inv_Delta would be large even with softening.
        if r < capture_radius:
            captured = True
            termination_reason = 1
            break
                
        r_norm    = (r - r_outer_horizon) / (r_outer_horizon * 4.0)
        r_factor  = r_norm if r_norm < 1.0 else 1.0
        r_factor  = r_factor if r_factor > 0.0 else 0.0

        sin2      = np.sin(theta) ** 2
        gap       = theta if theta < pi_2 else np.pi - theta
        theta_factor = sin2 / (sin2 + 0.05)
        pole_factor = gap / (gap + 0.05)

        far_factor = 1.0 + 3.0 * max(r / (sim_bounds * 0.1) - 1.0, 0.0)
        far_factor = far_factor if far_factor < 4.0 else 4.0

        dphi_scale = abs(L) / (sin2 + 1e-6)
        polar_cap = 0.15 / dphi_scale if dphi_scale > 0.15 else 1.0
        dt_local = dt * min(max(r_factor * theta_factor, 1e-4), polar_cap) * far_factor
        dt_half   = dt_local * 0.5
        
        old_r, old_theta, old_phi = r, theta, phi
        old_pr, old_ptheta = pr, ptheta

        dr1, dth1, dph1, dpr1, dpth1, dt1 = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, spin, mass)
        
        r2 = r + dr1*dt_half; th2 = theta + dth1*dt_half; ph2 = phi + dph1*dt_half
        pr2 = pr + dpr1*dt_half; pth2 = ptheta + dpth1*dt_half
        dr2, dth2, dph2, dpr2, dpth2, dt2 = _kerr_derivatives(r2, th2, ph2, pr2, pth2, E, L, Q, spin, mass)
        
        r3 = r + dr2*dt_half; th3 = theta + dth2*dt_half; ph3 = phi + dph2*dt_half
        pr3 = pr + dpr2*dt_half; pth3 = ptheta + dpth2*dt_half
        dr3, dth3, dph3, dpr3, dpth3, dt3 = _kerr_derivatives(r3, th3, ph3, pr3, pth3, E, L, Q, spin, mass)
        
        r4 = r + dr3*dt_local; th4 = theta + dth3*dt_local; ph4 = phi + dph3*dt_local
        pr4 = pr + dpr3*dt_local; pth4 = ptheta + dpth3*dt_local
        dr4, dth4, dph4, dpr4, dpth4, dt4 = _kerr_derivatives(r4, th4, ph4, pr4, pth4, E, L, Q, spin, mass)

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

        if hit_count < 4:
            if (old_theta - pi_2) * (theta - pi_2) <= 0.0:
                d_theta = theta - old_theta
                if d_theta != 0.0:
                    t_frac = (pi_2 - old_theta) / d_theta
                    hit_r = old_r + t_frac * (r - old_r)
                    
                    old_gap = old_theta if old_theta < pi_2 else np.pi - old_theta
                    if min(old_gap, gap) > 1e-3 and (disk_inner) <= hit_r <= (disk_outer):
                        hit_phi = old_phi + t_frac * (phi - old_phi)
                        hit_radii[hit_count] = hit_r
                        hit_phis[hit_count]  = hit_phi
                        
                        hit_pr = old_pr + t_frac * (pr - old_pr)
                        hit_ptheta = old_ptheta + t_frac * (ptheta - old_ptheta)
                        
                        hdr, hdth, hdph, _, _, _ = _kerr_derivatives(hit_r, pi_2, hit_phi, hit_pr, hit_ptheta, E, L, Q, spin, mass)
                        h_vx, h_vy, h_vz = _bl_to_cartesian_vel(hit_r, pi_2, hit_phi, hdr, hdth, hdph, spin)
                        
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

        if not (r <= sim_bounds):
            termination_reason = 3 
            break

    # If we exited the loop without setting a reason, mark natural termination
    if termination_reason == 0:
        termination_reason = 4
        
    if termination_reason == 4:
       if r<5.0*r_outer_horizon:
          termination_reason = 5 

    final_dir = np.empty(3, dtype=np.float64)
    if captured or termination_reason == 5:
        final_dir[0] = 0.0
        final_dir[1] = 0.0
        final_dir[2] = 0.0
    else:
        fdr, fdth, fdph, _, _, _ = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, spin, mass)
        f_vx, f_vy, f_vz = _bl_to_cartesian_vel(r, theta, phi, fdr, fdth, fdph, spin)
        final_dir[0] = f_vx
        final_dir[1] = f_vy
        final_dir[2] = f_vz

    if initial_state_code == 0:
        if captured or termination_reason == 2:
            initial_state_code = 4
        elif termination_reason == 3:
            initial_state_code = 5
        else:
            initial_state_code = 6
    
    return final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason, initial_state_code

@njit(nopython=True, cache=True)
def integrate_path_doctor(start_pos, start_vel, dt, max_steps, mass, a, r_outer_horizon, disk_inner, disk_outer, sim_bounds):
    """
    Diagnostic rendering engine. Skips path arrays to preserve RAM.
    Returns a pre-allocated doctor metric array packed with physics metadata.
    """
    px, py, pz = start_pos[0], start_pos[1], start_pos[2]
    vx, vy, vz = start_vel[0], start_vel[1], start_vel[2]

    speed = (vx*vx + vy*vy + vz*vz)**0.5
    if speed > 0:
        vx = (vx / speed) * C; vy = (vy / speed) * C; vz = (vz / speed) * C

    r, theta, phi, dot_r, dot_theta, dot_phi = _cartesian_to_bl(px, py, pz, vx, vy, vz, a)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dot_r, dot_theta, dot_phi, a, mass)
    
    # --- DOCTOR SENSORS ---
    stats = np.zeros(NUM_DOCTOR_METRICS, dtype=np.float64)
    stats[10] = E
    stats[11] = L
    stats[12] = Q
    stats[15] = L / E if E != 0 else 0
    stats[16] = Q / (E*E) if E != 0 else 0

    min_r = r
    max_r = r
    min_delta = r*r - 2.0*mass*r + a*a
    min_pole_gap = theta if theta < np.pi/2.0 else np.pi - theta
    
    orbit_phi = 0.0
    theta_turns = 0
    prev_dth = 0.0
    steps_in_ergo = 0
    entered_ergo = 0
    eq_crossings = 0
    max_dH = 0.0
    max_dQ = 0.0
    max_dE = 0.0
    max_dL = 0.0
    orbit_phi_signed = 0.0
    max_dphi_step = 0.0
    max_abs_dphdlam = 0.0
    max_inv_sin2    = 0.0

    captured = False
    termination_reason = 0
    hit_count = 0

    pi_2 = np.pi / 2.0
    capture_radius = r_outer_horizon * 1.05

    for i in range(max_steps):
        # ── PRE-STEP CAPTURE CHECK ─────────────────────────────────────────
        # Avoids evaluating derivatives at r < capture_radius where
        # Delta is negative and inv_Delta would be large even with softening.
        old_r, old_theta, old_phi = r, theta, phi
        
        r_norm    = (r - r_outer_horizon) / (r_outer_horizon * 4.0)
        r_factor  = r_norm if r_norm < 1.0 else 1.0
        r_factor  = r_factor if r_factor > 0.0 else 0.0

        sin2      = np.sin(theta) ** 2
        gap       = theta if theta < pi_2 else np.pi - theta
        theta_factor = sin2 / (sin2 + 0.05)
        pole_factor = gap / (gap + 0.05)

        dphi_scale = abs(L) / (sin2 + 1e-6)
        polar_cap = 0.15 / dphi_scale if dphi_scale > 0.15 else 1.0
        dt_local  = dt * min(max(r_factor * theta_factor * pole_factor, 1e-4), polar_cap)
        dt_half   = dt_local * 0.5

        if r < capture_radius:
            captured = True
            termination_reason = 1
            break
        
        # RK4
        dr1, dth1, dph1, dpr1, dpth1, dt1 = _kerr_derivatives(r, theta, phi, pr, ptheta, E, L, Q, a, mass)
        abs_dph1 = abs(dph1)
        if abs_dph1 > max_abs_dphdlam: max_abs_dphdlam = abs_dph1
        sin_t1 = np.sin(theta)
        inv_sin2_1 = 1.0 / (sin_t1*sin_t1 + 1e-7)
        if inv_sin2_1 > max_inv_sin2: max_inv_sin2 = inv_sin2_1

        r2 = r + dr1*dt_half
        if r2 < capture_radius: captured = True; termination_reason = 1; break
        th2 = theta + dth1*dt_half; ph2 = phi + dph1*dt_half
        pr2 = pr + dpr1*dt_half; pth2 = ptheta + dpth1*dt_half
        dr2, dth2, dph2, dpr2, dpth2, dt2 = _kerr_derivatives(r2, th2, ph2, pr2, pth2, E, L, Q, a, mass)
        abs_dph2 = abs(dph2)
        if abs_dph2 > max_abs_dphdlam: max_abs_dphdlam = abs_dph2
        sin_t2 = np.sin(th2)
        inv_sin2_2 = 1.0 / (sin_t2*sin_t2 + 1e-7)
        if inv_sin2_2 > max_inv_sin2: max_inv_sin2 = inv_sin2_2
        
        r3 = r + dr2*dt_half
        if r3 < capture_radius: captured = True; termination_reason = 1; break
        th3 = theta + dth2*dt_half; ph3 = phi + dph2*dt_half
        pr3 = pr + dpr2*dt_half; pth3 = ptheta + dpth2*dt_half
        dr3, dth3, dph3, dpr3, dpth3, dt3 = _kerr_derivatives(r3, th3, ph3, pr3, pth3, E, L, Q, a, mass)
        abs_dph3 = abs(dph3)
        if abs_dph3 > max_abs_dphdlam: max_abs_dphdlam = abs_dph3
        sin_t3 = np.sin(th3)
        inv_sin2_3 = 1.0 / (sin_t3*sin_t3 + 1e-7)
        if inv_sin2_3 > max_inv_sin2: max_inv_sin2 = inv_sin2_3

        r4 = r + dr3*dt_local
        if r4 < capture_radius: captured = True; termination_reason = 1; break
        th4 = theta + dth3*dt_local; ph4 = phi + dph3*dt_local
        pr4 = pr + dpr3*dt_local; pth4 = ptheta + dpth3*dt_local
        dr4, dth4, dph4, dpr4, dpth4, dt4 = _kerr_derivatives(r4, th4, ph4, pr4, pth4, E, L, Q, a, mass)
        abs_dph4 = abs(dph4)
        if abs_dph4 > max_abs_dphdlam: max_abs_dphdlam = abs_dph4
        sin_t4 = np.sin(th4)
        inv_sin2_4 = 1.0 / (sin_t4*sin_t4 + 1e-7)
        if inv_sin2_4 > max_inv_sin2: max_inv_sin2 = inv_sin2_4

        r      += (dt_local / 6.0) * (dr1 + 2*dr2 + 2*dr3 + dr4)
        theta  += (dt_local / 6.0) * (dth1 + 2*dth2 + 2*dth3 + dth4)
        phi    += (dt_local / 6.0) * (dph1 + 2*dph2 + 2*dph3 + dph4)
        pr     += (dt_local / 6.0) * (dpr1 + 2*dpr2 + 2*dpr3 + dpr4)
        ptheta += (dt_local / 6.0) * (dpth1 + 2*dpth2 + 2*dpth3 + dpth4)
        
        while theta < 0.0 or theta > np.pi:
            if theta < 0.0: theta = -theta; ptheta = -ptheta; phi += np.pi
            elif theta > np.pi: theta = 2.0 * np.pi - theta; ptheta = -ptheta; phi += np.pi
            
        ergo_bound = mass + np.sqrt(max(mass*mass - a*a*np.cos(theta)**2, 0.0))
        if r < ergo_bound:
            entered_ergo = 1
            steps_in_ergo += 1
        
        if r < capture_radius:
            captured = True
            termination_reason = 1
            break

        # --- SENSOR LOGIC ---
        if r < min_r: min_r = r
        if r > max_r: max_r = r
        
        cur_delta = r*r - 2.0*mass*r + a*a
        if cur_delta < min_delta: min_delta = cur_delta
            
        gap = theta if theta < pi_2 else np.pi - theta
        if gap < min_pole_gap: min_pole_gap = gap

        # FIX 1: Exact continuous phase tracking (ignores Pi jumps from coordinate bounding)
        dphi_step = (dt_local / 6.0) * (dph1 + 2*dph2 + 2*dph3 + dph4)
        orbit_phi += abs(dphi_step)
        
        if dth1 * prev_dth < 0.0: theta_turns += 1
        prev_dth = dth1
        
        # FIX 2: Hamiltonian Validator (Radial Drift)
        inv_Sigma = 1.0 / (r*r + a*a * np.cos(theta)**2 + 1e-9)
        H_val = 0.5 * inv_Sigma * (cur_delta*pr*pr + ptheta*ptheta - ((E*(r*r+a*a)-a*L)**2)/(cur_delta+1e-9) + ((L-a*E*np.sin(theta)**2)**2)/(np.sin(theta)**2 + 1e-9))
        if abs(H_val) > max_dH: max_dH = abs(H_val)
            
        # FIX 3: Carter Constant Validator (Polar Drift)
        sin_t_safe = max(abs(np.sin(theta)), 1e-3)
        cur_Q = ptheta*ptheta + np.cos(theta)**2 * ( (L*L)/(sin_t_safe**2) - a*a * E*E )
        dQ = abs(cur_Q - Q)
        if dQ > max_dQ: max_dQ = dQ
        _, _, dph_chk, _, _, dt_chk = _kerr_derivatives(
            r, theta, phi, pr, ptheta, E, L, Q, a, mass
        )
        sin_t_chk = np.sin(theta)
        cos_t_chk = np.cos(theta)
        Sigma_chk = r*r + a*a*cos_t_chk*cos_t_chk
        g_tt_chk    = -(1.0 - 2.0*mass*r/Sigma_chk)
        g_tphi_chk  = -2.0*mass*a*r * sin_t_chk**2 / Sigma_chk
        g_phiphi_chk = (r*r + a*a + 2.0*mass*a*a*r * sin_t_chk**2 / Sigma_chk) * sin_t_chk**2

        p_t_chk   = g_tt_chk * dt_chk + g_tphi_chk * dph_chk
        p_phi_chk = g_tphi_chk * dt_chk + g_phiphi_chk * dph_chk
        E_chk = -p_t_chk
        L_chk = p_phi_chk

        dE = abs(E_chk - E)
        dL = abs(L_chk - L)
        if dE > max_dE: max_dE = dE
        if dL > max_dL: max_dL = dL

        # Signed orbit count + max single-step dphi
        orbit_phi_signed += dphi_step
        abs_dphi = abs(dphi_step)
        if abs_dphi > max_dphi_step: max_dphi_step = abs_dphi

        if (not np.isfinite(r) or not np.isfinite(theta) or not np.isfinite(phi) or 
            not np.isfinite(pr) or not np.isfinite(ptheta) or abs(r - old_r) > 2.0):
            captured = True
            termination_reason = 2
            break

        if (old_theta - pi_2) * (theta - pi_2) <= 0.0:
            d_theta = theta - old_theta
            if d_theta != 0.0:
                # FIX 4: Equatorial Crossings independently tracked from disk hits
                eq_crossings += 1 
                
                t_frac = (pi_2 - old_theta) / d_theta
                hit_r = old_r + t_frac * (r - old_r)
                
                if (disk_inner) <= hit_r <= (disk_outer):
                    if hit_count == 0: stats[22] = hit_r
                    elif hit_count == 1: stats[23] = hit_r
                    elif hit_count == 2: stats[24] = hit_r
                    hit_count += 1

        if not (r <= sim_bounds): termination_reason = 3; break

    if termination_reason == 0:
        termination_reason = 4
    if termination_reason == 4:
       if r<5.0*r_outer_horizon:
          termination_reason = 5

    # Pack Data (mapped to core.indices)
    stats[0]  = 1.0 if captured else 0.0
    stats[1]  = termination_reason
    stats[2]  = float(i + 1)
    stats[3]  = min_r
    stats[4]  = max_r
    stats[5]  = r
    stats[6]  = theta
    stats[7]  = orbit_phi / (2.0 * np.pi)
    stats[8]  = eq_crossings
    stats[9]  = hit_count
    # Note: IDX_MAX_DQ (13) and IDX_MAX_DH (14) were swapped in indices.py
    stats[13] = max_dQ
    stats[14] = max_dH
    # stats[15..16] set earlier: impact param and Carter const
    stats[17] = min_pole_gap
    stats[18] = theta_turns
    stats[19] = steps_in_ergo
    stats[20] = entered_ergo
    stats[21] = min_delta
    stats[25] = max_dE
    stats[26] = max_dL
    stats[27] = orbit_phi_signed / (2.0 * np.pi)
    stats[28] = max_dphi_step
    stats[29] = max_abs_dphdlam
    stats[30] = max_inv_sin2

    return stats
