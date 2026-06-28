"""
doctor/tools/pixel_probe.py

Serial, bounds-checked diagnostic probe for a small window of pixels.
Bypasses the parallel tensor pipeline entirely, which is useful for ground-truth
verification when the full render shows suspicious patterns (e.g. the
TV-static corruption symptom of a stale Numba cache / index mismatch).

Usage:
    python doctor/tools/pixel_probe.py --probe-x 480 --probe-y 270 --radius 10
"""
import os

os.environ["NUMBA_BOUNDSCHECK"] = "1"   # force bounds checking for this run

import sys
import argparse
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.camera import generate_camera_rays
from core.geodesics import integrate_path_doctor
from core.indices import (
    IDX_CAPTURED,
    IDX_MIN_DELTA,
    IDX_TERM_REASON,
    IDX_MIN_R,
    IDX_MAX_DQ,
    IDX_MAX_DH,
    IDX_MIN_POLE_GAP,
    IDX_MAX_DE,
    IDX_MAX_DL,
    IDX_MAX_DPHI_STEP,
    IDX_MAX_ABS_DPHDLAM,
    IDX_MAX_INV_SIN2,
    NUM_DOCTOR_METRICS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width",  type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fov",    type=float, default=100.0)
    p.add_argument("--roll",   type=float, default=0.0)
    p.add_argument("--cam-pos",  nargs=3, type=float, default=[0.0, 0.0, 20.0])
    p.add_argument("--look-at",  nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--dt",        type=float, default=0.2)
    p.add_argument("--max-steps", type=int,   default=5000)
    p.add_argument("--mass",      type=float, default=1.0)
    p.add_argument("--spin",      type=float, default=0.998)
    p.add_argument("--probe-x",   type=int,   required=True)
    p.add_argument("--probe-y",   type=int,   required=True)
    p.add_argument("--radius",    type=int,   default=10)
    args = p.parse_args()

    if args.width <= 0 or args.height <= 0:
        p.error("--width and --height must be positive")
    if args.radius < 0:
        p.error("--radius must be non-negative")
    if not (0 <= args.probe_x < args.width and 0 <= args.probe_y < args.height):
        p.error("--probe-x and --probe-y must identify a pixel inside the image")

    a = args.spin * args.mass
    sqrt_term = (args.mass**2 - a**2) ** 0.5 if args.mass**2 >= a**2 else 0.0
    r_outer_horizon = args.mass + sqrt_term

    if abs(a) < 1e-10:
        disk_inner = 6.0 * args.mass
    else:
        Z1 = 1.0 + (1.0 - (a/args.mass)**2)**(1/3) * (
             (1.0 + a/args.mass)**(1/3) + (1.0 - a/args.mass)**(1/3))
        Z2 = (3.0*(a/args.mass)**2 + Z1**2)**0.5
        disk_inner = args.mass * (3.0 + Z2 - ((3.0-Z1)*(3.0+Z1+2.0*Z2))**0.5)

    disk_outer = 18.0 * (2.0 * args.mass)
    sim_bounds = 200.0 * (2.0 * args.mass)

    print(f"NUM_DOCTOR_METRICS = {NUM_DOCTOR_METRICS}")
    print(f"Probing ({args.probe_x},{args.probe_y}) +/- {args.radius}, "
          f"bounds-checking ON\n")

    ray_dirs = generate_camera_rays(
        args.width, args.height, args.fov,
        list(args.cam_pos), list(args.look_at), args.roll
    )
    cam_pos = np.array(args.cam_pos, dtype=np.float64)

    rows = []
    x0, x1 = args.probe_x - args.radius, args.probe_x + args.radius
    y0, y1 = args.probe_y - args.radius, args.probe_y + args.radius

    for y in range(max(0, y0), min(args.height, y1 + 1)):
        for x in range(max(0, x0), min(args.width, x1 + 1)):
            stats = integrate_path_doctor(
                cam_pos, ray_dirs[y, x], args.dt, args.max_steps,
                args.mass, a, r_outer_horizon, disk_inner, disk_outer, sim_bounds
            )
            
            max_P_over_D, max_K_over_D = _probe_horizon_ratios(
                list(cam_pos), list(ray_dirs[y, x]), args.dt, args.max_steps,
                args.mass, a, r_outer_horizon, disk_outer, sim_bounds
            )

            rows.append({
                "x": x, "y": y,
                "captured":      stats[IDX_CAPTURED],
                "term_reason":   stats[IDX_TERM_REASON],
                "min_r":         stats[IDX_MIN_R],
                "max_dphi_step": stats[IDX_MAX_DPHI_STEP],
                "max_dphdlam":   stats[IDX_MAX_ABS_DPHDLAM],
                "max_inv_sin2":  stats[IDX_MAX_INV_SIN2],
                "max_dE":        stats[IDX_MAX_DE],
                "max_dL":        stats[IDX_MAX_DL],
                "max_dQ":        stats[IDX_MAX_DQ],
                "max_dH":        stats[IDX_MAX_DH],
                "min_pole_gap":  stats[IDX_MIN_POLE_GAP],
                "min_delta":     stats[IDX_MIN_DELTA],
                "max_P_over_D":  max_P_over_D,
                "max_K_over_D":  max_K_over_D,
            })

    print(f"Collected {len(rows)} pixels, no crashes (bounds OK if you see this)\n")

    print(f"{'x':>4} {'y':>4} {'term':>5} {'min_r':>8} "
          f"{'max_dphdlam':>12} {'inv_sin2':>10} {'dE':>10} {'dL':>10}")
    for r in rows[:25]:   # print a sample, not the whole block
        print(f"{r['x']:>4} {r['y']:>4} {int(r['term_reason']):>5} "
              f"{r['min_r']:>8.3f} {r['max_dphdlam']:>12.3e} "
              f"{r['max_inv_sin2']:>10.3e} {r['max_dE']:>10.3e} {r['max_dL']:>10.3e}")
    worst_dphi = sorted(
        rows,
        key=lambda r: r["max_dphdlam"],
        reverse=True
    )

    print("\nTOP 10 max_dphdlam")
    for r in worst_dphi[:10]:
        print(
            f"({r['x']},{r['y']}) "
            f"dphdlam={r['max_dphdlam']:.3e} "
            f"invsin2={r['max_inv_sin2']:.3e} "
            f"dL={r['max_dL']:.3e}"
        )
    center = min(
        rows,
        key=lambda r: (r["x"] - args.probe_x)**2 + (r["y"] - args.probe_y)**2
    )

    print("\nCENTER PIXEL")
    for k, v in center.items():
        print(f"{k:15s}: {v}")
    
    worst_dL = sorted(
        rows,
        key=lambda r: r["max_dL"],
        reverse=True
    )

    print("\nTOP 10 dL drift")
    for r in worst_dL[:10]:
        print(
            f"({r['x']},{r['y']}) "
            f"dL={r['max_dL']:.3e}"
        )

    # Summary stats across the whole probed window.
    def col(name):
        return np.array([r[name] for r in rows])

    dphi = col("max_dphdlam")
    invs = col("max_inv_sin2")
    dL   = col("max_dL")
    dE   = col("max_dE")

    def print_correlation(label, left, right):
        if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
            print(f"{label} = n/a (need varying samples)")
        else:
            print(f"{label} =", np.corrcoef(left, right)[0, 1])

    print("\nCORRELATIONS")
    print_correlation("corr(dphdlam, invsin2)", dphi, invs)
    print_correlation("corr(dphdlam, dL)     ", dphi, dL)
    print_correlation("corr(dphdlam, dE)     ", dphi, dE)

    print("\nSUMMARY (min / max across probed window)")
    for field in ["min_r", "max_dphi_step", "max_dphdlam",
              "max_inv_sin2", "max_dE", "max_dL", "max_dQ",
              "max_dH", "min_pole_gap", "min_delta",
              "max_P_over_D", "max_K_over_D"]:
        v = col(field)
        print(f"  {field:<15} {v.min():>14.6e}   {v.max():>14.6e}")
        
def _probe_horizon_ratios(cam_pos, ray_dir, dt, max_steps, mass, a, 
                           r_outer_horizon, disk_outer, sim_bounds):
    """
    Serial single-ray probe that tracks max |P/Delta| and max |K/Delta|.
    Runs outside Numba so we can inspect per-step values freely.
    """
    from core.geodesics import _cartesian_to_bl, _compute_conserved_quantities
    
    px, py, pz = cam_pos
    vx, vy, vz = ray_dir
    speed = (vx**2 + vy**2 + vz**2)**0.5
    if speed > 0:
        vx /= speed; vy /= speed; vz /= speed

    r, theta, phi, dr, dth, dph = _cartesian_to_bl(px, py, pz, vx, vy, vz, a)
    E, L, Q, pr, ptheta = _compute_conserved_quantities(r, theta, dr, dth, dph, a, mass)

    capture_radius = r_outer_horizon + 0.002
    max_P_over_D = 0.0
    max_K_over_D = 0.0

    for _ in range(max_steps):
        if r < capture_radius:
            break
        
        Delta = r*r - 2.0*mass*r + a*a
        if abs(Delta) > 1e-12:
            P = E * (r*r + a*a) - a * L
            K = Q + (a*E - L)**2
            P_over_D = abs(P / Delta)
            K_over_D = abs(K / Delta)
            if P_over_D > max_P_over_D: max_P_over_D = P_over_D
            if K_over_D > max_K_over_D: max_K_over_D = K_over_D

        # minimal RK4 step — just enough to advance r
        from core.geodesics import _kerr_derivatives
        dr1,dth1,dph1,dpr1,dpth1,_ = _kerr_derivatives(r,theta,phi,pr,ptheta,E,L,Q,a,mass)
        dt_local = dt
        if r < 5.0: dt_local *= 0.5
        if r < 3.0: dt_local *= 0.25
        if r < 2.0: dt_local *= 0.1
        dt_h = dt_local * 0.5

        r2=r+dr1*dt_h; th2=theta+dth1*dt_h
        pr2=pr+dpr1*dt_h; pth2=ptheta+dpth1*dt_h
        ph2=phi+dph1*dt_h
        dr2,dth2,dph2,dpr2,dpth2,_ = _kerr_derivatives(r2,th2,ph2,pr2,pth2,E,L,Q,a,mass)

        r3=r+dr2*dt_h; th3=theta+dth2*dt_h
        pr3=pr+dpr2*dt_h; pth3=ptheta+dpth2*dt_h
        ph3=phi+dph2*dt_h
        dr3,dth3,dph3,dpr3,dpth3,_ = _kerr_derivatives(r3,th3,ph3,pr3,pth3,E,L,Q,a,mass)

        r4=r+dr3*dt_local; th4=theta+dth3*dt_local
        pr4=pr+dpr3*dt_local; pth4=ptheta+dpth3*dt_local
        ph4=phi+dph3*dt_local
        dr4,dth4,dph4,dpr4,dpth4,_ = _kerr_derivatives(r4,th4,ph4,pr4,pth4,E,L,Q,a,mass)

        r      += (dt_local/6.0)*(dr1+2*dr2+2*dr3+dr4)
        theta  += (dt_local/6.0)*(dth1+2*dth2+2*dth3+dth4)
        phi    += (dt_local/6.0)*(dph1+2*dph2+2*dph3+dph4)
        pr     += (dt_local/6.0)*(dpr1+2*dpr2+2*dpr3+dpr4)
        ptheta += (dt_local/6.0)*(dpth1+2*dpth2+2*dpth3+dpth4)

        while theta < 0.0 or theta > np.pi:
            if theta < 0.0: theta = -theta; ptheta = -ptheta; phi += np.pi
            elif theta > np.pi: theta = 2.0*np.pi-theta; ptheta=-ptheta; phi+=np.pi

        if not (r <= sim_bounds): break

    return max_P_over_D, max_K_over_D


if __name__ == "__main__":
    main()
