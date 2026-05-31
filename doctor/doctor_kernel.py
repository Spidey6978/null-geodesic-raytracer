"""
doctor_kernel.py — Simple diagnostic renderer that color-codes pixels
by the photon `termination_reason` to help spot stray pixels.

Usage: python scripts/doctor_kernel.py [--width W] [--height H]

This script is intentionally minimal: no stars, no fog, just a color
per-pixel based on the integrator's termination reason.
"""
import os
import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange

# Ensure repo root is on sys.path so `from core...` works when running
# this script from the repository root or the scripts/ folder.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.camera import generate_camera_rays
from core.geodesics import integrate_path_lean, integrate_path


def _term_color(term):
    # Color mapping for termination reasons:
    # 0 -> black (unknown), 1 -> red (captured), 2 -> magenta (non-finite),
    # 3 -> yellow (out of bounds), 4 -> cyan (natural completion)
    cmap = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 0.0, 1.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 0.8, 1.0),
    }
    return cmap.get(int(term), (0.5, 0.5, 0.5))


def main():
    p = argparse.ArgumentParser(description="Doctor Kernel — termination_reason visualiser")
    # Default to 80% of render_kernel's 960x540 (=> 768x432) for meaningful diagnostics
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=432)
    p.add_argument('--fov', type=float, default=100.0)
    # Match render_kernel integrator settings by default for comparability
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--outfile', type=str, default='doctor_kernel.png')
    p.add_argument('--cam-pos', nargs=3, type=float, default=[37.5, 0.4, 18.0])
    p.add_argument('--look-at', nargs=3, type=float, default=[-3.0, -1.0, 0.0])
    p.add_argument('--roll', type=float, default=-14.0)
    p.add_argument('--use-full', action='store_true', default=False,
                   help='Use the full `integrate_path` integrator (slower) instead of the lean integrator')
    args = p.parse_args()

    W = args.width; H = args.height
    FOV = args.fov
    CAM_POS = np.array(args.cam_pos, dtype=np.float64)
    LOOK_AT = np.array(args.look_at, dtype=np.float64)
    ROLL = args.roll
    dt = args.dt; max_steps = args.max_steps

    print(f"Doctor Kernel: {W}x{H}, dt={dt}, max_steps={max_steps}")

    # Generate ray directions (normalized)
    ray_dirs = generate_camera_rays(W, H, FOV, list(CAM_POS), list(LOOK_AT), roll_degrees=ROLL)

    # JIT warm-up (first call will compile the chosen integrator) — small steps
    print("Warming up integrator JIT (may take a few seconds)...")
    try:
        if args.use_full:
            _ = integrate_path(CAM_POS, ray_dirs[H//2, W//2], dt, 2)
        else:
            _ = integrate_path_lean(CAM_POS, ray_dirs[H//2, W//2], dt, 2)
    except Exception:
        # Ignore warmup failures — we'll handle per-pixel exceptions below
        pass

    image = np.zeros((H, W, 3), dtype=np.float64)

    # Numba-parallel batch renderer: calls the integrator per-pixel and writes color by termination reason.
    @njit(parallel=True, cache=True)
    def _doctor_batch(ray_dirs_local, cam_pos_local, dt_local, max_steps_local, use_full_local, out_img):
        h = ray_dirs_local.shape[0]
        w = ray_dirs_local.shape[1]
        counts = np.zeros(5, np.int64)
        for idx in prange(h * w):
            y = idx // w
            x = idx - y * w
            # Call the chosen integrator
            if use_full_local == 1:
                # integrate_path: returns path, steps_taken, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason
                _, _, _, _, _, _, _, term = integrate_path(cam_pos_local, ray_dirs_local[y, x], dt_local, max_steps_local)
            else:
                # integrate_path_lean: final_dir, captured, hit_count, hit_radii, hit_phis, hit_vels, termination_reason
                _, _, _, _, _, _, term = integrate_path_lean(cam_pos_local, ray_dirs_local[y, x], dt_local, max_steps_local)

            t = int(term)
            if t < 0 or t > 4:
                t = 0

            if t == 0:
                r = 0.0; g = 0.0; b = 0.0
            elif t == 1:
                r = 1.0; g = 0.0; b = 0.0
            elif t == 2:
                r = 1.0; g = 0.0; b = 1.0
            elif t == 3:
                r = 1.0; g = 1.0; b = 0.0
            else:
                r = 0.0; g = 0.8; b = 1.0

            out_img[y, x, 0] = r
            out_img[y, x, 1] = g
            out_img[y, x, 2] = b
            counts[t] += 1

        return counts

    print("Running Numba-parallel diagnostic batch (this compiles the JIT)...")
    t0 = time.time()
    counts_arr = _doctor_batch(ray_dirs, CAM_POS, dt, max_steps, 1 if args.use_full else 0, image)
    elapsed = time.time() - t0

    counts = {int(i): int(counts_arr[i]) for i in range(5)}
    print("Counts by termination reason:", counts)
    print(f"Elapsed: {elapsed:.2f}s — saving {args.outfile}")
    plt.imsave(args.outfile, image)


if __name__ == '__main__':
    main()
