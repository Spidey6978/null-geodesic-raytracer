"""
doctor/render.py — Diagnostic render dispatcher

Usage:
    python doctor/render.py --mode orbit_count
    python doctor/render.py --mode termination_map --width 480 --height 270
    python doctor/render.py --mode h_drift --cam-pos 6.5 0.4 18.0 --dt 0.1
    python doctor/render.py --modes   (list all available modes)
"""

import json
import os
import sys
import argparse as _ap
import importlib
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.camera      import generate_camera_rays
from doctor.diagnostics import collect_diagnostics
from doctor.utils.render_base import compute_tensor
from core.indices import *


# ── Mode registry ─────────────────────────────────────────────────────────────
MODES = {
    "capture_map":      "doctor.modes.capture_map",
    "termination_map":  "doctor.modes.termination_map",
    "orbit_count":      "doctor.modes.orbit_count",
    "hit_count":        "doctor.modes.hit_count",
    "min_r":            "doctor.modes.min_r_map",
    "h_drift":          "doctor.modes.h_drift_map",
    "ergosphere":       "doctor.modes.ergosphere_map",
    "steps":            "doctor.modes.steps_map",
    "impact_param":     "doctor.modes.impact_parameter",
    "min_pole_gap":     "doctor.modes.min_pole_gap",
    "e_drift":          "doctor.modes.e_drift_map",
    "l_drift":          "doctor.modes.l_drift_map",
    "orbit_count_signed": "doctor.modes.orbit_count_signed",
    "max_dphi_step":    "doctor.modes.max_dphi_step",
    "max_dphdlam":      "doctor.modes.max_dphdlam",
    "max_inv_sin2":     "doctor.modes.max_inv_sin2"
}


# ── Colormap application ──────────────────────────────────────────────────────

def apply_colormap(values_2d, config):
    """
    Converts a 2D float array into an RGB image using the mode's config.
    Handles both continuous (matplotlib colormap) and discrete (dict) modes.
    """
    cmap_spec = config["colormap"]

    # Discrete colormap — dict mapping int value to RGB tuple
    if isinstance(cmap_spec, dict):
        h, w = values_2d.shape
        img = np.zeros((h, w, 3), dtype=np.float64)
        rounded = np.round(values_2d).astype(np.int32)
        for val, color in cmap_spec.items():
            mask = (rounded == val)
            img[mask] = color
        return img

    # Continuous colormap — matplotlib colormap name string
    vmin  = config.get("vmin", float(np.nanmin(values_2d)))
    vmax  = config.get("vmax", float(np.nanmax(values_2d)))
    scale = config.get("scale", "linear")

    data = values_2d.copy()

    if scale == "log":
        positive = data[data > 0]
        if positive.size > 0:
            floor = float(np.nanmin(positive)) * 0.01
        else:
            floor = 1e-10
        data = np.where(data > 0, data, floor)
        data = np.log10(data)
        vmin = np.log10(max(vmin, 1e-10))
        vmax = np.log10(max(vmax, 1e-10))

    # Normalise to [0, 1]
    span = vmax - vmin
    if span < 1e-12:
        span = 1.0
    normalised = np.clip((data - vmin) / span, 0.0, 1.0)
    normalised = np.nan_to_num(normalised, nan=0.5, posinf=1.0, neginf=0.0)

    cmap = mcm.get_cmap(cmap_spec)
    rgb  = cmap(normalised)[:, :, :3]   # drop alpha channel
    return rgb


# ── Main dispatcher ───────────────────────────────────────────────────────────

def main():
    parser = _ap.ArgumentParser(
        description="Doctor — diagnostic visualiser for the Kerr raytracer",
        formatter_class=_ap.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help="Diagnostic mode to run. Use --modes to list all available."
    )
    parser.add_argument(
        "--modes", action="store_true",
        help="List all available diagnostic modes and exit."
    )

    # Render settings
    parser.add_argument("--width",     type=int,   default=960)
    parser.add_argument("--height",    type=int,   default=540)
    parser.add_argument("--fov",       type=float, default=100.0)
    parser.add_argument("--roll",      type=float, default=0.0)
    parser.add_argument("--dt",        type=float, default=0.2)
    parser.add_argument("--max-steps", type=int,   default=5000)
    
    # Tracking limits
    parser.add_argument("--lambda-max", type=float, default=500.0, 
                        help="Max affine parameter distance")

    # Physics Overrides (Moved from the Pre-Parser directly into the main runtime args)
    parser.add_argument("--spin",       type=float, default=0.998)
    parser.add_argument("--mass",       type=float, default=1.0)
    parser.add_argument("--disk-inner", type=float, default=None)
    parser.add_argument("--disk-outer", type=float, default=None)

    # Camera
    parser.add_argument(
        "--cam-pos", nargs=3, type=float,
        default=[0.0, 0.0, 20.0],
        metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--look-at", nargs=3, type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z")
    )

    # Output
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output filename. Defaults to <mode>.png in doctor/output/"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show the image interactively after saving."
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Override colormap minimum value."
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Override colormap maximum value."
    )

    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.mass <= 0.0:
        parser.error("--mass must be positive")
    if abs(args.spin) > 1.0:
        parser.error("--spin must be in the dimensionless range [-1, 1]")
    if (
        args.disk_inner is not None and args.disk_inner <= 0.0
        or args.disk_outer is not None and args.disk_outer <= 0.0
    ):
        parser.error("--disk-inner and --disk-outer must be positive when provided")
    if (
        args.disk_inner is not None and args.disk_outer is not None
        and args.disk_outer <= args.disk_inner
    ):
        parser.error("--disk-outer must be greater than --disk-inner")

    # ── List modes and exit ───────────────────────────────────────────────────
    if args.modes:
        print("\nAvailable diagnostic modes:")
        for name in sorted(MODES.keys()):
            mod = importlib.import_module(MODES[name])
            label = mod.CONFIG.get("label", name)
            print(f"  {name:<20}  {label}")
        print()
        return

    if args.mode is None:
        parser.error("--mode is required. Use --modes to see options.")

    if args.mode not in MODES:
        parser.error(
            f"Unknown mode '{args.mode}'. "
            f"Available: {', '.join(sorted(MODES.keys()))}"
        )

    # ── Load the mode ─────────────────────────────────────────────────────────
    print(f"🔬  Mode: {args.mode}")
    module    = importlib.import_module(MODES[args.mode])
    get_value = module.get_value
    config    = module.CONFIG.copy()

    # CLI overrides for colormap range
    if args.vmin is not None: config["vmin"] = args.vmin
    if args.vmax is not None: config["vmax"] = args.vmax

    # ── Compute the diagnostic tensor ─────────────────────────────────────────
    CAM_POS = np.array(args.cam_pos, dtype=np.float64)
    LOOK_AT = np.array(args.look_at, dtype=np.float64)

    print(f"    {args.width}×{args.height}  dt={args.dt}  max_steps={args.max_steps}")
    print(f"    cam={args.cam_pos}  look_at={args.look_at}  spin={args.spin}")

    t0     = time.time()
    # Explicitly pass the dynamic physics variables to compute_tensor
    tensor = compute_tensor(
        args.width, args.height,
        args.fov, CAM_POS, LOOK_AT,
        dt=args.dt, max_steps=args.max_steps,
        roll=args.roll, spin=args.spin, mass=args.mass,
        disk_inner=args.disk_inner, disk_outer=args.disk_outer
    )
    t1 = time.time()
    print(f"✅  Tensor computed in {t1-t0:.1f}s")

    # ── Extract the layer this mode cares about ───────────────────────────────
    H, W = args.height, args.width
    values = np.zeros((H, W), dtype=np.float64)

    if hasattr(module, "get_value_from_raw"):
        # Fast path — mode operates directly on the raw stats array
        get_raw = module.get_value_from_raw
        for y in range(H):
            for x in range(W):
                values[y, x] = get_raw(tensor[y, x])
    else:
        # Standard path — mode uses DoctorData
        for y in range(H):
            for x in range(W):
                data = collect_diagnostics(tensor[y, x])
                values[y, x] = get_value(data)

    # ── Apply colormap and save ───────────────────────────────────────────────
    img = apply_colormap(values, config)

    outdir = os.path.join(REPO_ROOT, "doctor", "outputs")
    os.makedirs(outdir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{args.mode}_a{args.spin:.3f}_{timestamp}.png"
    outfile = args.out or os.path.join(outdir, default_name)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="black")
    ax.imshow(img, origin="upper", aspect="auto")
    ax.axis("off")
    ax.set_title(
        f"Doctor — {config.get('label', args.mode)}\n"
        f"{args.width}×{args.height}  dt={args.dt}  max_steps={args.max_steps}  "
        f"a={args.spin}  M={args.mass}",
        color="white", fontsize=10, pad=8
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"💾  Saved → {outfile}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        
    meta = {
        "mode":       args.mode,
        "label":      config.get("label", args.mode),
        "spin":       args.spin,
        "mass":       args.mass,
        "cam_pos":    args.cam_pos,
        "look_at":    args.look_at,
        "fov":        args.fov,
        "dt":         args.dt,
        "max_steps":  args.max_steps,
        "disk_inner": args.disk_inner,
        "disk_outer": args.disk_outer,
        "vmin":       config.get("vmin"),
        "vmax":       config.get("vmax"),
        "cli":        " ".join(sys.argv),   # full command that produced this
        "compute_time_s": t1 - t0,
    }
    
    meta_path = str(Path(outfile).with_suffix(".json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"📋  Metadata → {meta_path}")


if __name__ == "__main__":
    main()
