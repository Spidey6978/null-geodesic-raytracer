"""
blender/bh_install.py
Run this in Blender's Scripting workspace to verify the repo is importable.
If this works, the addon will work.
"""
import sys
import os

REPO_ROOT = "C:/dev stuff/projects/python/black-hole-sim"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Testing raytracer imports from Blender Python...")

try:
    from core.constants import SPIN, MASS, R_OUTER_HORIZON, DISK_INNER, RS
    print(f"  constants OK — SPIN={SPIN:.4f}  R_horizon={R_OUTER_HORIZON:.4f}")
except Exception as e:
    print(f"  constants FAILED: {e}")

try:
    import numpy as np
    print(f"  numpy OK — {np.__version__}")
except Exception as e:
    print(f"  numpy FAILED: {e}")

try:
    import numba
    print(f"  numba OK — {numba.__version__}")
except Exception as e:
    print(f"  numba FAILED: {e}")

try:
    from scripts.render_kernel import render_pixel_batch
    print("  render_kernel OK")
except Exception as e:
    print(f"  render_kernel FAILED: {e}")

print("\nIf all OK above, install bh_render_engine.py as a Blender addon.")
print("Edit > Preferences > Add-ons > Install > select bh_render_engine.py")