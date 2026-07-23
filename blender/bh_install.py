"""
blender/bh_install.py
Run this in Blender's Scripting workspace to verify the repo is importable.
If this works, the addon will work.
"""
import sys
import os
import importlib

REPO_ROOT = "C:/dev stuff/projects/python/black-hole-sim"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Testing raytracer imports from Blender Python...")
print(f"Python executable: {sys.executable}")
print(f"Python prefix: {sys.prefix}")
print(f"sys.path entries: {len(sys.path)}")

# Core project imports.
try:
    from core.constants import SPIN, MASS, R_OUTER_HORIZON, DISK_INNER, RS
    print(f"  constants OK — SPIN={SPIN:.4f}  R_horizon={R_OUTER_HORIZON:.4f}")
except Exception as e:
    print(f"  constants FAILED: {e}")

# Packages that support Blender integration and the renderer.
package_imports = {
    "numpy": ["numpy"],
    "numba": ["numba"],
    "matplotlib": ["matplotlib"],
    "scipy": ["scipy"],
}

for package_name, modules in package_imports.items():
    imported = False
    last_error = None
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  {package_name} OK — imported {module_name}")
            imported = True
            break
        except Exception as e:
            last_error = e
    if not imported:
        print(f"  {package_name} FAILED: {last_error}")

try:
    from scripts.render_kernel import render_pixel_batch
    print("  render_kernel OK")
except Exception as e:
    print(f"  render_kernel FAILED: {e}")

print("\nIf all OK above, install bh_render_engine.py as a Blender addon.")
print("Edit > Preferences > Add-ons > Install > select bh_render_engine.py")