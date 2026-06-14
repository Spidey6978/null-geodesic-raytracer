# test_imports.py — run this first, fix any errors before attempting a render
import sys, os

from core.indices import NUM_DOCTOR_METRICS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")
from core.constants import SPIN, MASS, DISK_INNER, RS
print(f"  constants OK — SPIN={SPIN:.4f} DISK_INNER={DISK_INNER:.4f}")

from core.geodesics import integrate_path_doctor
print("  geodesics OK")

from doctor.diagnostics import DoctorData, collect_diagnostics
print("  diagnostics OK")

from core.indices import IDX_ORBIT_COUNT, NUM_DOCTOR_METRICS
print(f"  indices OK — NUM_DOCTOR_METRICS={NUM_DOCTOR_METRICS}")

from doctor.utils.render_base import compute_tensor
print("  render_base OK")

import importlib
mod = importlib.import_module("doctor.modes.orbit_count")
print(f"  orbit_count mode OK — label='{mod.CONFIG['label']}'")

mod = importlib.import_module("doctor.modes.termination_map")
print(f"  termination_map mode OK")

print("\nAll imports clean. Ready to run dispatcher.")