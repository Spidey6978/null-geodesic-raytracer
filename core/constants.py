"""
core/constants.py
Physical constants for the Kerr/Schwarzschild Black Hole.
Geometrized Units (G = c = 1).

Override at runtime via environment variables:
    BH_SPIN=0.5 python doctor/render.py --mode orbit_count
"""
import os
import numpy as np

G = 1.0
C = 1.0

MASS = float(os.environ.get("BH_MASS", "1.0"))
_raw_spin = float(os.environ.get("BH_SPIN", "0.998"))
if MASS <= 0.0:
    raise ValueError("BH_MASS must be positive")
if abs(_raw_spin) > 1.0:
    raise ValueError("BH_SPIN must be in the dimensionless range [-1, 1]")
SPIN = _raw_spin * MASS

RS = 2.0 * G * MASS / (C**2)
R_PHOTON = 1.5 * RS

_sqrt_term      = (MASS**2 - SPIN**2) ** 0.5
R_OUTER_HORIZON = MASS + _sqrt_term
R_INNER_HORIZON = MASS - _sqrt_term

def _compute_isco(M, a):
    if abs(a) < 1e-10:
        return 6.0 * M
    Z1 = 1.0 + (1.0 - (a/M)**2)**(1/3) * (
         (1.0 + a/M)**(1/3) + (1.0 - a/M)**(1/3))
    Z2 = (3.0*(a/M)**2 + Z1**2)**0.5
    return M * (3.0 + Z2 - ((3.0-Z1)*(3.0+Z1+2.0*Z2))**0.5)

_disk_inner_override = os.environ.get("BH_DISK_INNER")
_disk_outer_override = os.environ.get("BH_DISK_OUTER")

DISK_INNER = (float(_disk_inner_override) * RS
              if _disk_inner_override
              else _compute_isco(MASS, SPIN))

DISK_OUTER = (float(_disk_outer_override) * RS
              if _disk_outer_override
              else 18.0 * RS)

SIM_BOUNDS = 200.0 * RS
