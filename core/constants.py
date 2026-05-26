"""
Module: core.constants
Physical constants for the Schwarzschild Black Hole.
We use Geometrized Units (G = c = 1) for numerical stability.
"""

# Fundamental Constants
G = 1.0
C = 1.0

# Black Hole Properties
MASS = 1.0

# Schwarzschild Radius (The Event Horizon)
# Rs = 2GM / c^2
RS = 2.0 * G * MASS / (C**2)

# The Photon Sphere
R_PHOTON = 1.5 * RS

# The Accretion Disk Limits
DISK_INNER = 3.0 * RS    # == R_ISCO for Schwarzschild
DISK_OUTER = 18.0 * RS

# Simulation boundary.
# 50*RS == 100 was too small: lensed rays on long arcs were terminated early
# before escaping, producing missing pixels near the shadow edge.
# 200*RS costs nothing (escaped rays stop immediately once dist > bound).
SIM_BOUNDS = 200.0 * RS