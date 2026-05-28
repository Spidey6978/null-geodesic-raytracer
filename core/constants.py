"""
Module: core.constants
Physical constants for the Kerr/Schwarzschild Black Hole.
We use Geometrized Units (G = c = 1) for numerical stability.
"""

# ── Fundamental Constants ─────────────────────────────────────────────────────
G = 1.0
C = 1.0

# ── Black Hole Properties ─────────────────────────────────────────────────────
MASS = 1.0

# Spin parameter 'a' (0.0 = Schwarzschild, 0.999 = Extreme Kerr)
# SET TO 0.0 FOR STEP 1 TESTING!
SPIN = 0.998 * MASS    

# Schwarzschild Radius (Used for legacy fallbacks and scaling)
RS = 2.0 * G * MASS / (C**2)
R_PHOTON = 1.5 * RS

# ── Kerr Horizons ─────────────────────────────────────────────────────────────
# In Kerr, the event horizon splits into an outer and inner (Cauchy) horizon.
_sqrt_term = (MASS**2 - SPIN**2)**0.5
R_OUTER_HORIZON = MASS + _sqrt_term     # The true event horizon
R_INNER_HORIZON = MASS - _sqrt_term     # The Cauchy horizon

# ── Kerr ISCO (Prograde) ──────────────────────────────────────────────────────
def _compute_isco(M, a):
    """
    Calculates the Innermost Stable Circular Orbit for a spinning black hole.
    Formula derived from Bardeen, Press, and Teukolsky (1972).
    """
    if abs(a) < 1e-10:
        return 6.0 * M    # Exact Schwarzschild limit
    
    # Complex polynomial roots for Kerr ISCO
    Z1 = 1.0 + (1.0 - (a/M)**2)**(1/3) * ((1.0 + a/M)**(1/3) + (1.0 - a/M)**(1/3))
    Z2 = (3.0 * (a/M)**2 + Z1**2)**0.5
    
    # Prograde orbit subtraction
    return M * (3.0 + Z2 - ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**0.5)

# ── Accretion Disk Limits ─────────────────────────────────────────────────────
# Inner edge maps perfectly to the ISCO.
DISK_INNER = _compute_isco(MASS, SPIN)
DISK_OUTER = 18.0 * RS

# ── Simulation boundary ───────────────────────────────────────────────────────
SIM_BOUNDS = 200.0 * RS