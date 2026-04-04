"""
Module: core.constants
Physical constants for the Schwarzschild Black Hole.
We use Geometrized Units (G = c = 1) for numerical stability.
"""

# Fundamental Constants
G = 1.0
C = 1.0

# Black Hole Properties
# MASS = 1.0 represents a standard unit mass. 
MASS = 1.0 

# Schwarzschild Radius (The Event Horizon)
# Rs = 2GM / c^2
RS = 2.0 * G * MASS / (C**2)

# The Photon Sphere (The Ring of Fire)
# R_photon = 1.5 * Rs
R_PHOTON = 1.5 * RS

# The Accretion Disk Limits (Visuals)
DISK_INNER = 3.0 * RS
DISK_OUTER = 12.0 * RS

# SIMULATION BOUNDS (Dynamic)
# How far a ray must travel before we consider it "Escaped"
# 50x the radius ensures we are effectively at "infinity"
SIM_BOUNDS = 50.0 * RS