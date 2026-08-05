"""
Pytest configuration fixture: Force Matplotlib non-interactive Agg backend to prevent Tkinter GUI thread issues on Windows.
"""

import matplotlib
matplotlib.use("Agg")
