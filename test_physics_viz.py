import numpy as np
import matplotlib.pyplot as plt
import time
from core.geodesics import integrate_path
from core.constants import RS, R_PHOTON, SIM_BOUNDS

def test_rays():
    print(f"⚫ Simulating Schwarzschild Black Hole (Rs = {RS})")
    print(f"   Termination Boundary: {SIM_BOUNDS}")
    print("   Firing test photons...")

    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # 1. Draw the Black Hole
    bh = plt.Circle((0, 0), RS, color='black', zorder=10)
    ps = plt.Circle((0, 0), R_PHOTON, color='orange', fill=False, linestyle='--', alpha=0.5)
    
    ax.add_patch(bh)
    ax.add_patch(ps)

    # 2. Fire a spread of rays
    impact_parameters = np.linspace(0.1, 10.0, 50) # Increased to 50 rays

    start_time = time.time()
    
    for b in impact_parameters:
        # Numba requires explicit numpy arrays as input
        start_pos = np.array([-SIM_BOUNDS + 5.0, b,0.0]) 
        start_vel = np.array([1.0, 0.0, 0.0])
        
        # RUN THE ENGINE
        path, captured, _, _ = integrate_path(start_pos, start_vel, dt=0.2)
        
        # Visual Logic
        if captured:
            color = '#ff3333' # Red
            alpha = 0.4
        else:
            final_pos = path[-1]
            angle = np.arctan2(final_pos[1], final_pos[0])
            if abs(angle) > 0.1: 
                color = '#33ccff' # Cyan (Lensed)
            else:
                color = '#ffffff' # White (Straight)
            alpha = 0.6

        plt.plot(path[:,0], path[:,1], color=color, alpha=alpha, linewidth=1)

    elapsed = time.time() - start_time
    print(f"⚡ Calculation complete in {elapsed:.4f} seconds")
    
    plt.xlim(-20, 20)
    plt.ylim(-15, 15)
    plt.title("Schwarzschild Geodesics (Numba Optimized)", color='white')
    plt.grid(True, alpha=0.1)
    
    print("✅ Plot generated. Showing window...")
    plt.show()

if __name__ == "__main__":
    test_rays()