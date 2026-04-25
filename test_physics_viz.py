import numpy as np
import matplotlib.pyplot as plt
import time
from core.geodesics import integrate_path
from core.constants import RS, R_PHOTON, SIM_BOUNDS

def test_rays():
    print(f"⚫ Simulating Schwarzschild Black Hole (Rs = {RS})")
    print("   Firing test photons...")

    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    bh = plt.Circle((0, 0), RS, color='black', zorder=10)
    ps = plt.Circle((0, 0), R_PHOTON, color='orange', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(bh)
    ax.add_patch(ps)

    impact_parameters = np.linspace(0.1, 10.0, 50) 
    start_time = time.time()
    
    for b in impact_parameters:
        start_pos = np.array([-SIM_BOUNDS + 5.0, b, 0.0]) 
        start_vel = np.array([1.0, 0.0, 0.0])
        
        # Unpack 6 variables now!
        path, captured, _, _, _, _ = integrate_path(start_pos, start_vel, dt=0.2)
        
        if captured:
            color = '#ff3333'
            alpha = 0.4
        else:
            final_pos = path[-1]
            angle = np.arctan2(final_pos[1], final_pos[0])
            if abs(angle) > 0.1: 
                color = '#33ccff' 
            else:
                color = '#ffffff' 
            alpha = 0.6

        plt.plot(path[:,0], path[:,1], color=color, alpha=alpha, linewidth=1)

    elapsed = time.time() - start_time
    print(f"⚡ Calculation complete in {elapsed:.4f} seconds")
    
    plt.xlim(-20, 20)
    plt.ylim(-15, 15)
    plt.title("Schwarzschild Geodesics (Numba Optimized)", color='white')
    plt.grid(True, alpha=0.1)
    plt.show()

if __name__ == "__main__":
    test_rays()