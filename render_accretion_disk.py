import numpy as np
import matplotlib.pyplot as plt
import time
from core.camera import generate_camera_rays
from core.geodesics import integrate_path
from core.constants import DISK_INNER, DISK_OUTER, RS

def render():
    WIDTH = 600 # Bumping resolution slightly for a better visual
    HEIGHT = 400
    FOV = 60
    
    CAM_POS = [0.0, 1.5, 15.0] 
    LOOK_AT = [0.0, 0.0, 0.0]

    print(f"📷 Initializing Camera ({WIDTH}x{HEIGHT})...")
    ray_directions = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    
    print("🚀 Firing Photons... Calculating Doppler Beaming!")
    start_time = time.time()
    
    for y in range(HEIGHT):
        if y % 20 == 0:
            print(f"   Rendering row {y}/{HEIGHT} ({(y/HEIGHT)*100:.1f}%)")
            
        for x in range(WIDTH):
            start_pos = np.array(CAM_POS)
            start_vel = ray_directions[y, x]
            
            # Unpack the 6 returned values
            path, captured, hit_disk, hit_radius, hit_pos, hit_vel = integrate_path(start_pos, start_vel, dt=0.5, max_steps=1000)
            
            if captured:
                image[y, x] = [0.0, 0.0, 0.0] 
                
            elif hit_disk:
                # 1. Base Thermodynamic Intensity
                r_norm = (hit_radius - DISK_INNER) / (DISK_OUTER - DISK_INNER)
                base_intensity = (1.0 - r_norm) ** 1.5 
                
                # --- SPECIAL RELATIVITY ---
                # 2. Orbital Velocity (Fraction of Speed of Light)
                v_mag = np.sqrt(RS / (2.0 * hit_radius))
                
                # 3. Gas Tangent Vector (Counter-Clockwise)
                tangent = np.array([-hit_pos[2], 0.0, hit_pos[0]])
                tangent = tangent / np.linalg.norm(tangent)
                v_gas = v_mag * tangent
                
                # 4. Ray Direction
                ray_dir = hit_vel / np.linalg.norm(hit_vel)
                
                # 5. Lorentz Factor & Doppler Shift (g-factor)
                gamma = 1.0 / np.sqrt(1.0 - v_mag**2)
                g_factor = (1.0 / gamma) / (1.0 + np.dot(v_gas, ray_dir))
                
                # 6. Apply Relativistic Beaming
                beamed_intensity = base_intensity * (g_factor ** 3)
                
                # --- COLOR MAPPING ---
                r = beamed_intensity * 2.0
                g = beamed_intensity * 0.6
                b = beamed_intensity * 0.1
                
                # Shift towards blue/white if approaching (g_factor > 1)
                if g_factor > 1.0:
                    b += (g_factor - 1.0) * beamed_intensity * 2.0
                    g += (g_factor - 1.0) * beamed_intensity * 1.0
                
                image[y, x] = np.clip([r, g, b], 0, 1)
                
            else:
                image[y, x] = [0.0, 0.0, 0.02]

    elapsed = time.time() - start_time
    print(f"✅ Render complete in {elapsed:.2f} seconds.")
    
    plt.figure(figsize=(12, 8), facecolor='black')
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Relativistic Accretion Disk (Doppler Beamed)\n{WIDTH}x{HEIGHT} px | {elapsed:.1f}s render time", color='white')
    plt.savefig("relativistic_disk.png", bbox_inches='tight', dpi=300, facecolor='black')
    plt.show()

if __name__ == "__main__":
    render()