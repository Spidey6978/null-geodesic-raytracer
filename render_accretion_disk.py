import numpy as np
import matplotlib.pyplot as plt
import time
from core.camera import generate_camera_rays
from core.geodesics import integrate_path
from core.constants import DISK_INNER, DISK_OUTER

def render():
    WIDTH = 400
    HEIGHT = 400
    FOV = 60
    
    # We move the camera up slightly (Y=1.0) to look slightly down on the disk
    CAM_POS = [0.0, 1.0, 15.0] 
    LOOK_AT = [0.0, 0.0, 0.0]

    print(f"📷 Initializing Camera ({WIDTH}x{HEIGHT})...")
    ray_directions = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    
    # Create a blank image (Deep Space Black)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    
    print("🚀 Firing Photons... Look for the Halo!")
    start_time = time.time()
    
    for y in range(HEIGHT):
        if y % 20 == 0:
            print(f"   Rendering row {y}/{HEIGHT} ({(y/HEIGHT)*100:.1f}%)")
            
        for x in range(WIDTH):
            start_pos = np.array(CAM_POS)
            start_vel = ray_directions[y, x]
            
            # Run the engine (Notice we unpack 4 variables now)
            path, captured, hit_disk, hit_radius = integrate_path(start_pos, start_vel, dt=0.5, max_steps=1000)
            
            # --- SHADER LOGIC (Painting the Pixel) ---
            if captured:
                # The Event Horizon Shadow
                image[y, x] = [0.0, 0.0, 0.0] 
                
            elif hit_disk:
                # 1. Normalize the hit radius (0.0 = Inner Edge, 1.0 = Outer Edge)
                r_norm = (hit_radius - DISK_INNER) / (DISK_OUTER - DISK_INNER)
                
                # 2. Temperature Profile: Gas gets vastly hotter/brighter closer to the center
                intensity = (1.0 - r_norm) ** 1.5 
                
                # 3. Apply a Fiery Color Palette (RGB)
                r = np.clip(intensity * 1.5 + 0.1, 0, 1) # High Red
                g = np.clip(intensity * 0.8 + 0.05, 0, 1) # Mid Green (makes orange/yellow)
                b = np.clip(intensity * 0.3, 0, 1) # Low Blue (White hot at the very core)
                
                image[y, x] = [r, g, b]
            else:
                # Background Space
                image[y, x] = [0.0, 0.0, 0.02] # Pitch black makes the disk pop

    elapsed = time.time() - start_time
    print(f"✅ Render complete in {elapsed:.2f} seconds.")
    
    # --- DISPLAY ---
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Schwarzschild Accretion Disk\n{WIDTH}x{HEIGHT} px | {elapsed:.1f}s render time", color='white')
    plt.savefig("accretion_disk.png", bbox_inches='tight', dpi=300, facecolor='black')
    plt.show()

if __name__ == "__main__":
    render()