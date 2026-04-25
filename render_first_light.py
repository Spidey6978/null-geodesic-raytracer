import numpy as np
import matplotlib.pyplot as plt
import time
from core.camera import generate_camera_rays
from core.geodesics import integrate_path

def render():
    WIDTH = 400
    HEIGHT = 400
    FOV = 60
    CAM_POS = [0.0, 5.0, 15.0] 
    LOOK_AT = [0.0, 0.0, 0.0]

    print(f"📷 Initializing Camera ({WIDTH}x{HEIGHT})...")
    ray_directions = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    
    print("🚀 Firing Photons into the Metric...")
    start_time = time.time()
    
    for y in range(HEIGHT):
        if y % 10 == 0:
            print(f"   Rendering row {y}/{HEIGHT} ({(y/HEIGHT)*100:.1f}%)")
            
        for x in range(WIDTH):
            start_pos = np.array(CAM_POS)
            start_vel = ray_directions[y, x] 
            
            # Unpack 6 variables now!
            path, captured, _, _, _, _ = integrate_path(start_pos, start_vel, dt=0.5, max_steps=1000)
            
            if captured:
                image[y, x] = [0.0, 0.0, 0.0] 
            else:
                final_vel = path[-1] - path[-2] 
                final_vel = final_vel / np.linalg.norm(final_vel)
                color = (final_vel + 1.0) / 2.0
                image[y, x] = color

    elapsed = time.time() - start_time
    print(f"✅ Render complete in {elapsed:.2f} seconds.")
    
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"First Light - Schwarzschild Shadow\n{WIDTH}x{HEIGHT} px | {elapsed:.1f}s render time", color='white')
    plt.savefig("first_light.png", bbox_inches='tight', dpi=300, facecolor='black')
    
if __name__ == "__main__":
    render()