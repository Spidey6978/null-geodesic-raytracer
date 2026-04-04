import numpy as np
import matplotlib.pyplot as plt
import time
from core.camera import generate_camera_rays
from core.geodesics import integrate_path

def render():
    # --- RENDER SETTINGS ---
    # We start with a low resolution (400x400) so it renders in ~30 seconds.
    # We can bump this up later.
    WIDTH = 400
    HEIGHT = 400
    FOV = 60
    
    # Camera is 15 units away, looking directly at the center (0,0,0)
    CAM_POS = [0.0, 5.0, 15.0] 
    LOOK_AT = [0.0, 0.0, 0.0]

    print(f"📷 Initializing Camera ({WIDTH}x{HEIGHT})...")
    ray_directions = generate_camera_rays(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    
    # Create a blank image (Black)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    
    print("🚀 Firing Photons into the Kerr/Schwarzschild Metric...")
    start_time = time.time()
    
    # We loop through every pixel.
    # Note: In the future, we will push this loop into Numba or Celery to make it instantly fast.
    # For this first test, pure Python looping is fine for a small image.
    total_pixels = WIDTH * HEIGHT
    
    for y in range(HEIGHT):
        if y % 10 == 0: # Print progress every 10 rows
            print(f"   Rendering row {y}/{HEIGHT} ({(y/HEIGHT)*100:.1f}%)")
            
        for x in range(WIDTH):
            # 1. Get the starting parameters for this specific pixel
            start_pos = np.array(CAM_POS)
            start_vel = ray_directions[y, x] # Speed of light C is handled inside the integrator
            
            # 2. Trace the ray
            # We don't need a tiny dt for image rendering, 0.5 is fine for the shadow outline
            path, captured = integrate_path(start_pos, start_vel, dt=0.5, max_steps=1000)
            
            # 3. Paint the pixel
            if captured:
                # The Event Horizon (Shadow)
                image[y, x] = [0.0, 0.0, 0.0] 
            else:
                # The ray escaped! Let's color it based on its FINAL direction
                # to visualize how space was bent.
                final_vel = path[-1] - path[-2] # Final direction vector
                final_vel = final_vel / np.linalg.norm(final_vel)
                
                # Map vector [-1, 1] to RGB color [0, 1] for a cool background effect
                color = (final_vel + 1.0) / 2.0
                image[y, x] = color

    elapsed = time.time() - start_time
    print(f"✅ Render complete in {elapsed:.2f} seconds.")
    
    # --- DISPLAY THE IMAGE ---
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"First Light - Schwarzschild Shadow\n{WIDTH}x{HEIGHT} px | {elapsed:.1f}s render time", color='white')
    
    # Save a high-quality PNG to your disk!
    plt.savefig("first_light.png", bbox_inches='tight', dpi=300, facecolor='black')
    print("💾 Image saved as 'first_light.png'")
    
    plt.show()

if __name__ == "__main__":
    render()