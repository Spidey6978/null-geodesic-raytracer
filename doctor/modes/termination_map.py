# doctor/modes/termination_map.py
import numpy as np
import time

from doctor.utils.render_base import compute_tensor, save_diagnostic_image
from doctor.utils.colormaps import apply_discrete_colormap, TERMINATION_COLORS
from doctor.utils.indices import IDX_TERM_REASON

def run():
    WIDTH = 960
    HEIGHT = 540
    CAM_POS = np.array([12.5, 1.5, 0.0], dtype=np.float64)
    LOOK_AT = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    FOV = 50.0
    
    print("🔬 Running Termination Reason Diagnostic...")
    start = time.time()
    
    # 1. Build the massive data tensor
    tensor = compute_tensor(WIDTH, HEIGHT, FOV, CAM_POS, LOOK_AT)
    
    # 2. Slice out just the layer we want to look at
    term_layer = tensor[:, :, IDX_TERM_REASON]
    
    # 3. Apply the heatmap colors
    img = apply_discrete_colormap(term_layer, TERMINATION_COLORS)
    
    # 4. Save
    save_diagnostic_image(img, "termination_map.png")
    
    print(f"✅ Diagnostic finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    run()