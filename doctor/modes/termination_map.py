import os
import sys
import numpy as np
import time

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from doctor.utils.render_base import compute_tensor, save_diagnostic_image
from doctor.utils.colormaps import apply_discrete_colormap, TERMINATION_COLORS
from core.indices import IDX_TERM_REASON

def run(args):
    print("🔬 Running Termination Reason Diagnostic...")
    start = time.time()
    
    # Pass CLI args into the compute engine
    tensor = compute_tensor(
        width=args.width, height=args.height, fov=args.fov,
        cam_pos=args.cam_pos, look_at=args.look_at, roll=args.roll,
        dt=args.dt, max_steps=args.max_steps,
        spin=args.spin if args.spin is not None else 0.0,
        mass=args.mass
    )
    
    term_layer = tensor[:, :, IDX_TERM_REASON]
    
    img = apply_discrete_colormap(term_layer, TERMINATION_COLORS)
    save_diagnostic_image(img, "doctor/outputs/termination_map.png")
    
    print(f"✅ Diagnostic finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    run(None)