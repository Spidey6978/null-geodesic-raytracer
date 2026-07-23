# doctor/utils/colormaps.py
import numpy as np

# Map termination reason to standard diagnostic colors
# 0 = Unknown/Ongoing
# 1 = Captured (Event Horizon) -> RED
# 2 = NaN/Math Explosion -> MAGENTA (Immediate flag)
# 3 = SIM_BOUNDS (Escaped to stars) -> GREEN
# 4 = Orphan (Photon Sphere Trapped/MAX_STEPS_REACHED) -> YELLOW
TERMINATION_COLORS = {
    0: (0.0, 0.0, 0.0),       
    1: (1.0, 0.0, 0.0),       
    2: (1.0, 0.0, 1.0),       
    3: (0.0, 1.0, 0.0),       
    4: (1.0, 1.0, 0.0)        
}

def apply_discrete_colormap(tensor_layer, color_dict):
    """Converts a 2D array of integers into an RGB image."""
    h, w = tensor_layer.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    
    for val, color in color_dict.items():
        mask = (np.round(tensor_layer).astype(np.int32) == val)
        img[mask] = color
        
    return img

def apply_gradient_colormap(tensor_layer, vmin, vmax, colormap='hot'):
    """Normalizes a float layer to [0,1] and applies a continuous color gradient (placeholder)."""
    normalized = np.clip((tensor_layer - vmin) / (vmax - vmin), 0.0, 1.0)
    # Note: You can plug cv2.applyColorMap here later if you want heatmap visuals!
    img = np.zeros((*tensor_layer.shape, 3), dtype=np.float64)
    img[:, :, 0] = normalized # Grayscale output for now
    img[:, :, 1] = normalized
    img[:, :, 2] = normalized
    return img