import bpy
import os

# ── Output ─────────────────────────────────────────────────────────────

OUTPUT_DIR    = r"C:/dev stuff/projects/python/black-hole-sim/blender_anim"
OUTPUT_PREFIX  = "frame_"

bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode  = 'RGB'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resolution
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

# Frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 180

# Physics
bpy.context.scene["bh_dt"]        = 0.2
bpy.context.scene["bh_max_steps"] = 5000

print("BH render settings applied. Hit Ctrl+F12 to render.")
# Find the last completed frame and resume from next one
existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
if existing:
    last_frame = int(existing[-1].replace('OUTPUT_PREFIX', '').replace('.png', ''))
    bpy.context.scene.frame_start = last_frame + 1
    print(f"Resuming from frame {last_frame + 1}")
else:
    bpy.context.scene.frame_start = 1
    print("Starting from frame 1")