import bpy
import os

# ── Output ─────────────────────────────────────────────────────────────

OUTPUT_DIR    = r"C:/dev stuff/projects/python/black-hole-sim/blender_anim"
OUTPUT_PREFIX  = "frame_"

bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode  = 'RGB'

os.makedirs(OUTPUT_DIR, exist_ok=True)

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

FRAME_START = 1
FRAME_END = 180

bpy.context.scene.frame_end = FRAME_END

bpy.context.scene["bh_dt"] = 0.2
bpy.context.scene["bh_max_steps"] = 5000

print("BH render settings applied. Hit Ctrl+F12 to render.")
# Find the last completed frame and resume from next one
def frame_number(filename):
    stem = os.path.splitext(filename)[0]
    return int(stem.removeprefix(OUTPUT_PREFIX))

existing = sorted(
    (f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")),
    key=frame_number
)

last_frame = None

if existing:
    last_file = existing[-1]

    stem = os.path.splitext(last_file)[0]

    try:
        last_frame = int(stem.removeprefix(OUTPUT_PREFIX))
    except ValueError:
        raise RuntimeError(
            f"Couldn't parse frame number from '{last_file}'"
        )
if last_frame is None:
    bpy.context.scene.frame_start = FRAME_START
    print("▶ Starting from frame 1")

elif last_frame >= FRAME_END:
    bpy.context.scene.frame_start = FRAME_END
    print("✓ Animation already complete.")

else:
    bpy.context.scene.frame_start = last_frame + 1
    print(f"↩ Resuming from frame {last_frame + 1}")
print()
print("=" * 55)
print("Black Hole Render Configuration")
print("=" * 55)
print(f"Resolution : {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")
print(f"Frames     : {bpy.context.scene.frame_start} -> {FRAME_END}")
print(f"dt         : {bpy.context.scene['bh_dt']}")
print(f"Max steps  : {bpy.context.scene['bh_max_steps']}")
print(f"Output     : {OUTPUT_DIR}")
print("=" * 55)
print("Press Ctrl+F12 to begin rendering.")