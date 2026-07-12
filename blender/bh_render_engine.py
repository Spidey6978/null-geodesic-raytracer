"""
blender/bh_render_engine.py
Custom Blender render engine that uses the null geodesic raytracer.

Installation:
    1. Open Blender → Edit → Preferences → Add-ons → Install
    2. Select this file
    3. Enable "Render: Black Hole Raytracer"
    4. In Properties > Render, select "Black Hole" from the engine dropdown

Usage:
    Press F12 to render. The active camera's position and FOV are read
    from the Blender scene automatically.
"""

bl_info = {
    "name":        "Black Hole Raytracer",
    "author":      "SpyD",
    "version":     (1, 0, 0),
    "blender":     (3, 6, 0),
    "location":    "Render > Engine > Black Hole",
    "description": "Physically accurate Kerr black hole renderer via null geodesics",
    "category":    "Render",
}

import bpy
import numpy as np
import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
# Tell Python where your raytracer repo lives so Blender can import it.
# Change this to your actual repo path.
REPO_ROOT = os.path.expanduser("C:/dev stuff/projects/python/black-hole-sim")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _import_raytracer():
    """
    Lazy import of the raytracer. Called at render time, not at registration,
    so Blender doesn't try to import Numba before the addon is actually used.
    Returns (render_pixel_batch, physics_params_dict) or raises ImportError.
    """
    from core.constants import (MASS, SPIN, R_OUTER_HORIZON,
                                 DISK_INNER, DISK_OUTER, SIM_BOUNDS, RS)
    from scripts.render_kernel import (
        render_pixel_batch,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB, _HIT_W,
        aces_tonemap,
    )

    phys = dict(
        mass=MASS, spin=SPIN,
        r_outer_horizon=R_OUTER_HORIZON,
        disk_inner=DISK_INNER, disk_outer=DISK_OUTER,
        sim_bounds=SIM_BOUNDS, rs=RS, r_isco=DISK_INNER,
    )
    stars = (_STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL)
    planck = (_PT, _PR, _PG, _PB)

    return render_pixel_batch, aces_tonemap, phys, stars, planck, _HIT_W


# ── Render Engine ─────────────────────────────────────────────────────────────

class BlackHoleRenderEngine(bpy.types.RenderEngine):
    """
    Blender render engine subclass.
    Blender calls render() for every F12 press or timeline frame.
    """
    bl_idname      = "BLACK_HOLE"
    bl_label       = "Black Hole"
    bl_use_preview = False   # disable viewport preview for now (Phase 4)

    def render(self, depsgraph):
        scene  = depsgraph.scene
        frame  = scene.frame_current

    # Evaluate scene at current frame so keyframes and constraints apply
        bpy.context.scene.frame_set(frame)

        render = scene.render
        width  = int(render.resolution_x * render.resolution_percentage / 100)
        height = int(render.resolution_y * render.resolution_percentage / 100)

        self.report({'INFO'}, f"BH Render: frame {frame}  {width}×{height}")

        try:
            render_pixel_batch, aces_tonemap, phys, stars, planck, hit_w = \
            _import_raytracer()
        except ImportError as e:
            self.report({'ERROR'}, f"Failed to import raytracer: {e}")
        return

        from blender.bh_camera import blender_camera_to_ray_dirs
        cam_pos, ray_dirs, fov_deg = blender_camera_to_ray_dirs(scene, width, height)

        self.report({'INFO'}, f"Frame {frame} | cam={cam_pos.tolist()}")

        image  = np.zeros((height, width, 3), dtype=np.float64)

        # Read dt and max_steps from scene custom properties if set,
        # otherwise use sensible animation defaults
        dt     = scene.get("bh_dt",        0.1)
        msteps = scene.get("bh_max_steps", 2000)

        t0 = time.time()
        render_pixel_batch(
            ray_dirs, cam_pos,
            stars[0], stars[1], stars[2], stars[3], stars[4],
            planck[0], planck[1], planck[2], planck[3],
            image, width, height,
            phys['mass'], phys['spin'], phys['r_outer_horizon'],
            phys['disk_inner'], phys['disk_outer'], phys['sim_bounds'],
            phys['rs'], phys['r_isco'], hit_w,
            dt, msteps
        )
        elapsed = time.time() - t0
        self.report({'INFO'}, f"Frame {frame} done in {elapsed:.1f}s")

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = aces_tonemap(image)

        result = self.begin_result(0, 0, width, height)
        layer  = result.layers[0].passes["Combined"]

        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = image.astype(np.float32)
        rgba_flipped = np.ascontiguousarray(rgba[::-1, :, :])

        # Pass numpy array directly — not .tolist()
        layer.rect = rgba_flipped.reshape(-1, 4)

        self.end_result(result)


# ── Registration ──────────────────────────────────────────────────────────────

def register():
    bpy.utils.register_class(BlackHoleRenderEngine)


def unregister():
    bpy.utils.unregister_class(BlackHoleRenderEngine)


if __name__ == "__main__":
    register()