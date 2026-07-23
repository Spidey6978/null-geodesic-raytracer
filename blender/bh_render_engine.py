"""
blender/bh_render_engine.py
Custom Blender render engine that uses the null geodesic raytracer.
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
import traceback

REPO_ROOT = os.path.expanduser("C:/dev stuff/projects/python/black-hole-sim")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Per-session timing accumulator ───────────────────────────────────────────
_session_stats = {
    "frames_completed": 0,
    "frames_failed":    0,
    "total_time":       0.0,
    "frame_times":      {},   # frame_number -> elapsed_seconds
}


def _import_raytracer():
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
    stars  = (_STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL)
    planck = (_PT, _PR, _PG, _PB)
    return render_pixel_batch, aces_tonemap, phys, stars, planck, _HIT_W


def _fmt_time(seconds):
    """Human-readable duration: 1h 23m 45s or 2m 03s or 47.3s"""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60:02d}m {s%60:02d}s"
    elif s >= 60:
        return f"{s//60}m {s%60:02d}s"
    else:
        return f"{seconds:.1f}s"


def _eta(elapsed_so_far, frames_done, frames_remaining):
    if frames_done == 0:
        return "unknown"
    avg = elapsed_so_far / frames_done
    return _fmt_time(avg * frames_remaining)


class BlackHoleRenderEngine(bpy.types.RenderEngine):
    bl_idname      = "BLACK_HOLE"
    bl_label       = "Black Hole"
    bl_use_preview = False

    def render(self, depsgraph):
        scene  = depsgraph.scene
        frame  = scene.frame_current
        frame_start = scene.frame_start
        frame_end   = scene.frame_end
        total_frames = frame_end - frame_start + 1
        frame_idx    = frame - frame_start + 1   # 1-based index in this batch

        bpy.context.scene.frame_set(frame)

        render = scene.render
        width  = int(render.resolution_x * render.resolution_percentage / 100)
        height = int(render.resolution_y * render.resolution_percentage / 100)

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  FRAME {frame}  [{frame_idx}/{total_frames}]  {width}×{height}")
        print(sep)

        # ── Import ────────────────────────────────────────────────────────────
        try:
            render_pixel_batch, aces_tonemap, phys, stars, planck, hit_w = \
                _import_raytracer()
        except Exception as e:
            msg = f"Import failed: {e}"
            self.report({'ERROR'}, msg)
            print(f"  ✗ {msg}")
            traceback.print_exc()
            _session_stats["frames_failed"] += 1
            return

        # ── Camera ────────────────────────────────────────────────────────────
        try:
            from blender.bh_camera import blender_camera_to_ray_dirs
            cam_pos, ray_dirs, fov_deg = blender_camera_to_ray_dirs(scene, width, height)
            print(f"  Camera  pos={[f'{v:.3f}' for v in cam_pos]}  FOV={fov_deg:.1f}°")
        except Exception as e:
            msg = f"Camera read failed: {e}"
            self.report({'ERROR'}, msg)
            print(f"  ✗ {msg}")
            traceback.print_exc()
            _session_stats["frames_failed"] += 1
            return

        # ── Physics params ────────────────────────────────────────────────────
        dt     = scene.get("bh_dt",        0.2)
        msteps = scene.get("bh_max_steps", 5000)
        print(f"  Physics  a={phys['spin']:.4f}  M={phys['mass']:.4f}  "
              f"dt={dt}  steps={msteps}")

        # ── Render ────────────────────────────────────────────────────────────
        image = np.zeros((height, width, 3), dtype=np.float64)
        t0 = time.time()

        try:
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
        except Exception as e:
            elapsed = time.time() - t0
            msg = f"render_pixel_batch failed after {_fmt_time(elapsed)}: {e}"
            self.report({'ERROR'}, msg)
            print(f"  ✗ {msg}")
            traceback.print_exc()
            _session_stats["frames_failed"] += 1
            return

        elapsed = time.time() - t0
        _session_stats["frames_completed"] += 1
        _session_stats["total_time"]       += elapsed
        _session_stats["frame_times"][frame] = elapsed

        # ── ETA ───────────────────────────────────────────────────────────────
        completed = _session_stats["frames_completed"]
        remaining = total_frames - frame_idx
        eta_str   = _eta(_session_stats["total_time"], completed, remaining)
        avg_str   = _fmt_time(_session_stats["total_time"] / completed)

        print(f"  ✓ Frame {frame} complete  {_fmt_time(elapsed)}")
        print(f"    avg/frame={avg_str}  remaining={remaining}  ETA={eta_str}")

        # Slowest/fastest so far
        times = list(_session_stats["frame_times"].values())
        if len(times) > 1:
            slowest = max(times)
            fastest = min(times)
            print(f"    session: {completed} done  "
                  f"fastest={_fmt_time(fastest)}  slowest={_fmt_time(slowest)}")

        print(sep)

        # ── Tonemap ───────────────────────────────────────────────────────────
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = aces_tonemap(image)

        # ── Write to Blender ──────────────────────────────────────────────────
        result = self.begin_result(0, 0, width, height)
        layer  = result.layers[0].passes["Combined"]

        rgba             = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3]   = image.astype(np.float32)
        rgba_flipped     = np.ascontiguousarray(rgba[::-1, :, :])
        layer.rect       = rgba_flipped.reshape(-1, 4)

        self.end_result(result)
        self.report({'INFO'}, f"Frame {frame} ✓  {_fmt_time(elapsed)}")


def register():
    bpy.utils.register_class(BlackHoleRenderEngine)


def unregister():
    bpy.utils.unregister_class(BlackHoleRenderEngine)


if __name__ == "__main__":
    register()