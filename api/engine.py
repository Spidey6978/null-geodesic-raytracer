"""
Module: api.engine
Pure Python library interface for invoking the Kerr Black Hole Raytracer
programmatically without CLI global side-effects.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from core.config import RenderConfig, RenderMode
from core.camera import generate_camera_rays
from core.constants import RS, C, SIM_BOUNDS
from scripts.cam_presets import CAMERA_PRESETS
from scripts.render_kernel import (
    render_pixel_batch,
    aces_tonemap,
    _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
    _PT, _PR, _PG, _PB, _HIT_W
)


def resolve_config_defaults(config: RenderConfig) -> RenderConfig:
    """Fills in default resolution, dt, and max_steps based on preset mode if not set."""
    mode_presets = {
        RenderMode.PREVIEW:    dict(width=600,  height=400,  dt=0.1, max_steps=1500),
        RenderMode.QUALITY:    dict(width=960,  height=540,  dt=0.1, max_steps=5000),
        RenderMode.PRODUCTION: dict(width=1920, height=1080, dt=0.2, max_steps=8000),
    }
    m = mode_presets.get(config.mode, mode_presets[RenderMode.QUALITY])

    if config.width is None: config.width = m["width"]
    if config.height is None: config.height = m["height"]
    if config.dt is None: config.dt = m["dt"]
    if config.max_steps is None: config.max_steps = m["max_steps"]

    if config.camera.preset and config.camera.preset in CAMERA_PRESETS:
        p = CAMERA_PRESETS[config.camera.preset]
        config.camera.cam_pos = p["cam_pos"]
        config.camera.look_at = p["look_at"]
        config.camera.fov = p["fov"]
        if "roll" in p:
            config.camera.roll = p["roll"]

    return config


def calculate_kerr_isco(spin: float, mass: float = 1.0) -> float:
    """Calculates exact Kerr ISCO radius for given spin and mass."""
    a = min(abs(spin), 0.998)
    z1 = 1.0 + (1.0 - a*a)**(1/3) * ((1.0 + a)**(1/3) + (1.0 - a)**(1/3))
    z2 = np.sqrt(3.0 * a*a + z1*z1)
    r_isco = 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0*z2))
    return r_isco * mass


def render_frame_from_config(config: RenderConfig, out_filepath: str) -> dict:
    """
    Renders a single frame programmatically based on RenderConfig and writes output PNG/JSON.
    Returns metadata dictionary.
    """
    config = resolve_config_defaults(config)

    mass = config.black_hole.mass
    spin = config.black_hole.spin
    r_outer_horizon = mass + np.sqrt(max(mass * mass - spin * spin, 0.0))
    r_isco = calculate_kerr_isco(spin, mass)
    disk_inner = config.black_hole.disk_inner if config.black_hole.disk_inner is not None else r_isco
    disk_outer = config.black_hole.disk_outer
    rs = 2.0 * mass

    width = config.width
    height = config.height
    dt = config.dt
    max_steps = config.max_steps
    cam_pos = np.array(config.camera.cam_pos, dtype=np.float64)
    look_at = config.camera.look_at
    fov = config.camera.fov
    roll = config.camera.roll
    frame_time = config.frame_time

    ray_dirs = generate_camera_rays(width, height, fov, list(cam_pos), look_at, roll_degrees=roll)
    image = np.zeros((height, width, 3), dtype=np.float64)
    ray_debug = np.zeros((height, width), dtype=np.uint8)

    t0 = time.time()
    render_pixel_batch(
        ray_dirs, cam_pos,
        _STAR_DIRS, _STAR_BRIGHT, _STAR_COS_RADII, _STAR_COLOUR, _STAR_PAL,
        _PT, _PR, _PG, _PB,
        image, ray_debug, width, height,
        mass, spin, r_outer_horizon,
        disk_inner, disk_outer, SIM_BOUNDS,
        rs, r_isco, _HIT_W, dt, max_steps,
        frame_time
    )
    elapsed = time.time() - t0

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = aces_tonemap(image)

    out_path = Path(out_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(width / 80, height / 80), facecolor="black")
    ax.imshow(image, origin="upper")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150, facecolor="black")
    plt.close(fig)

    meta = {
        "out_file": str(out_path),
        "render_time_s": elapsed,
        "width": width,
        "height": height,
        "spin": spin,
        "mass": mass,
        "dt": dt,
        "max_steps": max_steps,
        "frame_time": frame_time,
        "cam_pos": list(cam_pos),
        "look_at": look_at,
    }

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def render_animation_sequence(anim_config: "AnimationConfig", out_video_path: str, progress_callback=None) -> dict:
    """
    Renders a multi-frame camera flyby animation sequence along a 3D spline trajectory.
    Compiles individual PNG frames into an MP4 video or image sequence.
    """
    from core.camera import generate_spline_camera_path
    from core.config import AnimationConfig, CameraConfig

    out_video_file = Path(out_video_path)
    frames_dir = out_video_file.parent / f"{out_video_file.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_path = generate_spline_camera_path(
        waypoints=anim_config.waypoints,
        num_frames=anim_config.num_frames,
        look_at=anim_config.look_at,
        fov=anim_config.fov,
        roll=anim_config.roll
    )

    frame_files = []
    total_time = 0.0

    for i, frame_spec in enumerate(camera_path):
        frame_time = i / float(anim_config.fps)
        frame_cfg = RenderConfig(
            black_hole=anim_config.black_hole,
            camera=CameraConfig(
                cam_pos=frame_spec["cam_pos"],
                look_at=frame_spec["look_at"],
                fov=frame_spec["fov"],
                roll=frame_spec["roll"]
            ),
            dt=anim_config.dt,
            max_steps=anim_config.max_steps,
            mode=anim_config.mode,
            frame_time=frame_time
        )

        frame_file = str(frames_dir / f"frame_{i:04d}.png")
        meta = render_frame_from_config(frame_cfg, frame_file)
        frame_files.append(frame_file)
        total_time += meta["render_time_s"]

        if progress_callback:
            pct = ((i + 1) / anim_config.num_frames) * 100.0
            progress_callback(i + 1, anim_config.num_frames, pct)

    # Try compiling MP4 video using imageio or matplotlib
    video_compiled = False
    try:
        import imageio
        writer = imageio.get_writer(str(out_video_file), fps=anim_config.fps)
        for ff in frame_files:
            writer.append_data(imageio.v2.imread(ff))
        writer.close()
        video_compiled = True
    except Exception:
        video_compiled = False

    meta = {
        "video_file": str(out_video_file) if video_compiled else None,
        "frames_dir": str(frames_dir),
        "num_frames": anim_config.num_frames,
        "fps": anim_config.fps,
        "total_render_time_s": total_time,
        "video_compiled": video_compiled,
        "frames": frame_files
    }

    json_path = out_video_file.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
