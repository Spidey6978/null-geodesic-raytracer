"""
Module: tests.test_animation_engine
Automated unit tests for AnimationConfig model and multi-frame animation sequence renderer.
"""

from pathlib import Path
from core.config import AnimationConfig, RenderMode, BlackHoleConfig
from api.engine import render_animation_sequence


def test_animation_config_model():
    anim = AnimationConfig(
        waypoints=[[0.0, 5.0, 15.0], [5.0, 2.0, 10.0]],
        num_frames=5,
        fps=15,
        mode=RenderMode.PREVIEW
    )
    assert anim.num_frames == 5
    assert anim.fps == 15
    assert anim.mode == RenderMode.PREVIEW


def test_render_animation_sequence_preview(tmp_path):
    out_video = tmp_path / "test_anim.mp4"
    anim_config = AnimationConfig(
        black_hole=BlackHoleConfig(mass=1.0, spin=0.998),
        waypoints=[[0.0, 5.0, 15.0], [3.0, 2.0, 10.0]],
        num_frames=2,
        fps=10,
        mode=RenderMode.PREVIEW
    )

    meta = render_animation_sequence(anim_config, str(out_video))
    assert meta["num_frames"] == 2
    assert len(meta["frames"]) == 2
    assert Path(meta["frames"][0]).exists()
    assert Path(meta["frames"][1]).exists()
