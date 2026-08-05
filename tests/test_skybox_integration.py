"""
Module: tests.test_skybox_integration
Automated integration tests for equirectangular skybox background rendering.
"""

from pathlib import Path
from core.config import RenderConfig, RenderMode, BlackHoleConfig
from api.engine import render_frame_from_config


def test_render_frame_with_skybox_fallback(tmp_path):
    out_file = tmp_path / "skybox_test.png"
    config = RenderConfig(
        black_hole=BlackHoleConfig(mass=1.0, spin=0.998),
        mode=RenderMode.PREVIEW,
        skybox_path="procedural"
    )

    meta = render_frame_from_config(config, str(out_file))
    assert Path(out_file).exists()
    assert meta["render_time_s"] > 0.0
