"""
Module: tests.test_skybox
Automated unit tests for equirectangular skybox texture sampling and procedural galaxy generation.
"""

import numpy as np
from core.skybox import (
    sample_skybox_equirectangular,
    generate_procedural_skybox,
    load_skybox_texture,
)


def test_generate_procedural_skybox():
    sky_img = generate_procedural_skybox(width=100, height=50)
    assert sky_img.shape == (50, 100, 3)
    assert sky_img.min() >= 0.0
    assert sky_img.max() <= 1.0


def test_sample_skybox_equirectangular_poles():
    sky_img = np.zeros((100, 200, 3), dtype=np.float64)
    sky_img[0, :, 0] = 1.0  # Top pole is red
    sky_img[99, :, 2] = 1.0 # Bottom pole is blue

    # Ray pointing straight up (+Y) -> top pole
    rgb_up = sample_skybox_equirectangular(0.0, 1.0, 0.0, sky_img, 200, 100)
    assert rgb_up[0] == 1.0

    # Ray pointing straight down (-Y) -> bottom pole
    rgb_down = sample_skybox_equirectangular(0.0, -1.0, 0.0, sky_img, 200, 100)
    assert rgb_down[2] == 1.0


def test_load_skybox_texture_fallback():
    img, w, h = load_skybox_texture("non_existent_file.hdr")
    assert w == 2048
    assert h == 1024
    assert img.shape == (1024, 2048, 3)
