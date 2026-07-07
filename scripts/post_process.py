"""
post_process.py — Modular post-processing pipeline for the BH renderer.

Architecture:
  - Each effect is an isolated function that takes an HDR float32 image
    and returns a float32 image. No side effects, no shared state.
  - Pipelines are lists of (function, kwargs) tuples executed in order.
  - Modes are named pipeline configurations stored in PIPELINES dict.
  - Tonemappers are pluggable via tonemap(img, mode="aces").
  - Simulation mode can skip tonemapping entirely for HDR export.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Any


# ── Dataclasses for pipeline settings ────────────────────────────────────────

@dataclass
class SimulationSettings:
    tonemap_mode: str  = "aces"
    export_hdr:   bool = False   # if True, skip tonemap and return linear


@dataclass
class ObservationSettings:
    exposure:          float = 1.0
    psf_sigma:         float = 1.2
    bloom_threshold:   float = 0.8
    bloom_strength:    float = 0.3
    noise_photon_scale: float = 500.0   # higher = less noise
    read_noise_sigma:  float = 0.002
    tonemap_mode:      str   = "aces"
    gamma:             float = 2.2


@dataclass
class PortfolioSettings:
    contrast:           float = 1.15
    saturation_warm:    float = 1.3    # applied to warm (disk) tones
    saturation_global:  float = 1.1
    vignette_strength:  float = 0.25
    unsharp_radius:     float = 1.0
    unsharp_strength:   float = 0.4
    tonemap_mode:       str   = "aces"
    gamma:              float = 2.2


# ── Tonemappers ───────────────────────────────────────────────────────────────

def _tonemap_aces(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)


def _tonemap_reinhard(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + x)


def _tonemap_filmic(x: np.ndarray) -> np.ndarray:
    """Hejl-Burgess-Dawson filmic."""
    x = np.maximum(0.0, x - 0.004)
    return (x * (6.2*x + 0.5)) / (x * (6.2*x + 1.7) + 0.06)


def _tonemap_linear(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


_TONEMAPPERS = {
    "aces":     _tonemap_aces,
    "reinhard": _tonemap_reinhard,
    "filmic":   _tonemap_filmic,
    "linear":   _tonemap_linear,
}


def tonemap(img: np.ndarray, mode: str = "aces") -> np.ndarray:
    if mode not in _TONEMAPPERS:
        raise ValueError(f"Unknown tonemap mode '{mode}'. "
                         f"Available: {list(_TONEMAPPERS.keys())}")
    return _TONEMAPPERS[mode](img)


# ── Individual effect functions ───────────────────────────────────────────────
# Each takes an HDR float32 (H, W, 3) image and returns float32.
# Parameters are explicit — no hidden state.

def apply_exposure(img: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Linear exposure multiplier before tonemapping."""
    return img * exposure


def apply_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Gamma correction. Apply after tonemapping."""
    return np.clip(img, 0.0, 1.0) ** (1.0 / gamma)


def apply_psf(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    Gaussian PSF blur simulating finite telescope resolution.
    Applied per-channel to preserve colour accuracy.
    Tier 4 will replace this with Airy disk / Moffat profiles.
    """
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = gaussian_filter(img[:, :, c], sigma=sigma)
    return out


def apply_bloom(img: np.ndarray,
                threshold: float = 0.8,
                strength:  float = 0.3,
                radius:    float = 8.0) -> np.ndarray:
    """
    Physically motivated bloom: bright regions scatter into neighbours.
    Uses a smooth sigmoid threshold to avoid hard clipping edges.
    """
    from scipy.ndimage import gaussian_filter

    # Smooth threshold via sigmoid instead of hard clip
    # Transitions smoothly from 0 at threshold-margin to full above threshold
    margin   = 0.15
    t        = np.clip((img - (threshold - margin)) / margin, 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)   # smoothstep
    highlights = img * smooth_t

    bloom = np.empty_like(highlights)
    for c in range(3):
        bloom[:, :, c] = gaussian_filter(highlights[:, :, c], sigma=radius)

    return img + bloom * strength


def apply_noise(img: np.ndarray,
                photon_scale: float = 500.0,
                read_sigma:   float = 0.002) -> np.ndarray:
    """
    Shot noise (Poisson) + read noise (Gaussian).
    photon_scale: higher = more photons = less shot noise.
    read_sigma:   standard deviation of sensor read noise.
    """
    rng = np.random.default_rng()
    # Shot noise — Poisson on scaled counts
    counts      = np.clip(img * photon_scale, 0, None)
    shot        = rng.poisson(counts).astype(np.float32) / photon_scale
    # Read noise — tiny Gaussian
    read        = rng.normal(0.0, read_sigma, img.shape).astype(np.float32)
    return np.clip(shot + read, 0.0, None)


def apply_contrast_curve(img: np.ndarray,
                         contrast: float = 1.15) -> np.ndarray:
    """
    Filmic S-curve contrast on luminance channel.
    Preserves hue and saturation while lifting shadows/compressing highlights.
    """
    luma      = (0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2])
    luma_c    = np.clip(luma, 1e-6, 1.0) ** (1.0 / contrast)
    scale     = luma_c / (luma + 1e-6)
    return np.clip(img * scale[:, :, np.newaxis], 0.0, 1.0)


def apply_selective_saturation(img: np.ndarray,
                               warm_boost: float = 1.3,
                               global_boost: float = 1.1) -> np.ndarray:
    """
    Boosts saturation of warm (disk) tones more than cool/neutral tones.
    Avoids oversaturating stars while making disk emission pop.
    """
    luma      = (0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2])
    gray      = luma[:, :, np.newaxis]

    # Warmth mask: pixels where R > B are "warm" (disk emission)
    warmth    = np.clip(img[:,:,0] - img[:,:,2], 0.0, 1.0)[:, :, np.newaxis]
    sat_boost = global_boost + warmth * (warm_boost - global_boost)

    out = gray + sat_boost * (img - gray)
    return np.clip(out, 0.0, 1.0)


def apply_vignette(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Radial darkening toward frame edges."""
    h, w  = img.shape[:2]
    y, x  = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist  = np.sqrt(((x - cx) / cx)**2 + ((y - cy) / cy)**2)
    mask  = 1.0 - strength * np.clip(dist, 0.0, 1.0)
    return img * mask[:, :, np.newaxis]


def apply_unsharp_mask(img: np.ndarray,
                       radius: float = 1.0,
                       strength: float = 0.4) -> np.ndarray:
    """
    Recovers fine detail lost during PSF/bloom.
    At small radius: sharpening. At large radius: local contrast enhancement.
    """
    from scipy.ndimage import gaussian_filter
    blurred = np.empty_like(img)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(img[:, :, c], sigma=radius)
    return np.clip(img + strength * (img - blurred), 0.0, 1.0)


def apply_nan_guard(img: np.ndarray) -> np.ndarray:
    """Always first in every pipeline. Cleans up physics output."""
    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


# ── Pipeline definitions ──────────────────────────────────────────────────────
# A pipeline is an ordered list of (effect_fn, kwargs) tuples.
# Executed left to right. Each fn receives the current image as first arg.

Pipeline = List[Tuple[Callable, dict]]


def build_simulation_pipeline(s: SimulationSettings) -> Pipeline:
    pipe = [
        (apply_nan_guard, {}),
    ]
    if not s.export_hdr:
        pipe.append((tonemap, {"mode": s.tonemap_mode}))
    return pipe


def build_observation_pipeline(s: ObservationSettings) -> Pipeline:
    return [
        (apply_nan_guard,    {}),
        (apply_exposure,     {"exposure":      s.exposure}),
        (apply_bloom,        {"threshold":     s.bloom_threshold,
                              "strength":      s.bloom_strength}),
        (apply_psf,          {"sigma":         s.psf_sigma}),
        (apply_noise,        {"photon_scale":  s.noise_photon_scale,
                              "read_sigma":    s.read_noise_sigma}),
        (tonemap,            {"mode":          s.tonemap_mode}),
        (apply_gamma,        {"gamma":         s.gamma}),
    ]


def build_portfolio_pipeline(s: PortfolioSettings) -> Pipeline:
    return [
        (apply_nan_guard,            {}),
        (apply_bloom,                {"threshold": 0.75, "strength": 0.2}),
        (tonemap,                    {"mode":      s.tonemap_mode}),
        (apply_contrast_curve,       {"contrast":  s.contrast}),
        (apply_selective_saturation, {"warm_boost":    s.saturation_warm,
                                      "global_boost":  s.saturation_global}),
        (apply_unsharp_mask,         {"radius":    s.unsharp_radius,
                                      "strength":  s.unsharp_strength}),
        (apply_vignette,             {"strength":  s.vignette_strength}),
        (apply_gamma,                {"gamma":     s.gamma}),
    ]


# ── Pipeline executor ─────────────────────────────────────────────────────────

def run_pipeline(img: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    """Execute a pipeline in order. Each stage gets the previous output."""
    result = img.copy()
    for fn, kwargs in pipeline:
        result = fn(result, **kwargs)
    return result


# ── Named mode builder ────────────────────────────────────────────────────────

def build_pipeline(mode: str,
                   simulation_settings:  SimulationSettings  = None,
                   observation_settings: ObservationSettings = None,
                   portfolio_settings:   PortfolioSettings   = None) -> Pipeline:
    if mode == "simulation":
        return build_simulation_pipeline(simulation_settings  or SimulationSettings())
    elif mode == "observation":
        return build_observation_pipeline(observation_settings or ObservationSettings())
    elif mode == "portfolio":
        return build_portfolio_pipeline(portfolio_settings   or PortfolioSettings())
    else:
        raise ValueError(f"Unknown post-processing mode '{mode}'. "
                         f"Available: simulation, observation, portfolio")