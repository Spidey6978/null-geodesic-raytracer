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
    exposure:            float = 1.0
    psf_sigma:           float = 1.2
    bloom_threshold:     float = 0.80
    bloom_strength:      float = 0.25
    highlight_threshold: float = 0.92
    highlight_strength:  float = 0.12
    halation_strength:   float = 0.06
    noise_photon_scale:  float = 500.0
    read_noise_sigma:    float = 0.002
    sensor_bit_depth:    int   = 14
    tonemap_mode:        str   = "aces"
    gamma:               float = 2.2


@dataclass
class PortfolioSettings:
    contrast:             float = 1.15
    local_contrast_r:     float = 30.0
    local_contrast_str:   float = 0.4
    saturation_warm:      float = 1.3
    saturation_global:    float = 1.1
    vignette_strength:    float = 0.25
    unsharp_radius:       float = 1.0
    unsharp_strength:     float = 0.35
    halation_strength:    float = 0.05
    highlight_strength:   float = 0.10
    film_grain_strength:  float = 0.010
    chromatic_shift:      float = 0.6
    tonemap_mode:         str   = "aces"
    gamma:                float = 2.2

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

# ── T2 — Physically Motivated Effects ────────────────────────────────────────

def apply_halation(img: np.ndarray,
                   threshold: float = 0.85,
                   strength:  float = 0.08,
                   radius:    float = 12.0) -> np.ndarray:
    """
    Film halation: red/orange bleed around intense highlights.
    Light scatters back through film emulsion base, biased toward red.
    Subtle — only the brightest disk/corona emission should trigger it.
    """
    from scipy.ndimage import gaussian_filter

    luma       = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    margin     = 0.1
    t          = np.clip((luma - (threshold - margin)) / margin, 0.0, 1.0)
    smooth_t   = t * t * (3.0 - 2.0 * t)

    # Red-biased scatter kernel — slightly larger radius for red channel
    halo_r     = gaussian_filter(img[:,:,0] * smooth_t, sigma=radius * 1.2)
    halo_g     = gaussian_filter(img[:,:,1] * smooth_t, sigma=radius * 0.6)
    halo_b     = gaussian_filter(img[:,:,2] * smooth_t, sigma=radius * 0.3)

    out        = img.copy()
    out[:,:,0] = np.clip(img[:,:,0] + halo_r * strength * 1.0, 0.0, None)
    out[:,:,1] = np.clip(img[:,:,1] + halo_g * strength * 0.4, 0.0, None)
    out[:,:,2] = np.clip(img[:,:,2] + halo_b * strength * 0.1, 0.0, None)
    return out


def apply_film_grain(img: np.ndarray,
                     strength:   float = 0.012,
                     seed:       int   = None) -> np.ndarray:
    """
    Luminance-weighted film grain.
    Grain is stronger in midtones, weaker in highlights and shadows —
    matches real film grain behaviour and avoids graining the black BH shadow.
    """
    rng    = np.random.default_rng(seed)
    luma   = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

    # Grain visibility peaks in midtones (luma ~0.5), falls off at extremes
    grain_mask = 4.0 * luma * (1.0 - luma)   # parabola: 0 at 0 and 1, peak at 0.5
    grain      = rng.normal(0.0, strength, img.shape[:2])
    grain     *= grain_mask

    out = img + grain[:, :, np.newaxis]
    return np.clip(out, 0.0, 1.0)


def apply_highlight_glow(img: np.ndarray,
                         threshold: float = 0.92,
                         strength:  float = 0.15,
                         radius:    float = 20.0) -> np.ndarray:
    """
    Very soft halo around extremely bright regions only.
    Different from bloom — much larger radius, much lower strength,
    only fires at near-saturation brightness. Represents optical scatter
    in the lens/detector system.
    """
    from scipy.ndimage import gaussian_filter

    luma     = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    margin   = 0.05
    t        = np.clip((luma - (threshold - margin)) / margin, 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)

    glow = np.empty_like(img)
    for c in range(3):
        glow[:,:,c] = gaussian_filter(img[:,:,c] * smooth_t, sigma=radius)

    return np.clip(img + glow * strength, 0.0, None)


def apply_sensor_clip(img: np.ndarray,
                      bit_depth: int = 14) -> np.ndarray:
    """
    Simulate finite detector bit depth.
    Quantises values to 2^bit_depth levels before saving.
    14-bit is typical for modern scientific CCDs.
    """
    levels = float(2 ** bit_depth - 1)
    return np.clip(np.round(img * levels) / levels, 0.0, 1.0)


# ── T3 — Portfolio Effects ────────────────────────────────────────────────────

def apply_local_contrast(img: np.ndarray,
                         radius:   float = 30.0,
                         strength: float = 0.5) -> np.ndarray:
    """
    Local contrast enhancement (large-radius unsharp mask on luma).
    Increases apparent detail and depth without changing global brightness.
    Radius much larger than apply_unsharp_mask — affects structure, not edges.
    """
    from scipy.ndimage import gaussian_filter

    luma      = (0.2126*img[:,:,0] + 0.7152*img[:,:,1]
                 + 0.0722*img[:,:,2])[:, :, np.newaxis]
    luma_blur = np.empty_like(img)
    for c in range(3):
        luma_blur[:,:,c] = gaussian_filter(img[:,:,c], sigma=radius)

    detail    = img - luma_blur
    out       = img + detail * strength
    return np.clip(out, 0.0, 1.0)


def apply_chromatic_shift(img: np.ndarray,
                          shift_pixels: float = 0.8) -> np.ndarray:
    """
    Chromatic aberration: tiny lateral colour fringing.
    Red channel shifted slightly outward from center, blue inward.
    Use only for portfolio/cinematic — not physically motivated for telescopes.
    Shift is sub-pixel by default so it reads as texture not distortion.
    """
    from scipy.ndimage import shift as ndshift

    h, w  = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    out = img.copy()
    # Red: shift outward (away from center) by shift_pixels
    out[:,:,0] = ndshift(img[:,:,0],
                          shift=[ shift_pixels * 0.5,  shift_pixels * 0.5],
                          mode='reflect')
    # Blue: shift inward (toward center)
    out[:,:,2] = ndshift(img[:,:,2],
                          shift=[-shift_pixels * 0.5, -shift_pixels * 0.5],
                          mode='reflect')
    return np.clip(out, 0.0, 1.0)


def apply_nan_guard(img: np.ndarray) -> np.ndarray:
    """Always first in every pipeline. Cleans up physics output."""
    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

# ── T4 — Advanced Effects ─────────────────────────────────────────────────────

def apply_airy_disk_psf(img: np.ndarray,
                        radius: float = 2.5,
                        rings:  int   = 2) -> np.ndarray:
    """
    Diffraction-limited telescope PSF — Airy disk pattern.
    More physically accurate than Gaussian for space telescopes.
    
    The Airy disk is the diffraction pattern from a circular aperture:
        I(r) = (2*J1(x)/x)^2  where x = pi*r*D/(lambda*f)
    
    We approximate it with a central Gaussian core plus attenuating rings,
    which is accurate for the first 2-3 diffraction rings and much faster
    than computing the full Bessel function kernel.
    
    radius: half-width of central disk in pixels (controls telescope resolution)
    rings:  number of diffraction rings to simulate (1-3 realistic)
    """
    from scipy.ndimage import gaussian_filter

    # Central Airy disk — tighter than a pure Gaussian
    core   = np.empty_like(img)
    for c in range(3):
        core[:,:,c] = gaussian_filter(img[:,:,c], sigma=radius * 0.42)

    result = core * 0.84   # central disk contains 84% of total energy

    # Diffraction rings — each ring broader and dimmer than previous
    ring_fractions = [0.07, 0.03, 0.02][:rings]
    ring_radii     = [radius * 1.64, radius * 2.66, radius * 3.70][:rings]

    for frac, r in zip(ring_fractions, ring_radii):
        ring = np.empty_like(img)
        for c in range(3):
            ring[:,:,c] = gaussian_filter(img[:,:,c], sigma=r)
        # Rings are annular — subtract inner from outer contribution
        result = result + ring * frac

    return result


def apply_moffat_psf(img: np.ndarray,
                     fwhm:  float = 3.0,
                     beta:  float = 2.5) -> np.ndarray:
    """
    Moffat PSF — more realistic than Gaussian for ground-based telescopes.
    Models atmospheric seeing (turbulence in Earth's atmosphere).
    
    Moffat profile: I(r) = (1 + (r/alpha)^2)^(-beta)
    
    beta controls the wing falloff:
        beta=1.5  : strong seeing, fat wings
        beta=2.5  : typical good seeing
        beta=4.0+ : approaches Gaussian (space telescope equivalent)
    
    fwhm: full-width half-maximum in pixels
    """
    h, w = img.shape[:2]

    # Build the Moffat kernel
    alpha  = fwhm / (2.0 * np.sqrt(2.0 ** (1.0/beta) - 1.0))
    size   = int(np.ceil(fwhm * 4)) | 1   # odd size
    half   = size // 2
    y, x   = np.mgrid[-half:half+1, -half:half+1]
    r2     = (x**2 + y**2) / alpha**2
    kernel = (1.0 + r2) ** (-beta)
    kernel = kernel / kernel.sum()   # normalise to unit energy

    from scipy.ndimage import convolve
    out = np.empty_like(img)
    for c in range(3):
        out[:,:,c] = convolve(img[:,:,c], kernel, mode='reflect')
    return out


def apply_adaptive_bloom(img: np.ndarray,
                         base_threshold: float = 0.75,
                         base_strength:  float = 0.25,
                         radius:         float = 8.0) -> np.ndarray:
    """
    Adaptive bloom: strength scales with local scene brightness.
    Bright scenes get stronger bloom, dark scenes get weaker bloom.
    More physically accurate than fixed-strength bloom since scatter
    in optical systems scales with incident flux.
    
    The adaptation factor is computed from the mean luminance of the
    bright region, so a nearly-saturated disk produces more scatter
    than a dim secondary image even if both exceed the threshold.
    """
    from scipy.ndimage import gaussian_filter

    luma     = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    margin   = 0.12
    t        = np.clip((luma - (base_threshold - margin)) / margin, 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)

    # Adaptation: bloom strength scales with mean brightness above threshold
    bright_region   = luma * smooth_t
    mean_brightness = bright_region.mean() + 1e-6
    adapt_factor    = np.clip(mean_brightness * 8.0, 0.5, 2.5)
    effective_str   = base_strength * adapt_factor

    bloom = np.empty_like(img)
    for c in range(3):
        bloom[:,:,c] = gaussian_filter(img[:,:,c] * smooth_t, sigma=radius)

    return img + bloom * effective_str

def apply_false_color(img: np.ndarray,
                      quantity_map: np.ndarray,
                      colormap:     str   = "inferno",
                      vmin:         float = None,
                      vmax:         float = None,
                      alpha:        float = 1.0,
                      log_scale:    bool  = False) -> np.ndarray:
    """
    Overlays or replaces the image with a false-color map of a scalar quantity.
    
    quantity_map: (H, W) float array — any doctor diagnostic field.
                  e.g. Doppler factor, gravitational redshift, orbit count,
                  Hamiltonian drift, steps taken.
    alpha:        0.0 = pure false-color, 1.0 = pure physics render,
                  intermediate = blend. 0.0 useful for papers, 0.5 for
                  showing quantity overlaid on the actual image.
    log_scale:    True for quantities spanning many orders of magnitude
                  (H drift, inv_sin2) — False for linear quantities (orbit count).
    
    Usage example:
        doppler_map = tensor[:,:, IDX_IMPACT_PARAM]
        result = apply_false_color(image, doppler_map, colormap="coolwarm",
                                   vmin=-8, vmax=8, alpha=0.0)
    """
    import matplotlib.cm as mcm

    q = quantity_map.astype(np.float64)

    if log_scale:
        q = np.where(q > 0, np.log10(np.clip(q, 1e-12, None)), -12.0)
        if vmin is not None: vmin = np.log10(max(vmin, 1e-12))
        if vmax is not None: vmax = np.log10(max(vmax, 1e-12))

    lo = vmin if vmin is not None else float(np.nanmin(q))
    hi = vmax if vmax is not None else float(np.nanmax(q))
    span = hi - lo if (hi - lo) > 1e-12 else 1.0

    norm    = np.clip((q - lo) / span, 0.0, 1.0)
    norm    = np.nan_to_num(norm, 0.0)
    cmap    = mcm.get_cmap(colormap)
    colored = cmap(norm)[:, :, :3].astype(np.float32)

    # Blend with physics render
    return (alpha * img + (1.0 - alpha) * colored).astype(np.float32)


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
        (apply_nan_guard,       {}),
        (apply_exposure,        {"exposure":      s.exposure}),
        (apply_highlight_glow,  {"threshold":     s.highlight_threshold,
                                 "strength":      s.highlight_strength}),
        (apply_bloom,           {"threshold":     s.bloom_threshold,
                                 "strength":      s.bloom_strength}),
        (apply_halation,        {"strength":      s.halation_strength}),
        (apply_psf,             {"sigma":         s.psf_sigma}),
        (apply_noise,           {"photon_scale":  s.noise_photon_scale,
                                 "read_sigma":    s.read_noise_sigma}),
        (tonemap,               {"mode":          s.tonemap_mode}),
        (apply_sensor_clip,     {"bit_depth":     s.sensor_bit_depth}),
        (apply_gamma,           {"gamma":         s.gamma}),
    ]


def build_portfolio_pipeline(s: PortfolioSettings) -> Pipeline:
    return [
        (apply_nan_guard,            {}),
        (apply_highlight_glow,       {"strength":      s.highlight_strength}),
        (apply_halation,             {"strength":      s.halation_strength}),
        (apply_bloom,                {"threshold":     0.75,
                                     "strength":      0.18}),
        (tonemap,                    {"mode":          s.tonemap_mode}),
        (apply_contrast_curve,       {"contrast":      s.contrast}),
        (apply_local_contrast,       {"radius":        s.local_contrast_r,
                                     "strength":      s.local_contrast_str}),
        (apply_selective_saturation, {"warm_boost":    s.saturation_warm,
                                     "global_boost":  s.saturation_global}),
        (apply_unsharp_mask,         {"radius":        s.unsharp_radius,
                                     "strength":      s.unsharp_strength}),
        (apply_chromatic_shift,      {"shift_pixels":  s.chromatic_shift}),
        (apply_film_grain,           {"strength":      s.film_grain_strength}),
        (apply_vignette,             {"strength":      s.vignette_strength}),
        (apply_gamma,               {"gamma":          s.gamma}),
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