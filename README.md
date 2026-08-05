# KERR-TRACE — Null Geodesic Raytracer & Relativity Simulator

A research-oriented General Relativity raytracer and distributed rendering service that numerically integrates photon geodesics through curved Kerr spacetime to simulate gravitational lensing, relativistic accretion disks, and frame-dragging around black holes.

![KERR-TRACE Hero Render](docs/images/hero.png)
*Figure 1: High-resolution 1080p render of a near-extremal Kerr black hole ($a = 0.998$) with Keplerian differential rotation, Novikov-Thorne thermal envelope, and relativistic Doppler boosting.*

---

## 🖼️ Physical Render Gallery

To demonstrate the physical range of **KERR-TRACE**, the gallery below illustrates distinct gravitational spacetimes, inclination angles, and observer geometries:

| Schwarzschild ($a = 0.0$) | Near-Extremal Kerr ($a = 0.998$) |
| :---: | :---: |
| ![Schwarzschild Luminet](docs/images/schwarzschild_luminet.png) | ![Kerr High Inclination](docs/images/kerr_high_inclination.png) |
| **Symmetric Lensing (Luminet 1979 Geometry)**: Non-rotating black hole showing symmetric photon ring and circular shadow. | **Asymmetric Lensing ($85^\circ$ Inclination)**: Strong frame-dragging and asymmetric shadow distortion. |

| Strong Frame-Dragging | Plasma Turbulence Advection ($t=0 \to t=2$) |
| :---: | :---: |
| ![Frame Dragging](docs/images/kerr_frame_dragging.png) | ![Plasma Motion](docs/images/plasma_t2.png) |
| **Close-Range Observer**: Equatorial viewing geometry exposing frame-dragging warping near ISCO ($r_{\text{ISCO}} = 1.237 r_g$). | **Time-Advected Plasma**: Multi-octave turbulence advected via Keplerian angular frequency $\Omega(r)$. |

---

## 💫 Accretion Disk Physics & Relativistic Frequency Shifts

The accretion disk renderer implements a **Novikov-Thorne relativistic thin-disk model** coupled with Doppler boosting and gravitational redshift.

![Relativistic Frequency Shifts](docs/images/hero.png)
*Figure 2: Physical breakdown of accretion disk emission features under extreme Kerr lensing:*

1. **Approaching Side (Left / Foreground)**: Plasma moving toward the observer at relativistic speeds ($\sim 0.5c$). Intensity is strongly amplified by the relativistic Doppler factor $I_\nu \propto g^4$, shifting thermal emission into bright blueshifted tones.
2. **Receding Side (Right)**: Plasma moving away from the observer. Emission is attenuated and redshifted into dim, warm tones.
3. **Lensed Secondary Arch**: Light rays emitted from the back-side of the disk travel under the lower pole of the event horizon, bend around the photon sphere, and appear as a thin, highly brightened secondary arch above and below the shadow.

---

## 🌌 Celestial Skybox & Gravitational Lensing

When photon rays escape the strong-field region ($r > R_{\text{bounds}}$), they sample an equirectangular celestial skybox texture (`core/skybox.py`).

![Celestial Skybox Lensing](docs/images/skybox_lensing.png)
*Figure 3: Gravitational lensing of background celestial panorama around the black hole shadow:*

- **Einstein Rings & Star Distortion**: Background stars and deep space nebulae passing near the photon sphere are gravitationally lensed into circular Einstein arcs surrounding the central shadow boundary.

---

## 🔬 Doctor Mode & Conservation Diagnostics

Doctor Mode is a dedicated diagnostic subsystem (`doctor/diagnostics.py`) that computes 31 spatial diagnostic metrics per photon ray to inspect conservation laws and numerical behavior.

![Doctor Mode Diagnostic Map](docs/images/doctor_max_dphi_after.png)
*Figure 4: Doctor Mode diagnostic map plotting max single-step azimuthal step size $|d\phi|_{\text{step}}$ across 518,400 photon rays ($960 \times 540$).*

### Diagnostic Metrics Monitored:
- **Hamiltonian & Energy Drift**: Monitors $\Delta E / E_0$ and $\Delta L / L_0$ across $3,000+$ integration steps.
- **Minimum Radius & Orbit Counts**: Tracks closet approach $r_{\text{min}}$ and equatorial plane crossing counts.
- **Termination Reason Mapping**: Distinguishes genuine escape ($r > R_{\text{bounds}}$), horizon capture ($r < r_+$), and photon sphere trapping.

---

## 🛡️ Numerical Guardrails & Polar-Axis Artifact Fix

For computational physicists reviewing the codebase, this section documents the numerical resolution of the **Boyer-Lindquist Polar Singularity** ($\theta \to 0, \pi$).

| Guardrail OFF (Coordinate Singularity Artifact) | Guardrail ON (Polar-Conformal Adaptive Cap) |
| :---: | :---: |
| ![Needle Beam Artifact](docs/images/doctor_max_dphi_before.png) | ![Fixed Polar Cap](docs/images/doctor_max_dphi_after.png) |
| **Vertical Needle Beam Artifact**: Near poles ($\theta \to 0$), $g_{\phi\phi} \to 0 \implies d\phi/d\lambda \to \infty$. Finite RK4 step sizes cause single-step jumps $\Delta \phi > \pi$, creating a spurious vertical needle streak. | **Smooth Shadow Boundary**: Bounding $\Delta \phi_{\text{step}} \le 0.15\text{ rad}$ bounds RK4 Taylor truncation error to machine precision $\mathcal{O}(dt^5)$ without altering GR metric physics. |

### The Mathematical Formulation
In Boyer-Lindquist coordinates $(t, r, \theta, \phi)$, the Kretschmann scalar $K$ is completely smooth at $\theta = 0, \pi$. The polar needle is **100% a coordinate artifact** caused by metric term $g_{\phi\phi} = \left( r^2 + a^2 + \frac{2 M a^2 r \sin^2\theta}{\Sigma} \right) \sin^2\theta \to 0$.

The azimuthal coordinate velocity gives:
$$\frac{d\phi}{d\lambda} \propto \frac{L}{\sin^2\theta}$$

In `core/geodesics.py`, we implement the **Polar-Conformal Adaptive Step Cap**:
```python
# Bounding azimuthal displacement to 0.15 rad per step
dphi_scale = abs(L) / (sin2 + 1e-6)
polar_cap = 0.15 / dphi_scale if dphi_scale > 0.15 else 1.0

dt_local = dt * min(max(r_factor * theta_factor, 1e-4), polar_cap) * far_factor
```

---

## 🎨 KERR-TRACE Interactive Web UI

Served live at **`http://localhost:8000/ui`** or via ngrok tunnel **`https://touchily-steamerless-alyssa.ngrok-free.dev/ui`**:

![KERR-TRACE Web UI](docs/images/hero.png)

- **Sacred Edge-to-Edge Viewport**: Viewport occupies ~90% of screen area.
- **Lightroom-Style Slide-Up Drawer**: Bottom dock resting at 56px height, sliding up to expose active tab controls (`Spacetime`, `Observer`, `Accretion`, `Diagnostics`, `Export`).
- **Zero Input Box Outlines**: Clean, borderless text values (`Spin 0.998`, `Mass 1 M☉`, `Inclination 70°`).
- **Hero Render Button**: Prominent sky-blue (`#5CA9FF`) `⚡ Render Image` button.
- **Raw Frame Progress**: Displays real-time frame progress `Frame 37/100 | ETA 42s`.

---

## 🚀 Usage & Quick Start

### 1. Installation
```bash
git clone https://github.com/Spidey6978/null-geodesic-raytracer.git
cd null-geodesic-raytracer
pip install -r requirements.txt
```

### 2. Launching Web UI & Public Tunnel
```bash
# Launch public server with pre-configured static domain
python scripts/run_public_server.py --domain touchily-steamerless-alyssa.ngrok-free.dev
```
Open in browser:
👉 **`https://touchily-steamerless-alyssa.ngrok-free.dev/ui`**

### 3. Programmatic Python API
```python
from core.config import RenderConfig, BlackHoleConfig, CameraConfig, RenderMode
from api.engine import render_frame_from_config

config = RenderConfig(
    black_hole=BlackHoleConfig(mass=1.0, spin=0.998),
    camera=CameraConfig(preset="hero", fov=100.0),
    mode=RenderMode.PRODUCTION,
    skybox_path="procedural"
)

meta = render_frame_from_config(config, "output/my_blackhole.png")
print(f"Rendered in {meta['render_time_s']:.2f} seconds.")
```

### 4. Automated Test Suite
```bash
pytest
```

---

## 📜 Citation & License

```bibtex
@software{gopani_kerr_trace_2026,
  author = {Veer Gopani},
  title = {KERR-TRACE: A Distributed Null Geodesic Raytracer for Kerr Spacetime},
  url = {https://github.com/Spidey6978/null-geodesic-raytracer},
  year = {2026}
}
```

Licensed under the MIT License.