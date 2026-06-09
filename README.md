# Geodesic Ray Tracer

A research-oriented General Relativity ray tracer that numerically integrates photon geodesics through curved spacetime to simulate gravitational lensing around black holes.

Unlike conventional graphics-based black hole renderers that rely on visual approximations, this project traces light rays directly through Schwarzschild and Kerr spacetimes using conserved quantities and numerical integration. The result is a physically motivated rendering framework capable of reproducing black hole shadows, Einstein rings, higher-order lensing structures, and frame-dragging effects around rotating black holes.

---

## Highlights

### Relativistic Ray Tracing

* Implemented photon propagation in both Schwarzschild and Kerr spacetimes.
* Numerically integrates null geodesics rather than approximating lensing with shaders.
* Supports near-extremal Kerr black holes (`a ≈ 0.998`).
* Computes photon trajectories using conserved quantities:

  * Energy (E)
  * Angular Momentum (L)
  * Carter Constant (Q)

### Gravitational Lensing

* Black hole shadow formation
* Einstein rings
* Photon sphere structures
* Higher-order lensing images
* Multi-orbit photon trajectories
* Frame dragging around rotating black holes

### Accretion Disk Rendering

* Multi-hit disk intersections
* Front-side and back-side disk imaging
* Higher-order lensed disk images
* Relativistic viewing geometries
* Configurable disk parameters

### Numerical Stability

Significant effort has been devoted to handling strong-field numerical challenges:

* Adaptive timestep integration
* Horizon-aware capture logic
* Near-singularity safeguards
* Pole proximity handling
* NaN / Inf detection
* Geodesic termination diagnostics
* Strong-field stability testing

---

## Accomplishments (XYZ Format)

### Implemented physically-based black hole rendering

**Measured by**

* Support for Schwarzschild and Kerr metrics
* Stable rendering at near-extremal spin (`a ≈ 0.998`)
* Accurate black hole shadow generation

**By**

* Numerically integrating null geodesics in curved spacetime using conserved-quantity formulations.

---

### Rendered higher-order gravitational lensing structures

**Measured by**

* Einstein ring formation
* Multi-hit accretion disk imaging
* Secondary and higher-order photon paths

**By**

* Tracing photon trajectories through strong-field Kerr spacetime rather than relying on image-space distortion techniques.

---

### Built a physics validation framework

**Measured by**

* Hamiltonian drift monitoring
* Conservation checks for E, L, and Q
* Geodesic diagnostic visualization modes

**By**

* Developing a dedicated Doctor Mode subsystem for numerical verification and debugging.

---

### Improved strong-field integration robustness

**Measured by**

* Elimination of horizon leakage artifacts
* Stable photon capture behavior
* Successful rendering of complex Kerr configurations

**By**

* Introducing adaptive stepping, capture diagnostics, and singularity-aware safeguards.

---

## Metric Framework

The renderer is built around a metric-driven architecture.

Rather than hard-coding a specific spacetime solution, the simulation pipeline is designed so that alternative metrics can be integrated while reusing the same rendering and diagnostic infrastructure.

### Implemented Metrics

#### Schwarzschild Metric

* Non-rotating black holes
* Photon sphere modeling
* Event horizon capture

#### Kerr Metric

* Rotating black holes
* Frame dragging
* Spin-dependent lensing
* Ergosphere-related studies

### Planned Metrics

* Reissner–Nordström
* Kerr–Newman
* Naked singularity solutions
* User-defined experimental metrics

This architecture allows the project to function as a spacetime experimentation framework rather than a single-purpose renderer.

---

## Doctor Mode

Doctor Mode is a dedicated diagnostic subsystem designed to validate the simulation and investigate unusual geodesic behavior.

Current and planned diagnostics include:

* Capture Maps
* Termination Maps
* Orbit Count Maps
* Disk Hit Maps
* Minimum Radius Maps
* Step Count Heatmaps
* Pole Proximity Diagnostics
* Ergosphere Occupancy Maps
* Impact Parameter Visualizations
* Hamiltonian Drift Maps
* Energy Drift Maps
* Angular Momentum Drift Maps
* Carter Constant Drift Maps

The goal is to distinguish genuine physical phenomena from numerical artifacts and provide visibility into photon behavior inside strong gravitational fields.

---

## Architecture

```text
Camera Rays
      │
      ▼
Coordinate Conversion
(Cartesian → Boyer-Lindquist)
      │
      ▼
Conserved Quantity Solver
(E, L, Q)
      │
      ▼
Geodesic Integrator
(Schwarzschild / Kerr)
      │
      ▼
Disk Intersection Engine
(Multi-Hit Support)
      │
      ▼
Radiative Model
      │
      ▼
Image Generation
```

The rendering layer is intentionally separated from the physics layer, allowing future integration with external visualization tools such as Blender without modifying the geodesic engine.

---

## Example Phenomena Reproduced

* Black hole shadows
* Einstein rings
* Frame dragging
* Photon orbit structures
* Higher-order accretion disk images
* Multi-crossing photon trajectories
* Strong-field gravitational lensing

---

## Performance

Typical render times vary significantly depending on camera placement, black hole spin, and geodesic complexity.

Observed render times range from a few seconds to over a minute per frame, with near-critical photon trajectories requiring substantially more integration work than direct escape or capture paths.

---

## Future Work

### Physics

* Symplectic / Hamiltonian integrators
* Ergosphere visualization
* Cauchy horizon studies
* Naked singularity investigations
* Higher-order photon ring analysis
* Geodesic family classification
* Additional spacetime metrics

### Diagnostics

* Full conservation-law monitoring
* Automated geodesic validation
* Escape basin analysis
* Orbit-family visualization
* Lyapunov and chaos diagnostics

### Rendering

* Blender integration pipeline
* Animation workflows
* GPU acceleration
* Volumetric accretion flow models
* Relativistic Doppler and beaming enhancements

---

## Why This Project Exists

This project began as an exploration of Python and computational physics and gradually evolved into a General Relativity simulation framework.

The long-term goal is not only to create realistic black hole renders, but also to build a platform for experimenting with spacetime geometries, studying geodesic behavior, validating numerical methods, and investigating both established and hypothetical gravitational systems.
