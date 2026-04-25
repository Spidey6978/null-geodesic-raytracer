🌌 Null Geodesic Raytracer

Overview

This project is a high-performance computational physics engine designed to simulate the extreme gravitational environments around black holes. By bypassing standard 3D graphics rasterization, this engine calculates the exact paths of light (null geodesics) through curved spacetime by numerically integrating differential equations derived from General Relativity.

🔬 Core Physics & Rendering Architecture

The simulation bridges complex astrophysics with optimized backend software engineering to handle massive computational loads.

Runge-Kutta 4 (RK4) Integration: Tracing photon trajectories deep within infinite gravity wells requires high-precision math to prevent orbital degradation. The engine implements a custom RK4 numerical solver based on the exact mathematical effective potential for the Schwarzschild Metric in Geometrized Units ($G=c=1$).

Vectorized Backward Ray-Tracing: To bypass the computational bottleneck of standard Python loops, the camera model is entirely vectorized using NumPy. Coupled with Numba JIT compilation, the core engine achieves near-C++ execution speeds, effectively calculating over 160,000 simultaneous light paths per frame during local shadow and halo rendering.

Accretion Disk Radiative Transfer: The system calculates precise ray-plane intersections at the black hole's equatorial plane ($y=0$), applying thermodynamic intensity gradients to generate the iconic lensed "halo" effect visible in astrophysical models.

🚀 Distributed Infrastructure

For high-resolution batch processing and eventual 60fps video generation, the engine scales horizontally rather than relying on a single thread.

Asynchronous Task Queue: The project implements a distributed rendering pipeline utilizing Celery and Redis. This architecture successfully distributes heavy differential math workloads across isolated Docker containers, allowing multiple worker nodes to compute independent chunks of the spacetime grid asynchronously.

🛠️ Technology Stack

Languages: Python 3

Scientific Computing: NumPy, SciPy

Performance Optimization: Numba (Just-In-Time Compilation)

Distributed Systems: Celery, Redis, Docker

Visualization: Matplotlib, Pillow

🗺️ Project Scope & Roadmap

The project is structured in iterative phases, moving from static mathematical proofs to a dynamic, relativistic rendering cluster:

[x] Phase 1: Spacetime Foundation: Implement Schwarzschild geodesic math and RK4 integrators.

[x] Phase 2: Spatial Optics: Vectorize the camera engine and implement accretion disk collision detection.

[ ] Phase 3: Special Relativity: Implement Doppler beaming and Lorentz transformations to calculate asymmetric luminosity caused by the disk spinning at near-light speeds.

[ ] Phase 4: The Kerr Metric: Upgrade the core Hamiltonians to support spinning black holes and frame dragging.

[ ] Phase 5: API Gateway: Deploy a FastAPI layer for remote simulation triggers and configuration uploads.

⚡ Quick Start

1. Environment Setup

python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt


2. Run Local Simulations

Run the single-threaded rendering scripts to visualize the current physics implementation:

python render_accretion_disk.py
python test_physics_viz.py


3. Start the Distributed Workers

To utilize the asynchronous worker pipeline for batch rendering:

# Start the Redis message broker
docker-compose up -d

# Start the Celery worker node
celery -A workers.celery_app worker --pool=solo --loglevel=info
