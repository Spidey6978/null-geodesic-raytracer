🌌 Null Geodesic Raytracer & Relativistic Astrophysics Engine

Overview

This project is a high-performance computational physics engine and distributed rendering pipeline designed to simulate the extreme environments around black holes. Bypassing traditional 3D graphics rasterization, this engine calculates the exact paths of light (null geodesics) through curved spacetime by numerically integrating differential equations derived from General and Special Relativity.

The complete architectural scope encompasses an offline scientific renderer, a distributed asynchronous computing cluster for rendering high-resolution video arrays, and a RESTful API for triggering rendering jobs. It accurately visualizes the event horizon shadow, photon spheres, gravitational lensing of background starfields, and the asymmetric luminosity of relativistic accretion disks.

🔬 Core Physics & Mathematical Engine

The simulation bridges complex astrophysics with optimized backend software engineering:

Spacetime Metrics: Implements the Schwarzschild Metric (static) and is architected to support the Kerr Metric (rotating), simulating phenomena like frame-dragging (Lense-Thirring effect) and oblate shadow deformation.

Runge-Kutta 4 (RK4) Integration: Developed a high-precision 4th-order numerical solver to accurately trace photon trajectories deep within infinite gravity wells, ensuring zero orbital degradation over thousands of integration steps.

Vectorized Backward Ray-Tracing: Engineered a pinhole camera model utilizing NumPy vectorization, generating millions of 3D directional vectors simultaneously to bypass the computational bottleneck of standard Python loops.

Accretion Disk Radiative Transfer: Calculates precise ray-plane intersections at the black hole's equatorial plane. Applies thermodynamic intensity gradients based on the photon's radial impact distance to generate the iconic lensed "halo" effect.

Special Relativity (Doppler Beaming): Incorporates Lorentz transformations to calculate redshift and blueshift, generating physically accurate asymmetric luminosity (Doppler boosting) caused by gas orbiting at near-light speeds.

🚀 Architecture & Systems Engineering

JIT Compilation (Numba): The core physics loops are decorated with @njit, compiling the Python integration math directly into optimized LLVM machine code. This bypasses the Global Interpreter Lock (GIL) and achieves near C-level execution speeds for massive ray-tracing workloads.

Distributed Task Queue (Celery & Redis): Architected a distributed computing pipeline capable of rendering high-definition video frames across a cluster. Utilizes Celery as the asynchronous task executor and containerized Redis (via Docker) as the message broker to distribute millions of ray calculations across multiple CPU cores.

RESTful API Engine: Features a FastAPI interface to manage simulation parameters, queue rendering jobs, and monitor worker node status in real-time.

🛠️ Technology Stack

Languages: Python 3

Scientific Computing: NumPy, SciPy

Performance Optimization: Numba (Just-In-Time Compilation)

Distributed Systems: Celery, Redis, Docker

Web/API Backend: FastAPI, Uvicorn

Visualization/Media: Matplotlib, OpenCV, Pillow

🗺️ Project Scope & Roadmap

The project is structured in iterative phases, moving from static mathematical proofs to a fully dynamic, distributed rendering cluster:

[x] Phase 1: Spacetime Foundation: Implement Schwarzschild geodesic math and RK4 integrators.

[x] Phase 2: Spatial Optics: Vectorize the camera engine and implement accretion disk collision detection (the "Halo" effect).

[ ] Phase 3: Special Relativity: Implement Doppler beaming, relativistic aberration, and frequency shifts for the accretion disk.

[ ] Phase 4: The Kerr Metric: Upgrade the core Hamiltonians to support spinning black holes and frame dragging.

[ ] Phase 5: Environment Mapping: Integrate EXR skybox distortion to accurately simulate the gravitational lensing of the surrounding galaxy.

[ ] Phase 6: Distributed Video Pipeline: Finalize the Celery/Redis architecture to process batch jobs for 60fps video rendering via OpenCV.

[ ] Phase 7: API Gateway: Deploy the FastAPI layer for remote simulation triggers and configuration uploads.

⚡ Quick Start

1. Environment Setup

python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt


2. Run Local Simulations

Run the single-threaded rendering scripts to visualize the current physics implementation:

python render_accretion_disk.py
python render_first_light.py


3. Distributed Rendering (Infrastructure)

To utilize the asynchronous worker pipeline for batch processing:

# Start the Redis message broker
docker-compose up -d

# Start the Celery worker node
celery -A workers.celery_app worker --pool=solo --loglevel=info
