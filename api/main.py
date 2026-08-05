"""
Module: api.main
FastAPI application entry point for the Null Geodesic Raytracer.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.rate_limiter import RateLimitMiddleware

app = FastAPI(
    title="Null Geodesic Raytracer API",
    description="Asynchronous distributed General Relativity Kerr black hole rendering web service.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(RateLimitMiddleware, max_requests=20, window_seconds=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(router)


@app.get("/ui")
@app.get("/app")
def serve_ui():
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))


@app.get("/")
def root():
    return {
        "service": "Null Geodesic Raytracer API",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "presets": "/api/v1/presets",
            "submit_render": "/api/v1/renders/image",
            "job_status": "/api/v1/jobs/{job_id}"
        }
    }
