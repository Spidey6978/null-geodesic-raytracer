"""
Module: core.config
Pydantic data models and schemas for Black Hole Simulation configurations,
camera setups, and asynchronous render job tracking.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RenderMode(str, Enum):
    PREVIEW = "preview"
    QUALITY = "quality"
    PRODUCTION = "production"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BlackHoleConfig(BaseModel):
    mass: float = Field(default=1.0, gt=0.0, le=100.0, description="Mass of the black hole in M_sun units")
    spin: float = Field(default=0.998, ge=0.0, le=0.998, description="Dimensionless spin parameter a (0 <= a <= 0.998)")
    disk_inner: Optional[float] = Field(default=None, description="Inner disk radius (defaults to ISCO)")
    disk_outer: float = Field(default=36.0, gt=1.0, le=200.0, description="Outer disk radius")


class CameraConfig(BaseModel):
    preset: Optional[str] = Field(default="hero", description="Camera preset name (e.g. hero, high_inclination, luminet_1979)")
    cam_pos: List[float] = Field(default_factory=lambda: [4.5, 2.5, 10.0], description="Cartesian position [X, Y, Z]")
    look_at: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Target look-at vector [X, Y, Z]")
    fov: float = Field(default=100.0, ge=10.0, le=170.0, description="Field of view in degrees")
    roll: float = Field(default=0.0, description="Camera roll angle in degrees")


class RenderConfig(BaseModel):
    black_hole: BlackHoleConfig = Field(default_factory=BlackHoleConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    width: Optional[int] = Field(default=None, ge=100, le=3840, description="Image width in pixels")
    height: Optional[int] = Field(default=None, ge=100, le=2160, description="Image height in pixels")
    dt: Optional[float] = Field(default=None, gt=0.001, le=1.0, description="Integration step size")
    max_steps: Optional[int] = Field(default=None, ge=100, le=20000, description="Maximum steps per photon ray")
    mode: Optional[RenderMode] = Field(default=RenderMode.QUALITY, description="Quality preset mode")
    frame_time: float = Field(default=0.0, description="Frame timestamp for plasma rotation and advection")


class RenderJobRequest(BaseModel):
    config: RenderConfig = Field(default_factory=RenderConfig)


class RenderJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    created_at: str
    render_time_s: Optional[float] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None


class AnimationConfig(BaseModel):
    black_hole: BlackHoleConfig = Field(default_factory=BlackHoleConfig)
    waypoints: List[List[float]] = Field(
        default_factory=lambda: [[0.0, 5.0, 15.0], [5.0, 2.0, 10.0], [10.0, 0.0, 5.0]],
        description="Array of 3D camera waypoints [[X1,Y1,Z1], [X2,Y2,Z2], ...]"
    )
    look_at: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Target look-at vector")
    fov: float = Field(default=100.0, ge=10.0, le=170.0, description="Field of view in degrees")
    roll: float = Field(default=0.0, description="Camera roll angle in degrees")
    num_frames: int = Field(default=30, ge=2, le=300, description="Total animation frames")
    fps: int = Field(default=30, ge=1, le=60, description="Frames per second")
    dt: Optional[float] = Field(default=None, gt=0.001, le=1.0)
    max_steps: Optional[int] = Field(default=None, ge=100, le=20000)
    mode: Optional[RenderMode] = Field(default=RenderMode.PREVIEW)
