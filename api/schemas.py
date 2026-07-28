"""
Module: api.schemas
Pydantic API request and response schemas for FastAPI endpoints.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from core.config import RenderConfig, JobStatus


class HealthCheckResponse(BaseModel):
    status: str = "ok"
    redis_connected: bool
    service: str = "Null Geodesic Raytracer API"
    version: str = "1.0.0"


class CameraPresetInfo(BaseModel):
    name: str
    cam_pos: List[float]
    look_at: List[float]
    fov: float
    note: str


class PresetsResponse(BaseModel):
    presets: Dict[str, CameraPresetInfo]
    render_modes: List[str]


class RenderJobSubmissionResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    status_url: str


class RenderJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: float
    created_at: Optional[str] = None
    render_time_s: Optional[float] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
