"""
Module: api.routes
FastAPI REST API routes for job submission, progress tracking, and preset queries.
"""

import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from core.config import RenderConfig, JobStatus
from scripts.cam_presets import CAMERA_PRESETS
from api.schemas import (
    HealthCheckResponse,
    PresetsResponse,
    CameraPresetInfo,
    RenderJobSubmissionResponse,
    RenderJobStatusResponse,
)
from workers.celery_app import celery_app
from workers.tasks import render_image_task, test_redis_connection
from api.engine import render_frame_from_config

router = APIRouter(prefix="/api/v1", tags=["Render Service"])


@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    redis_ok = False
    try:
        ping_res = test_redis_connection.delay().get(timeout=2.0)
        redis_ok = "Successful" in str(ping_res)
    except Exception:
        redis_ok = False

    return HealthCheckResponse(redis_connected=redis_ok)


@router.get("/presets", response_model=PresetsResponse)
def list_presets():
    preset_dict = {}
    for name, p in CAMERA_PRESETS.items():
        preset_dict[name] = CameraPresetInfo(
            name=name,
            cam_pos=p["cam_pos"],
            look_at=p["look_at"],
            fov=p["fov"],
            note=p.get("note", "")
        )
    return PresetsResponse(
        presets=preset_dict,
        render_modes=["preview", "quality", "production"]
    )


@router.post("/renders/image", response_model=RenderJobSubmissionResponse)
def submit_render_job(config: RenderConfig, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    config_dict = config.model_dump()

    output_dir = Path(__file__).resolve().parent.parent / "output" / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_filepath = str(output_dir / f"{job_id}.png")

    try:
        # Enqueue task in Celery
        render_image_task.apply_async(args=[job_id, config_dict], task_id=job_id)
        status = JobStatus.QUEUED
        msg = "Job submitted successfully to background worker queue."
    except Exception as e:
        # Fallback to local async thread execution if Celery worker/Redis is offline
        status = JobStatus.PROCESSING
        msg = f"Worker queue unavailable ({str(e)}). Processing locally via background task."
        background_tasks.add_task(render_frame_from_config, config, out_filepath)

    return RenderJobSubmissionResponse(
        job_id=job_id,
        status=status,
        message=msg,
        status_url=f"/api/v1/jobs/{job_id}"
    )


@router.get("/jobs/{job_id}", response_model=RenderJobStatusResponse)
def get_job_status(job_id: str):
    res = celery_app.AsyncResult(job_id)

    output_dir = Path(__file__).resolve().parent.parent / "output" / "renders"
    file_path = output_dir / f"{job_id}.png"

    if file_path.exists():
        return RenderJobStatusResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress_pct=100.0,
            result_url=f"/output/renders/{job_id}.png"
        )

    if res.state == "PENDING":
        return RenderJobStatusResponse(job_id=job_id, status=JobStatus.QUEUED, progress_pct=0.0)
    elif res.state == "PROCESSING":
        meta = res.info or {}
        pct = meta.get("progress_pct", 50.0) if isinstance(meta, dict) else 50.0
        return RenderJobStatusResponse(job_id=job_id, status=JobStatus.PROCESSING, progress_pct=pct)
    elif res.state == "SUCCESS":
        return RenderJobStatusResponse(job_id=job_id, status=JobStatus.COMPLETED, progress_pct=100.0, result_url=f"/output/renders/{job_id}.png")
    elif res.state == "FAILURE":
        err = str(res.info) if res.info else "Execution failed"
        return RenderJobStatusResponse(job_id=job_id, status=JobStatus.FAILED, progress_pct=0.0, error_message=err)
    else:
        return RenderJobStatusResponse(job_id=job_id, status=JobStatus.PROCESSING, progress_pct=25.0)


@router.get("/jobs/{job_id}/image")
def get_job_image(job_id: str):
    output_dir = Path(__file__).resolve().parent.parent / "output" / "renders"
    file_path = output_dir / f"{job_id}.png"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Render result image not found or still processing.")

    return FileResponse(str(file_path), media_type="image/png")


@router.get("/jobs/{job_id}/metadata")
def get_job_metadata(job_id: str):
    output_dir = Path(__file__).resolve().parent.parent / "output" / "renders"
    json_path = output_dir / f"{job_id}.json"

    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Render metadata not found or still processing.")

    return FileResponse(str(json_path), media_type="application/json")
