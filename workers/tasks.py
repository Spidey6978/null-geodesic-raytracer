"""
Module: workers.tasks
Asynchronous background tasks for Celery worker nodes.
"""

import time
from pathlib import Path
from workers.celery_app import celery_app
from core.config import RenderConfig, JobStatus
from api.engine import render_frame_from_config


@celery_app.task(bind=True)
def test_redis_connection(self):
    """A sanity ping task to verify Redis and worker connectivity."""
    print(" Task received! Ping from Python to Docker/Redis...")
    time.sleep(1)
    return "✅ Connection Successful: Python <-> Redis <-> Docker Worker"


@celery_app.task(bind=True)
def render_image_task(self, job_id: str, config_dict: dict):
    """
    Executes background black hole ray tracing render job.
    Updates progress state and saves output PNG artifact.
    """
    try:
        self.update_state(state="PROCESSING", meta={"progress_pct": 10.0, "status": JobStatus.PROCESSING})

        config = RenderConfig(**config_dict)
        output_dir = Path(__file__).resolve().parent.parent / "output" / "renders"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_filepath = str(output_dir / f"{job_id}.png")

        self.update_state(state="PROCESSING", meta={"progress_pct": 30.0, "status": JobStatus.PROCESSING})

        meta = render_frame_from_config(config, out_filepath)

        self.update_state(state="SUCCESS", meta={"progress_pct": 100.0, "status": JobStatus.COMPLETED})

        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "progress_pct": 100.0,
            "render_time_s": meta["render_time_s"],
            "result_file": out_filepath,
            "result_url": f"/output/renders/{job_id}.png"
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"progress_pct": 0.0, "status": JobStatus.FAILED, "error": str(e)})
        raise e


@celery_app.task(bind=True)
def render_animation_task(self, job_id: str, anim_config_dict: dict):
    """
    Executes background multi-frame camera spline animation rendering.
    Tracks per-frame progress updates and compiles output video/image sequence.
    """
    from core.config import AnimationConfig
    from api.engine import render_animation_sequence

    try:
        self.update_state(state="PROCESSING", meta={"progress_pct": 5.0, "status": JobStatus.PROCESSING})

        anim_config = AnimationConfig(**anim_config_dict)
        output_dir = Path(__file__).resolve().parent.parent / "output" / "animations"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = str(output_dir / f"{job_id}.mp4")

        def progress_cb(current_frame, total_frames, pct):
            self.update_state(
                state="PROCESSING",
                meta={
                    "progress_pct": round(pct, 1),
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                    "status": JobStatus.PROCESSING
                }
            )

        meta = render_animation_sequence(anim_config, out_video_path, progress_callback=progress_cb)

        self.update_state(state="SUCCESS", meta={"progress_pct": 100.0, "status": JobStatus.COMPLETED})

        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "progress_pct": 100.0,
            "total_render_time_s": meta["total_render_time_s"],
            "video_file": meta["video_file"],
            "result_url": f"/output/animations/{job_id}.mp4" if meta["video_compiled"] else f"/output/animations/{job_id}_frames/"
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"progress_pct": 0.0, "status": JobStatus.FAILED, "error": str(e)})
        raise e