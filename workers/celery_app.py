"""
Module: workers.celery_app
Configuration entry point for Celery Distributed Task Queue.
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "black_hole_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=4,
    task_acks_late=True,
    task_track_started=True,
)

if __name__ == "__main__":
    celery_app.start()