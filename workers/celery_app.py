"""
Module: workers.celery_app
The configuration entry point for the Distributed Task Queue.
"""
import os
from celery import Celery
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

# 2. Get Redis URL from .env (defaults to localhost if missing)
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# 3. Initialize the Celery App
# 'black_hole_worker' is the name of this specific worker instance
celery_app = Celery(
    "black_hole_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["workers.test_task"] # This tells Celery where to look for jobs
)

# 4. Configure Optimization settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker Optimization:
    # If a task crashes, don't kill the whole worker
    worker_concurrency=4, 
    task_acks_late=True,
)

if __name__ == "__main__":
    celery_app.start()