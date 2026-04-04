import time
from .celery_app import celery_app

@celery_app.task(name="test_connection")
def test_redis_connection():
    """
    A dummy task to verify the pipeline works.
    It waits 3 seconds (simulating math) and returns a success message.
    """
    print("📡 Task received! Ping from Python to Docker...")
    time.sleep(3) # Simulate a heavy calculation
    return "✅ Connection Successful: Python <-> Redis <-> Docker"