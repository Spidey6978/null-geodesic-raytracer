import sys
import time
from workers.tasks import test_redis_connection

def run_check():
    print("--- 🚀 STARTING INFRASTRUCTURE CHECK ---")
    
    try:
        # 1. Send the task to Redis
        print("1. Sending task to Celery/Redis...")
        task = test_redis_connection.delay()
        print(f"   -> Task ID: {task.id}")
        
        # 2. Wait for the result
        print("2. Waiting for worker to pick it up (Timeout: 10s)...")
        # We poll the status every 0.5 seconds
        for _ in range(20):
            if task.ready():
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()
        
        if task.ready():
            result = task.get()
            print(f"3. Result received: {result}")
            print("\n✅ SYSTEM GREEN: Ready for Physics Engine.")
        else:
            print("\n❌ TIMEOUT: The worker didn't pick up the task.")
            print("   (Did you remember to start the 'celery' terminal?)")
            
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("   (Is Docker running? Is Redis up?)")

if __name__ == "__main__":
    run_check()