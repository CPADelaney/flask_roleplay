# celery_worker.py
import eventlet
eventlet.monkey_patch()

import subprocess

if __name__ == "__main__":
    # The same args you used before:
    cmd = [
        "celery",
        "-A", "main:celery_app",
        "worker",
        "-P", "eventlet",
        "--loglevel=INFO",
    ]
    print("DEBUG: Running subprocess:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    print("Celery CLI exited with code:", result.returncode)
