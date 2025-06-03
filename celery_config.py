# celery_config.py

from celery import Celery
import os
from celery.schedules import crontab
import logging  # Add logging

# Use REDIS as the default broker and result backend!
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create the Celery app
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']  # Explicitly include the tasks module
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # task_track_started=True, # Uncomment if needed
    # Improve worker behavior - consider adjusting concurrency later
    worker_prefetch_multiplier=1,  # Often better for I/O bound tasks
    task_acks_late=True,           # Ensure task only ack'd after completion
)

# --- Celery Beat Schedule ---
celery_app.conf.beat_schedule = {
    # --- Existing Tasks ---
    'npc-learning-cycle-every-15-mins': {
        'task': 'tasks.run_npc_learning_cycle_task',
        'schedule': crontab(minute='*/15'),
    },
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3:00 AM UTC
    },
    'memory-system-maintenance-daily': {
        'task': 'tasks.memory_maintenance_task',
        'schedule': crontab(hour=4, minute=30),  # Daily at 4:30 AM UTC
        'options': {'queue': 'low_priority'}  # Example queue name
    },
    'memory-embedding-consolidation-weekly': {
        'task': 'tasks.memory_embedding_consolidation_task',
        'schedule': crontab(day_of_week='sun', hour=5, minute=0),  # Weekly Sunday 5:00 AM UTC
        'options': {'queue': 'low_priority'}
    },
    
    # --- Modified Sweep Task (schedule remains, logic check added in task) ---
    "sweep-and-merge-nyx-split-brains-every-5min": {
        "task": "tasks.sweep_and_merge_nyx_split_brains",
        "schedule": crontab(minute="*/5"),  # Runs every 5 mins
        # Logic inside the task will check if app is ready before proceeding
    },
    
    # --- NEW Performance Monitoring Tasks ---
    'monitor-nyx-performance-every-5-mins': {
        'task': 'tasks.monitor_nyx_performance_task',
        'schedule': crontab(minute='*/5'),
    },
    'aggregate-learning-metrics-hourly': {
        'task': 'tasks.aggregate_learning_metrics_task',
        'schedule': crontab(minute=0),  # Every hour at :00
    },
    'cleanup-old-data-daily': {
        'task': 'tasks.cleanup_old_performance_data_task',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2:00 AM
        'options': {'queue': 'low_priority'}
    },
    
    # --- NEW Periodic LLM Checkpointing Task ---
    'llm-periodic-checkpoint-every-10min': {
        'task': 'tasks.run_llm_periodic_checkpoint_task',  # New task name
        'schedule': crontab(minute='*/10'),  # Run every 10 minutes (adjust as needed)
        # Args might be needed depending on how you manage NyxBrain instances per task
        # If each task gets the 'default' brain (user 0, conv 0):
        'args': (0, 0),  # Example: pass user_id=0, conversation_id=0
        # If targeting specific Nyx IDs, you might need a different scheduler approach
    },
}
