# celery_tasks/celery_config.py

"""
Enhanced Celery configuration with task prioritization, monitoring, and dead letter queues.
"""

from celery import Celery
import os
from celery.schedules import crontab
from celery.signals import task_failure, task_success, task_retry
from prometheus_client import Counter, Histogram, Gauge
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Prometheus metrics
TASK_LATENCY = Histogram(
    'celery_task_latency_seconds',
    'Task execution time in seconds',
    ['task_name']
)
TASK_FAILURES = Counter(
    'celery_task_failures_total',
    'Number of failed tasks',
    ['task_name']
)
TASK_RETRIES = Counter(
    'celery_task_retries_total',
    'Number of task retries',
    ['task_name']
)
TASK_SUCCESS = Counter(
    'celery_task_success_total',
    'Number of successful tasks',
    ['task_name']
)
TASKS_QUEUED = Gauge(
    'celery_tasks_queued',
    'Number of tasks in queue',
    ['queue_name']
)

# Environment variables with defaults
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost//")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create the Celery app with priority queues
celery_app = Celery('tasks', broker=RABBITMQ_URL, backend=REDIS_URL)

# Define queue priorities
QUEUE_PRIORITIES = {
    'high': 10,
    'default': 5,
    'low': 1
}

# Task routing based on priority
task_routes = {
    # High priority tasks
    'tasks.process_new_game_task': {'queue': 'high'},
    'tasks.create_npcs_task': {'queue': 'high'},
    
    # Default priority tasks
    'tasks.nyx_memory_maintenance_task': {'queue': 'default'},
    
    # Low priority tasks
    'tasks.cleanup_*': {'queue': 'low'},
    'tasks.analytics_*': {'queue': 'low'}
}

# Enhanced Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Queue settings
    task_queues={
        'high': {
            'exchange': 'high',
            'routing_key': 'high',
            'queue_arguments': {'x-max-priority': QUEUE_PRIORITIES['high']}
        },
        'default': {
            'exchange': 'default',
            'routing_key': 'default',
            'queue_arguments': {'x-max-priority': QUEUE_PRIORITIES['default']}
        },
        'low': {
            'exchange': 'low',
            'routing_key': 'low',
            'queue_arguments': {'x-max-priority': QUEUE_PRIORITIES['low']}
        }
    },
    
    # Task routing
    task_routes=task_routes,
    
    # Dead letter configuration
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    
    # Task retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Prevent worker starvation
    worker_max_tasks_per_child=1000,  # Prevent memory leaks
    
    # Monitoring settings
    worker_send_task_events=True,
    task_send_sent_event=True
)

# Schedule periodic tasks
celery_app.conf.beat_schedule = {
    # Memory maintenance - daily at 3am
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),
        'options': {'queue': 'default'}
    },
    
    # Cleanup tasks - daily at 4am
    'cleanup-old-data': {
        'task': 'tasks.cleanup_old_data_task',
        'schedule': crontab(hour=4, minute=0),
        'options': {'queue': 'low'}
    },
    
    # Analytics tasks - every hour
    'update-analytics': {
        'task': 'tasks.analytics_update_task',
        'schedule': crontab(minute=0),
        'options': {'queue': 'low'}
    }
}

# Task monitoring signals
@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failures."""
    task_name = sender.name if sender else 'unknown'
    TASK_FAILURES.labels(task_name=task_name).inc()
    logger.error(f"Task {task_name} (ID: {task_id}) failed: {exception}")

@task_success.connect
def task_success_handler(sender=None, **kwargs):
    """Handle task successes."""
    task_name = sender.name if sender else 'unknown'
    TASK_SUCCESS.labels(task_name=task_name).inc()

@task_retry.connect
def task_retry_handler(sender=None, **kwargs):
    """Handle task retries."""
    task_name = sender.name if sender else 'unknown'
    TASK_RETRIES.labels(task_name=task_name).inc()

# Dead letter queue handling
def setup_dead_letter_queues():
    """Set up dead letter queues for failed tasks."""
    from kombu import Exchange, Queue
    
    dead_letter_exchange = Exchange('dead-letter', type='direct')
    
    # Create dead letter queues for each priority queue
    dead_letter_queues = {
        priority: Queue(
            f'dead-letter-{priority}',
            exchange=dead_letter_exchange,
            routing_key=f'dead-letter-{priority}'
        )
        for priority in QUEUE_PRIORITIES.keys()
    }
    
    # Update queue settings to use dead letter queues
    for priority, queue in celery_app.conf.task_queues.items():
        queue['queue_arguments'].update({
            'x-dead-letter-exchange': 'dead-letter',
            'x-dead-letter-routing-key': f'dead-letter-{priority}'
        })
    
    return dead_letter_queues

# Initialize dead letter queues
dead_letter_queues = setup_dead_letter_queues()

def get_queue_stats():
    """Get statistics about task queues."""
    try:
        stats = {}
        connection = celery_app.connection()
        
        for queue_name in QUEUE_PRIORITIES.keys():
            queue = connection.SimpleQueue(queue_name)
            stats[queue_name] = {
                'messages': queue.qsize(),
                'consumers': len(queue.consumer_tags)
            }
            TASKS_QUEUED.labels(queue_name=queue_name).set(queue.qsize())
            queue.close()
        
        # Get dead letter queue stats
        for queue_name, queue in dead_letter_queues.items():
            dead_queue = connection.SimpleQueue(f'dead-letter-{queue_name}')
            stats[f'dead-letter-{queue_name}'] = {
                'messages': dead_queue.qsize()
            }
            dead_queue.close()
        
        connection.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        return {} 
