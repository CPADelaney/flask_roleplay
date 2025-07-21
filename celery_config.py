# celery_config.py

from celery import Celery
import os
from celery.schedules import crontab
import logging  # Add logging

# Use REDIS as the default broker and result backend!
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# celery_config.py

"""
Consolidated Celery configuration with task prioritization, monitoring, and dead letter queues.
Supports both Redis and RabbitMQ brokers.
"""

from celery import Celery
import os
from celery.schedules import crontab
from celery.signals import task_failure, task_success, task_retry
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
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
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("Prometheus client not available, metrics disabled")

# Broker configuration - support both Redis and RabbitMQ
USE_RABBITMQ = os.getenv("USE_RABBITMQ", "false").lower() == "true"

if USE_RABBITMQ:
    CELERY_BROKER_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost//")
else:
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Result backend - always use Redis
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# Create the Celery app
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']  # Explicitly include the tasks module
)

# Define queue priorities (only used with RabbitMQ)
QUEUE_PRIORITIES = {
    'high': 10,
    'default': 5,
    'low': 1,
    'low_priority': 1  # Alias for compatibility
}

# Task routing based on priority
task_routes = {
    # High priority tasks
    'tasks.process_new_game_task': {'queue': 'high'},
    'tasks.create_npcs_task': {'queue': 'high'},
    
    # Default priority tasks
    'tasks.run_npc_learning_cycle_task': {'queue': 'default'},
    'tasks.nyx_memory_maintenance_task': {'queue': 'default'},
    'tasks.sweep_and_merge_nyx_split_brains': {'queue': 'default'},
    'tasks.monitor_nyx_performance_task': {'queue': 'default'},
    'tasks.aggregate_learning_metrics_task': {'queue': 'default'},
    'tasks.run_llm_periodic_checkpoint_task': {'queue': 'default'},
    
    # Low priority tasks
    'tasks.memory_maintenance_task': {'queue': 'low_priority'},
    'tasks.memory_embedding_consolidation_task': {'queue': 'low_priority'},
    'tasks.cleanup_old_performance_data_task': {'queue': 'low_priority'},
    'tasks.cleanup_*': {'queue': 'low'},
    'tasks.analytics_*': {'queue': 'low'}
}

# Base configuration
base_config = {
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'result_expires': 3600,  # 1 hour
    'worker_max_tasks_per_child': 1000,  # Prevent memory leaks
}

# Enhanced configuration for RabbitMQ
if USE_RABBITMQ:
    rabbitmq_config = {
        # Queue settings
        'task_queues': {
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
            },
            'low_priority': {  # Alias for compatibility
                'exchange': 'low',
                'routing_key': 'low',
                'queue_arguments': {'x-max-priority': QUEUE_PRIORITIES['low']}
            }
        },
        
        # Task routing
        'task_routes': task_routes,
        
        # Dead letter configuration
        'task_reject_on_worker_lost': True,
        
        # Task retry settings
        'task_default_retry_delay': 60,  # 1 minute
        'task_max_retries': 3,
        
        # Monitoring settings
        'worker_send_task_events': True,
        'task_send_sent_event': True
    }
    celery_app.conf.update({**base_config, **rabbitmq_config})
else:
    # Simpler configuration for Redis
    celery_app.conf.update(base_config)

# Comprehensive beat schedule combining both configurations
celery_app.conf.beat_schedule = {
    # --- NPC and Game Tasks ---
    'npc-learning-cycle-every-15-mins': {
        'task': 'tasks.run_npc_learning_cycle_task',
        'schedule': crontab(minute='*/15'),
    },
    
    # --- Nyx Brain Tasks ---
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3:00 AM UTC
        'options': {'queue': 'default'}
    },
    "sweep-and-merge-nyx-split-brains-every-5min": {
        "task": "tasks.sweep_and_merge_nyx_split_brains",
        "schedule": crontab(minute="*/5"),  # Runs every 5 mins
    },
    'llm-periodic-checkpoint-every-10min': {
        'task': 'tasks.run_llm_periodic_checkpoint_task',
        'schedule': crontab(minute='*/10'),  # Run every 10 minutes
        'args': (0, 0),  # Example: pass user_id=0, conversation_id=0
    },
    
    # --- Memory System Tasks ---
    'memory-system-maintenance-daily': {
        'task': 'tasks.memory_maintenance_task',
        'schedule': crontab(hour=4, minute=30),  # Daily at 4:30 AM UTC
        'options': {'queue': 'low_priority'}
    },
    'memory-embedding-consolidation-weekly': {
        'task': 'tasks.memory_embedding_consolidation_task',
        'schedule': crontab(day_of_week='sun', hour=5, minute=0),  # Weekly Sunday 5:00 AM UTC
        'options': {'queue': 'low_priority'}
    },
    
    # --- Performance Monitoring Tasks ---
    'monitor-nyx-performance-every-5-mins': {
        'task': 'tasks.monitor_nyx_performance_task',
        'schedule': crontab(minute='*/5'),
    },
    'aggregate-learning-metrics-hourly': {
        'task': 'tasks.aggregate_learning_metrics_task',
        'schedule': crontab(minute=0),  # Every hour at :00
    },
    
    # --- Cleanup Tasks ---
    'cleanup-old-performance-data-daily': {
        'task': 'tasks.cleanup_old_performance_data_task',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2:00 AM
        'options': {'queue': 'low_priority'}
    },
    'cleanup-old-data': {
        'task': 'tasks.cleanup_old_data_task',
        'schedule': crontab(hour=4, minute=0),  # Daily at 4:00 AM
        'options': {'queue': 'low'}
    },
    
    # --- Analytics Tasks ---
    'update-analytics': {
        'task': 'tasks.analytics_update_task',
        'schedule': crontab(minute=0),  # Every hour at :00
        'options': {'queue': 'low'}
    }
}

# Task monitoring signals (only if Prometheus is available)
if PROMETHEUS_AVAILABLE:
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
else:
    # Basic logging-only handlers
    @task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
        """Handle task failures."""
        task_name = sender.name if sender else 'unknown'
        logger.error(f"Task {task_name} (ID: {task_id}) failed: {exception}")

# Dead letter queue handling (only for RabbitMQ)
def setup_dead_letter_queues():
    """Set up dead letter queues for failed tasks."""
    if not USE_RABBITMQ:
        return {}
        
    try:
        from kombu import Exchange, Queue
        
        dead_letter_exchange = Exchange('dead-letter', type='direct')
        
        # Create dead letter queues for each priority queue
        dead_letter_queues = {}
        for priority in ['high', 'default', 'low']:
            dead_letter_queues[priority] = Queue(
                f'dead-letter-{priority}',
                exchange=dead_letter_exchange,
                routing_key=f'dead-letter-{priority}'
            )
        
        # Update queue settings to use dead letter queues
        for priority, queue in celery_app.conf.task_queues.items():
            if priority in ['high', 'default', 'low']:
                queue['queue_arguments'].update({
                    'x-dead-letter-exchange': 'dead-letter',
                    'x-dead-letter-routing-key': f'dead-letter-{priority}'
                })
        
        return dead_letter_queues
    except Exception as e:
        logger.warning(f"Could not set up dead letter queues: {e}")
        return {}

# Initialize dead letter queues
dead_letter_queues = setup_dead_letter_queues()

def get_queue_stats():
    """Get statistics about task queues."""
    try:
        stats = {}
        connection = celery_app.connection()
        
        queue_names = ['high', 'default', 'low'] if USE_RABBITMQ else ['celery']
        
        for queue_name in queue_names:
            try:
                queue = connection.SimpleQueue(queue_name)
                queue_size = queue.qsize()
                stats[queue_name] = {
                    'messages': queue_size,
                    'consumers': len(getattr(queue, 'consumer_tags', []))
                }
                if PROMETHEUS_AVAILABLE:
                    TASKS_QUEUED.labels(queue_name=queue_name).set(queue_size)
                queue.close()
            except Exception as e:
                logger.warning(f"Could not get stats for queue {queue_name}: {e}")
        
        # Get dead letter queue stats (RabbitMQ only)
        if USE_RABBITMQ and dead_letter_queues:
            for queue_name in ['high', 'default', 'low']:
                try:
                    dead_queue = connection.SimpleQueue(f'dead-letter-{queue_name}')
                    stats[f'dead-letter-{queue_name}'] = {
                        'messages': dead_queue.qsize()
                    }
                    dead_queue.close()
                except Exception as e:
                    logger.warning(f"Could not get stats for dead letter queue {queue_name}: {e}")
        
        connection.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        return {}

# Export commonly used items
__all__ = ['celery_app', 'get_queue_stats', 'QUEUE_PRIORITIES']
