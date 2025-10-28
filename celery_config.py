# celery_config.py

"""
Consolidated Celery configuration with task prioritization, monitoring, and dead letter queues.
Supports both Redis and RabbitMQ brokers.
Worker initialization is handled by db.connection module for proper asyncpg pool management.
"""

from celery import Celery
import os
from celery.schedules import crontab
from celery.signals import task_failure, task_success, task_retry
import logging

try:
    from nyx.tasks.queues import QUEUES as NYX_TASK_QUEUES, ROUTES as NYX_TASK_ROUTES
except Exception:  # pragma: no cover - queues are optional during tests
    NYX_TASK_QUEUES = ()
    NYX_TASK_ROUTES = {}

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

try:
    from nyx.tasks.beat.periodic import BEAT_SCHEDULE as NYX_BEAT_SCHEDULE
except Exception:  # pragma: no cover - optional during tests
    NYX_BEAT_SCHEDULE = {}

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
    include=['tasks', 'nyx.tasks']  # Explicitly include the legacy and new task modules
)

# Define queue priorities (only used with RabbitMQ)
QUEUE_PRIORITIES = {
    'realtime': 10,
    'background': 5,
    'heavy': 1,
}

LIVE_QUEUE_NAMES = tuple(QUEUE_PRIORITIES.keys())
CELERY_QUEUE_EXCHANGE = os.getenv("CELERY_QUEUE_EXCHANGE", "nyx")

# Task routing based on priority
task_routes = {
    # Background processing tasks
    'tasks.process_new_game_task': {'queue': 'background'},
    'tasks.create_npcs_task': {'queue': 'background'},
    'tasks.ensure_npc_pool_task': {'queue': 'background'},
    'tasks.background_chat_task_with_memory': {'queue': 'background'},
    'tasks.generate_lore_background_task': {'queue': 'background'},
    'tasks.generate_initial_conflict_task': {'queue': 'background'},
    'tasks.refresh_cultural_conflict_cache': {'queue': 'heavy'},
    'tasks.refresh_all_cultural_conflict_caches': {'queue': 'heavy'},

    'tasks.generate_and_cache_mpf_lore': {'queue': 'heavy'}, # LLM-intensive
    'tasks.lore_evolution_task': {'queue': 'heavy'},         # LLM-intensive
    'tasks.quick_setup_world_task': {'queue': 'heavy'},      # Very LLM-intensive

    ## NEW ##: Add routes for the new conflict system tasks
    'tasks.update_edge_case_scan': {'queue': 'background'},
    'tasks.update_tension_bundle_cache': {'queue': 'heavy'}, # This is LLM-heavy, good for this queue
    'tasks.periodic_edge_case_maintenance': {'queue': 'background'}, # The maintenance task itself is lightweight

    # Default priority tasks
    'tasks.run_npc_learning_cycle_task': {'queue': 'background'},
    'tasks.nyx_memory_maintenance_task': {'queue': 'heavy'},
    'tasks.sweep_and_merge_nyx_split_brains': {'queue': 'heavy'},
    'tasks.monitor_nyx_performance_task': {'queue': 'background'},
    'tasks.aggregate_learning_metrics_task': {'queue': 'background'},
    'tasks.run_llm_periodic_checkpoint_task': {'queue': 'heavy'},
    'tasks.process_memory_embedding_task': {'queue': 'heavy'},
    'tasks.retrieve_memories_task': {'queue': 'background'},
    'tasks.analyze_with_memory_task': {'queue': 'heavy'},

    # Low priority tasks routed to background/heavy queues
    'tasks.memory_maintenance_task': {'queue': 'heavy'},
    'tasks.memory_embedding_consolidation_task': {'queue': 'heavy'},
    'tasks.cleanup_old_performance_data_task': {'queue': 'background'},
    'tasks.cleanup_*': {'queue': 'background'},
    'tasks.analytics_*': {'queue': 'background'}
}

# Base configuration
base_config = {
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'worker_prefetch_multiplier': 1,  # Don't prefetch tasks
    'task_acks_late': True,
    'result_expires': 3600,  # 1 hour
    'worker_max_tasks_per_child': 50,  # Reduced from 1000 - restart workers more often to release connections
    'worker_pool': 'prefork',
    # IMPORTANT: Limit concurrency to prevent connection pool exhaustion
    'worker_concurrency': int(os.getenv('CELERY_CONCURRENCY', '2')),  # Keep low - was '1'
    'worker_disable_rate_limits': True,
    'worker_send_task_events': False,
    
    # Add these new settings:
    'task_time_limit': 900,  # 15 minutes hard limit
    'task_soft_time_limit': 840,  # 14 minutes soft limit
    'worker_max_memory_per_child': 1000000,  # 1GB - restart worker if it uses more
}

# Enhanced configuration for RabbitMQ
if USE_RABBITMQ:
    rabbitmq_task_queues = {
        queue_name: {
            'exchange': CELERY_QUEUE_EXCHANGE,
            'routing_key': queue_name,
            'queue_arguments': {'x-max-priority': priority},
        }
        for queue_name, priority in QUEUE_PRIORITIES.items()
    }

    rabbitmq_config = {
        # Queue settings
        'task_queues': rabbitmq_task_queues,
        
        # Task routing
        'task_routes': task_routes,
        
        # Dead letter configuration
        'task_reject_on_worker_lost': True,
        
        # Task retry settings
        'task_default_retry_delay': 60,  # 1 minute
        'task_max_retries': 3,
        
        # Monitoring settings (enable if needed)
        'worker_send_task_events': True,
        'task_send_sent_event': True
    }
    celery_app.conf.update({**base_config, **rabbitmq_config})
else:
    # Simpler configuration for Redis while ensuring task routes are configured
    redis_config = {**base_config, 'task_routes': task_routes}
    celery_app.conf.update(redis_config)


def _merge_task_queues():
    existing = celery_app.conf.get('task_queues')
    if isinstance(existing, dict):
        for queue in NYX_TASK_QUEUES:
            existing.setdefault(queue.name, {
                'exchange': queue.exchange.name if queue.exchange else None,
                'routing_key': queue.routing_key,
            })
        celery_app.conf.task_queues = existing
    else:
        queue_list = list(existing or [])
        existing_names = {getattr(q, 'name', None) for q in queue_list}
        queue_list.extend(
            queue for queue in NYX_TASK_QUEUES if getattr(queue, 'name', None) not in existing_names
        )
        if queue_list:
            celery_app.conf.task_queues = tuple(queue_list)


def _merge_task_routes():
    routes = celery_app.conf.get('task_routes') or {}
    if isinstance(routes, dict) and NYX_TASK_ROUTES:
        routes.update(NYX_TASK_ROUTES)
        celery_app.conf.task_routes = routes


if NYX_TASK_QUEUES:
    _merge_task_queues()
if NYX_TASK_ROUTES:
    _merge_task_routes()

# Comprehensive beat schedule
celery_app.conf.beat_schedule = {
    # --- NPC and Game Tasks ---
    'npc-learning-cycle-every-15-mins': {
        'task': 'tasks.run_npc_learning_cycle_task',
        'schedule': crontab(minute='*/15'),
    },

    'refresh-cultural-conflicts-every-30-mins': {
        'task': 'tasks.refresh_all_cultural_conflict_caches',
        'schedule': crontab(minute='*/30'),
        'options': {'queue': 'heavy'}
    },

    ## NEW ##: Add the periodic maintenance schedule for the conflict system
    'periodic-edge-case-maintenance': {
        'task': 'tasks.periodic_edge_case_maintenance',
        'schedule': crontab(minute='*/20'),  # Run every 20 minutes
        'options': {'queue': 'background'}
    },
    
    # --- Nyx Brain Tasks ---
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3:00 AM UTC
        'options': {'queue': 'heavy'}
    },
    "sweep-and-merge-nyx-split-brains-every-5min": {
        "task": "tasks.sweep_and_merge_nyx_split_brains",
        "schedule": crontab(minute="*/5"),
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
        'options': {'queue': 'heavy'}
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
        'options': {'queue': 'background'}
    },
}

if NYX_BEAT_SCHEDULE:
    celery_app.conf.beat_schedule.update(NYX_BEAT_SCHEDULE)

# =====================================================
# NOTE: Worker process initialization is handled by
# db.connection module via Celery signals.
# This ensures proper asyncpg pool management.
# =====================================================

# Import connection module to register signal handlers
try:
    import db.connection
    logger.info("Database connection module imported - worker lifecycle handlers registered")
except ImportError as e:
    logger.error(f"Failed to import db.connection module: {e}")
    logger.error("Worker processes may not initialize database pools correctly!")

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
        for queue_name in LIVE_QUEUE_NAMES:
            dead_letter_queues[queue_name] = Queue(
                f'dead-letter-{queue_name}',
                exchange=dead_letter_exchange,
                routing_key=f'dead-letter-{queue_name}'
            )

        # Update queue settings to use dead letter queues
        for queue_name, queue in celery_app.conf.task_queues.items():
            if queue_name in LIVE_QUEUE_NAMES:
                queue.setdefault('queue_arguments', {})
                queue['queue_arguments'].update({
                    'x-dead-letter-exchange': 'dead-letter',
                    'x-dead-letter-routing-key': f'dead-letter-{queue_name}'
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
        
        queue_names = list(LIVE_QUEUE_NAMES) if USE_RABBITMQ else ['celery']
        
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
            for queue_name in LIVE_QUEUE_NAMES:
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
