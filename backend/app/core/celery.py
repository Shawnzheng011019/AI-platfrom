from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    'ai_platform',
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=['app.tasks.training_tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.training_timeout_hours * 3600,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    task_routes={
        'app.tasks.training_tasks.run_training_job': {'queue': 'training'},
    },
    task_default_queue='default',
    task_create_missing_queues=True,
)
