from celery import Celery
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class CeleryService:
    def __init__(self):
        self.celery_app = None
        self._initialize_celery()

    def _initialize_celery(self):
        """Initialize Celery application"""
        try:
            self.celery_app = Celery(
                'ai_platform',
                broker=settings.redis_url,
                backend=settings.redis_url,
                include=['app.tasks.training_tasks']
            )
            
            # Configure Celery
            self.celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                task_track_started=True,
                task_time_limit=settings.training_timeout_hours * 3600,
                worker_prefetch_multiplier=1,
                worker_max_tasks_per_child=1000,
            )
            
            logger.info("Celery initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Celery: {e}")
            self.celery_app = None

    def is_available(self) -> bool:
        """Check if Celery is available"""
        return self.celery_app is not None

    async def submit_training_task(self, job_id: str) -> str:
        """Submit a training task to Celery"""
        if not self.celery_app:
            raise RuntimeError("Celery not available")
        
        from app.tasks.training_tasks import run_training_job
        
        task = run_training_job.delay(job_id)
        return task.id

    async def get_task_status(self, task_id: str) -> dict:
        """Get task status"""
        if not self.celery_app:
            return {"status": "UNKNOWN", "message": "Celery not available"}
        
        task = self.celery_app.AsyncResult(task_id)
        return {
            "status": task.status,
            "result": task.result,
            "traceback": task.traceback
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if not self.celery_app:
            return False
        
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
