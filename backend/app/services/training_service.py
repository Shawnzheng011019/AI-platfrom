import os
import json
import asyncio
import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.training_job import (
    TrainingJob, TrainingJobCreate, TrainingJobUpdate, 
    TrainingStatus, TrainingMetrics
)
from app.models.dataset import Dataset
from app.models.model import Model, ModelStatus
from app.core.config import settings
from app.services.celery_service import CeleryService


class TrainingService:
    def __init__(self):
        self.celery_service = CeleryService()

    async def create_training_job(self, job_data: TrainingJobCreate, owner_id: str) -> TrainingJob:
        """Create a new training job"""
        
        # Verify dataset exists and user has access
        dataset = await Dataset.get(job_data.dataset_id)
        if not dataset or (dataset.owner_id != owner_id and not dataset.is_public):
            raise ValueError("Dataset not found or access denied")
        
        # Create training job
        job = TrainingJob(
            name=job_data.name,
            description=job_data.description,
            config=job_data.config,
            dataset_id=job_data.dataset_id,
            owner_id=owner_id,
            priority=job_data.priority
        )
        
        await job.insert()
        return job

    async def list_training_jobs(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[TrainingJob]:
        """List training jobs for a user"""
        
        query = {"owner_id": user_id}
        if status:
            query["status"] = status
        
        jobs = await TrainingJob.find(query).skip(skip).limit(limit).to_list()
        return jobs

    async def get_training_job(self, job_id: str, user_id: str) -> Optional[TrainingJob]:
        """Get a specific training job"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return None
        
        return job

    async def update_training_job(
        self, 
        job_id: str, 
        job_update: TrainingJobUpdate, 
        user_id: str
    ) -> Optional[TrainingJob]:
        """Update a training job"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return None
        
        # Only allow updates if job is not running
        if job.status in [TrainingStatus.RUNNING]:
            raise ValueError("Cannot update running training job")
        
        update_data = job_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(job, field, value)
        
        await job.save()
        return job

    async def delete_training_job(self, job_id: str, user_id: str) -> bool:
        """Delete a training job"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return False
        
        # Stop job if running
        if job.status == TrainingStatus.RUNNING:
            await self.stop_training_job(job_id, user_id)
        
        # Delete job files
        if job.model_path and os.path.exists(job.model_path):
            import shutil
            shutil.rmtree(job.model_path, ignore_errors=True)
        
        await job.delete()
        return True

    async def queue_training_job(self, job_id: str, user_id: str) -> bool:
        """Queue a training job for execution"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return False
        
        if job.status != TrainingStatus.PENDING:
            raise ValueError("Job is not in pending status")
        
        job.status = TrainingStatus.QUEUED
        await job.save()
        
        return True

    async def start_training(self, job_id: str):
        """Start training job execution"""
        job = await TrainingJob.get(job_id)
        
        if not job:
            return
        
        try:
            # Update status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            await job.save()
            
            # Get dataset
            dataset = await Dataset.get(job.dataset_id)
            if not dataset:
                raise ValueError("Dataset not found")
            
            # Create output directory
            output_dir = os.path.join(settings.models_dir, str(job.id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare training config
            config = job.config.dict()
            config.update({
                'dataset_path': dataset.file_path,
                'output_dir': output_dir,
                'job_id': str(job.id)
            })
            
            # Save config file
            config_path = os.path.join(output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Select training script based on model type
            script_path = self._get_training_script(job.config.model_type)
            
            # Start training process
            await self._execute_training(script_path, config_path, job)
            
        except Exception as e:
            # Update job status on error
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await job.save()

    async def stop_training_job(self, job_id: str, user_id: str) -> bool:
        """Stop a running training job"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return False
        
        if job.status != TrainingStatus.RUNNING:
            return False
        
        # TODO: Implement process termination
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        await job.save()
        
        return True

    async def get_training_logs(self, job_id: str, user_id: str) -> Optional[List[str]]:
        """Get training job logs"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return None
        
        return job.logs

    async def get_training_metrics(self, job_id: str, user_id: str) -> Optional[List[TrainingMetrics]]:
        """Get training job metrics"""
        job = await TrainingJob.get(job_id)
        
        if not job or job.owner_id != user_id:
            return None
        
        return job.metrics

    def _get_training_script(self, model_type: str) -> str:
        """Get training script path based on model type"""
        script_mapping = {
            'llm': 'training-scripts/llm_fine_tuning.py',
            'diffusion': 'training-scripts/diffusion_training.py',
            'nlp_classification': 'training-scripts/nlp_classification.py',
            'nlp_ner': 'training-scripts/nlp_ner.py',
            'cv_classification': 'training-scripts/image_classification.py',
            'cv_detection': 'training-scripts/object_detection.py',
            'time_series': 'training-scripts/time_series_forecasting.py',
            'recommendation': 'training-scripts/recommendation_system.py',
            'reinforcement_learning': 'training-scripts/reinforcement_learning.py',
            'speech_recognition': 'training-scripts/speech_recognition.py',
            'multimodal': 'training-scripts/multimodal_training.py',
        }
        
        script_path = script_mapping.get(model_type.lower())
        if not script_path or not os.path.exists(script_path):
            raise ValueError(f"Training script not found for model type: {model_type}")
        
        return script_path

    async def _execute_training(self, script_path: str, config_path: str, job: TrainingJob):
        """Execute training script"""
        
        # Build command
        cmd = [
            'python', script_path,
            '--config', config_path
        ]
        
        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd()
        )
        
        # Monitor process output
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            log_line = line.decode().strip()
            job.logs.append(log_line)
            
            # Parse metrics from log line if needed
            # TODO: Implement metric parsing
            
            # Save progress periodically
            if len(job.logs) % 10 == 0:
                await job.save()
        
        # Wait for process completion
        await process.wait()
        
        # Update final status
        if process.returncode == 0:
            job.status = TrainingStatus.COMPLETED
            job.progress = 100.0
            
            # Create model record
            await self._create_model_from_job(job)
        else:
            job.status = TrainingStatus.FAILED
            job.error_message = "Training process failed"
        
        job.completed_at = datetime.utcnow()
        await job.save()

    async def _create_model_from_job(self, job: TrainingJob):
        """Create a model record from completed training job"""
        
        model = Model(
            name=f"{job.name}_model",
            description=f"Model trained from job: {job.name}",
            model_type=job.config.model_type,
            base_model=job.config.base_model,
            framework="pytorch",  # Default framework
            training_job_id=str(job.id),
            dataset_id=job.dataset_id,
            status=ModelStatus.READY,
            owner_id=job.owner_id
        )
        
        # Set model path
        if job.model_path:
            model.artifacts = {
                "model_path": job.model_path
            }
            
            # Get model size
            if os.path.exists(job.model_path):
                model.model_size = self._get_directory_size(job.model_path)
        
        # Set final metrics
        if job.final_metrics:
            model.metrics = job.final_metrics
        
        await model.insert()

    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
