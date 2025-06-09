import os
import json
import asyncio
import subprocess
from celery import Celery
from datetime import datetime

from app.core.config import settings
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.dataset import Dataset
from app.models.model import Model, ModelStatus

# Initialize Celery
celery_app = Celery(
    'ai_platform',
    broker=settings.redis_url,
    backend=settings.redis_url
)


@celery_app.task(bind=True)
def run_training_job(self, job_id: str):
    """Celery task to run training job"""
    
    try:
        # This is a synchronous function, so we need to handle async operations differently
        # In a real implementation, you might want to use celery-aio or similar
        
        # For now, we'll use a simple approach
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(_run_training_job_async(job_id, self))
        loop.close()
        
        return result
        
    except Exception as e:
        # Update job status on error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_handle_training_error(job_id, str(e)))
        loop.close()
        
        raise e


async def _run_training_job_async(job_id: str, task):
    """Async function to run training job"""
    
    # Get training job
    job = await TrainingJob.get(job_id)
    if not job:
        raise ValueError(f"Training job {job_id} not found")
    
    # Update status
    job.status = TrainingStatus.RUNNING
    job.started_at = datetime.utcnow()
    await job.save()
    
    try:
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
        script_path = _get_training_script(job.config.model_type)
        
        # Execute training
        success = await _execute_training_script(script_path, config_path, job, task)
        
        if success:
            # Create model record
            await _create_model_from_job(job)
            
            job.status = TrainingStatus.COMPLETED
            job.progress = 100.0
        else:
            job.status = TrainingStatus.FAILED
            job.error_message = "Training script execution failed"
        
        job.completed_at = datetime.utcnow()
        await job.save()
        
        return {
            "status": job.status,
            "job_id": str(job.id),
            "completed_at": job.completed_at.isoformat()
        }
        
    except Exception as e:
        job.status = TrainingStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        await job.save()
        raise e


async def _execute_training_script(script_path: str, config_path: str, job: TrainingJob, task):
    """Execute training script"""
    
    # Build command
    cmd = [
        'python', script_path,
        '--config', config_path
    ]
    
    try:
        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd()
        )
        
        # Monitor process output
        log_lines = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            log_line = line.decode().strip()
            log_lines.append(log_line)
            
            # Update task progress
            if task:
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'current_step': len(log_lines),
                        'status': 'Training in progress...',
                        'logs': log_lines[-10:]  # Last 10 log lines
                    }
                )
            
            # Parse progress from log line
            progress = _parse_progress_from_log(log_line)
            if progress is not None:
                job.progress = progress
                await job.save()
        
        # Wait for process completion
        await process.wait()
        
        # Update job logs
        job.logs.extend(log_lines)
        await job.save()
        
        return process.returncode == 0
        
    except Exception as e:
        job.logs.append(f"Error executing training script: {str(e)}")
        await job.save()
        return False


def _get_training_script(model_type: str) -> str:
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


def _parse_progress_from_log(log_line: str) -> float:
    """Parse training progress from log line"""
    # Look for common progress patterns
    import re
    
    # Pattern: "Epoch 2/10" or "Step 150/1000"
    epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
    step_pattern = r'Step\s+(\d+)/(\d+)'
    
    epoch_match = re.search(epoch_pattern, log_line)
    if epoch_match:
        current, total = map(int, epoch_match.groups())
        return (current / total) * 100
    
    step_match = re.search(step_pattern, log_line)
    if step_match:
        current, total = map(int, step_match.groups())
        return (current / total) * 100
    
    # Pattern: "Progress: 75%"
    progress_pattern = r'Progress:\s*(\d+(?:\.\d+)?)%'
    progress_match = re.search(progress_pattern, log_line)
    if progress_match:
        return float(progress_match.group(1))
    
    return None


async def _create_model_from_job(job: TrainingJob):
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
    output_dir = os.path.join(settings.models_dir, str(job.id))
    if os.path.exists(output_dir):
        model.artifacts = {
            "model_path": output_dir
        }
        
        # Get model size
        model.model_size = _get_directory_size(output_dir)
    
    # Set final metrics
    if job.final_metrics:
        model.metrics = job.final_metrics
    
    await model.insert()
    
    # Update job with model path
    job.model_path = output_dir
    await job.save()


def _get_directory_size(path: str) -> int:
    """Get total size of directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass
    return total_size


async def _handle_training_error(job_id: str, error_message: str):
    """Handle training error"""
    job = await TrainingJob.get(job_id)
    if job:
        job.status = TrainingStatus.FAILED
        job.error_message = error_message
        job.completed_at = datetime.utcnow()
        await job.save()
