from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.models.user import User
from app.models.training_job import (
    TrainingJob, TrainingJobCreate, TrainingJobUpdate, 
    TrainingJobResponse, TrainingStatus
)
from app.api.auth import get_current_active_user
from app.services.training_service import TrainingService

router = APIRouter()


@router.post("/", response_model=TrainingJobResponse)
async def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new training job"""
    training_service = TrainingService()
    job = await training_service.create_training_job(job_data, str(current_user.id))
    
    # Start training in background
    background_tasks.add_task(training_service.start_training, str(job.id))
    
    return TrainingJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        dataset_id=job.dataset_id,
        status=job.status,
        progress=job.progress,
        current_epoch=job.current_epoch,
        current_step=job.current_step,
        final_metrics=job.final_metrics,
        owner_id=job.owner_id,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.get("/", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """List training jobs for the current user"""
    training_service = TrainingService()
    jobs = await training_service.list_training_jobs(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        status=status
    )
    
    return [
        TrainingJobResponse(
            id=str(job.id),
            name=job.name,
            description=job.description,
            config=job.config,
            dataset_id=job.dataset_id,
            status=job.status,
            progress=job.progress,
            current_epoch=job.current_epoch,
            current_step=job.current_step,
            final_metrics=job.final_metrics,
            owner_id=job.owner_id,
            priority=job.priority,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific training job"""
    training_service = TrainingService()
    job = await training_service.get_training_job(job_id, str(current_user.id))
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return TrainingJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        dataset_id=job.dataset_id,
        status=job.status,
        progress=job.progress,
        current_epoch=job.current_epoch,
        current_step=job.current_step,
        final_metrics=job.final_metrics,
        owner_id=job.owner_id,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.put("/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: str,
    job_update: TrainingJobUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update a training job"""
    training_service = TrainingService()
    job = await training_service.update_training_job(job_id, job_update, str(current_user.id))

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        dataset_id=job.dataset_id,
        status=job.status,
        progress=job.progress,
        current_epoch=job.current_epoch,
        current_step=job.current_step,
        final_metrics=job.final_metrics,
        owner_id=job.owner_id,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.delete("/{job_id}")
async def delete_training_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a training job"""
    training_service = TrainingService()
    success = await training_service.delete_training_job(job_id, str(current_user.id))

    if not success:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {"message": "Training job deleted successfully"}


@router.post("/{job_id}/start")
async def start_training_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Start a training job"""
    training_service = TrainingService()
    success = await training_service.queue_training_job(job_id, str(current_user.id))

    if not success:
        raise HTTPException(status_code=404, detail="Training job not found")

    background_tasks.add_task(training_service.start_training, job_id)
    return {"message": "Training job started"}


@router.post("/{job_id}/stop")
async def stop_training_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Stop a training job"""
    training_service = TrainingService()
    success = await training_service.stop_training_job(job_id, str(current_user.id))

    if not success:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {"message": "Training job stopped"}


@router.get("/{job_id}/logs")
async def get_training_logs(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get training job logs"""
    training_service = TrainingService()
    logs = await training_service.get_training_logs(job_id, str(current_user.id))

    if logs is None:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {"logs": logs}


@router.get("/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get training job metrics"""
    training_service = TrainingService()
    metrics = await training_service.get_training_metrics(job_id, str(current_user.id))

    if metrics is None:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {"metrics": metrics}
