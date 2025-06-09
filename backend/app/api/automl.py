from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.models.user import User
from app.models.automl_job import (
    AutoMLJob, AutoMLJobCreate, AutoMLJobUpdate, 
    AutoMLJobResponse, AutoMLStatus, AutoMLRecommendation,
    AutoMLSummary
)
from app.api.auth import get_current_active_user
from app.services.automl_service import AutoMLService

router = APIRouter()


@router.post("/", response_model=AutoMLJobResponse)
async def create_automl_job(
    job_data: AutoMLJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new AutoML job"""
    automl_service = AutoMLService()
    job = await automl_service.create_automl_job(job_data.config, str(current_user.id))
    
    # Start AutoML in background
    background_tasks.add_task(automl_service.start_automl_job, str(job.id))
    
    return AutoMLJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        status=job.status,
        progress=job.progress,
        trial_results=job.trial_results,
        best_parameters=job.best_parameters,
        best_score=job.best_score,
        best_trial_id=job.best_trial_id,
        total_trials_completed=job.total_trials_completed,
        owner_id=job.owner_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.get("/", response_model=List[AutoMLJobResponse])
async def list_automl_jobs(
    current_user: User = Depends(get_current_active_user)
):
    """List all AutoML jobs for the current user"""
    automl_service = AutoMLService()
    jobs = await automl_service.list_automl_jobs(str(current_user.id))
    
    return [
        AutoMLJobResponse(
            id=str(job.id),
            name=job.name,
            description=job.description,
            config=job.config,
            status=job.status,
            progress=job.progress,
            trial_results=job.trial_results,
            best_parameters=job.best_parameters,
            best_score=job.best_score,
            best_trial_id=job.best_trial_id,
            total_trials_completed=job.total_trials_completed,
            owner_id=job.owner_id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=AutoMLJobResponse)
async def get_automl_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific AutoML job"""
    automl_service = AutoMLService()
    job = await automl_service.get_automl_job(job_id, str(current_user.id))
    
    if not job:
        raise HTTPException(status_code=404, detail="AutoML job not found")
    
    return AutoMLJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        status=job.status,
        progress=job.progress,
        trial_results=job.trial_results,
        best_parameters=job.best_parameters,
        best_score=job.best_score,
        best_trial_id=job.best_trial_id,
        total_trials_completed=job.total_trials_completed,
        owner_id=job.owner_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.put("/{job_id}", response_model=AutoMLJobResponse)
async def update_automl_job(
    job_id: str,
    job_update: AutoMLJobUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update an AutoML job"""
    automl_service = AutoMLService()
    job = await automl_service.get_automl_job(job_id, str(current_user.id))
    
    if not job:
        raise HTTPException(status_code=404, detail="AutoML job not found")
    
    # Update fields
    if job_update.name is not None:
        job.name = job_update.name
    if job_update.description is not None:
        job.description = job_update.description
    if job_update.status is not None:
        job.status = job_update.status
    
    await job.save()
    
    return AutoMLJobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        config=job.config,
        status=job.status,
        progress=job.progress,
        trial_results=job.trial_results,
        best_parameters=job.best_parameters,
        best_score=job.best_score,
        best_trial_id=job.best_trial_id,
        total_trials_completed=job.total_trials_completed,
        owner_id=job.owner_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.delete("/{job_id}")
async def delete_automl_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete an AutoML job"""
    automl_service = AutoMLService()
    job = await automl_service.get_automl_job(job_id, str(current_user.id))
    
    if not job:
        raise HTTPException(status_code=404, detail="AutoML job not found")
    
    await job.delete()
    return {"message": "AutoML job deleted successfully"}


@router.post("/{job_id}/start")
async def start_automl_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Start an AutoML job"""
    automl_service = AutoMLService()
    success = await automl_service.start_automl_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="AutoML job not found or cannot be started")
    
    return {"message": "AutoML job started"}


@router.post("/{job_id}/stop")
async def stop_automl_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Stop a running AutoML job"""
    automl_service = AutoMLService()
    success = await automl_service.stop_automl_job(job_id, str(current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="AutoML job not found or cannot be stopped")
    
    return {"message": "AutoML job stopped"}


@router.get("/{job_id}/summary", response_model=AutoMLSummary)
async def get_automl_summary(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get AutoML job summary with insights and recommendations"""
    automl_service = AutoMLService()
    job = await automl_service.get_automl_job(job_id, str(current_user.id))
    
    if not job:
        raise HTTPException(status_code=404, detail="AutoML job not found")
    
    # Calculate summary statistics
    successful_trials = len([t for t in job.trial_results if t.status == "completed"])
    total_time = None
    if job.started_at and job.completed_at:
        total_time = (job.completed_at - job.started_at).total_seconds() / 60
    
    # Generate insights and recommendations
    insights = []
    recommendations = []
    
    if job.trial_results:
        best_trial = max(job.trial_results, key=lambda x: x.score)
        insights.append(f"Best trial achieved {best_trial.score:.4f} {job.config.optimization_metric}")
        
        if len(job.trial_results) > 1:
            scores = [t.score for t in job.trial_results]
            avg_score = sum(scores) / len(scores)
            insights.append(f"Average score across all trials: {avg_score:.4f}")
            
        if successful_trials < len(job.trial_results):
            failed_trials = len(job.trial_results) - successful_trials
            insights.append(f"{failed_trials} trials failed during execution")
            recommendations.append("Consider adjusting hyperparameter ranges to avoid failed trials")
    
    if job.status == AutoMLStatus.COMPLETED:
        recommendations.append("Use the best parameters for production training")
        recommendations.append("Consider running additional trials with refined search space")
    
    return AutoMLSummary(
        job_id=str(job.id),
        job_name=job.name,
        status=job.status,
        best_score=job.best_score,
        improvement_over_baseline=None,  # Would need baseline comparison
        total_trials=len(job.trial_results),
        successful_trials=successful_trials,
        total_time_minutes=total_time,
        best_parameters=job.best_parameters,
        model_insights=insights,
        recommendations=recommendations
    )


@router.post("/recommendations", response_model=AutoMLRecommendation)
async def get_automl_recommendations(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get AutoML configuration recommendations based on dataset"""
    # This would analyze the dataset and provide recommendations
    # For now, return default recommendations
    
    from app.models.training_job import ModelType
    from app.models.automl_job import OptimizationType
    
    return AutoMLRecommendation(
        recommended_model_type=ModelType.CV_CLASSIFICATION,
        recommended_optimization_type=OptimizationType.HYPERPARAMETER,
        recommended_max_trials=20,
        recommended_search_space={
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32, 64],
            "num_epochs": [10, 20, 30]
        },
        reasoning="Based on dataset characteristics, hyperparameter optimization is recommended",
        estimated_time_minutes=60
    )
