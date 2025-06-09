from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from app.models.user import User
from app.models.inference_job import (
    InferenceJob, BatchInferenceJob, ModelEndpoint,
    InferenceJobCreate, BatchInferenceJobCreate, ModelEndpointCreate, ModelEndpointUpdate,
    InferenceJobResponse, BatchInferenceJobResponse, ModelEndpointResponse,
    InferenceRequest, InferenceResponse, InferenceStatus
)
from app.api.auth import get_current_active_user
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post("/single", response_model=InferenceResponse)
async def run_single_inference(
    job_data: InferenceJobCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Run a single inference request"""
    inference_service = InferenceService()
    
    request_id = str(uuid.uuid4())
    
    # Create inference request
    request = InferenceRequest(
        request_id=request_id,
        model_id=job_data.model_id,
        input_data=job_data.input_data,
        parameters=job_data.parameters,
        user_id=str(current_user.id)
    )
    
    # Run inference
    response = await inference_service.run_inference(request)
    
    # Save inference job to database
    inference_job = InferenceJob(
        request_id=request_id,
        model_id=job_data.model_id,
        input_data=job_data.input_data,
        result=response.result,
        parameters=job_data.parameters,
        status=InferenceStatus.COMPLETED if response.status == "success" else InferenceStatus.FAILED,
        inference_time_seconds=response.inference_time_seconds,
        error_message=response.error_message,
        user_id=str(current_user.id),
        priority=job_data.priority,
        completed_at=response.timestamp
    )
    
    await inference_job.insert()
    
    return response


@router.post("/batch", response_model=BatchInferenceJobResponse)
async def run_batch_inference(
    job_data: BatchInferenceJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Run batch inference"""
    batch_id = str(uuid.uuid4())
    
    # Create batch inference job
    batch_job = BatchInferenceJob(
        batch_id=batch_id,
        model_id=job_data.model_id,
        inputs=job_data.inputs,
        parameters=job_data.parameters,
        total_requests=len(job_data.inputs),
        user_id=str(current_user.id),
        priority=job_data.priority
    )
    
    await batch_job.insert()
    
    # Start batch processing in background
    background_tasks.add_task(
        _process_batch_inference,
        str(batch_job.id),
        job_data.model_id,
        job_data.inputs,
        job_data.parameters
    )
    
    return BatchInferenceJobResponse(
        id=str(batch_job.id),
        batch_id=batch_id,
        model_id=job_data.model_id,
        status=batch_job.status,
        progress=batch_job.progress,
        completed_requests=batch_job.completed_requests,
        failed_requests=batch_job.failed_requests,
        total_requests=batch_job.total_requests,
        user_id=str(current_user.id),
        created_at=batch_job.created_at
    )


async def _process_batch_inference(
    batch_job_id: str,
    model_id: str,
    inputs: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]]
):
    """Process batch inference in background"""
    inference_service = InferenceService()
    
    # Get batch job
    batch_job = await BatchInferenceJob.get(batch_job_id)
    if not batch_job:
        return
    
    batch_job.status = InferenceStatus.RUNNING
    batch_job.started_at = datetime.utcnow()
    await batch_job.save()
    
    results = []
    start_time = datetime.utcnow()
    
    try:
        for i, input_data in enumerate(inputs):
            request_id = f"{batch_job.batch_id}_{i}"
            
            # Create inference request
            request = InferenceRequest(
                request_id=request_id,
                model_id=model_id,
                input_data=input_data,
                parameters=parameters
            )
            
            # Run inference
            response = await inference_service.run_inference(request)
            results.append(response.dict())
            
            # Update progress
            if response.status == "success":
                batch_job.completed_requests += 1
            else:
                batch_job.failed_requests += 1
                batch_job.failed_request_ids.append(request_id)
            
            batch_job.progress = ((i + 1) / len(inputs)) * 100
            batch_job.results = results
            await batch_job.save()
        
        # Calculate final metrics
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        batch_job.status = InferenceStatus.COMPLETED
        batch_job.completed_at = end_time
        batch_job.total_inference_time_seconds = total_time
        batch_job.average_inference_time_seconds = total_time / len(inputs)
        await batch_job.save()
        
    except Exception as e:
        batch_job.status = InferenceStatus.FAILED
        batch_job.error_message = str(e)
        batch_job.completed_at = datetime.utcnow()
        await batch_job.save()


@router.get("/jobs", response_model=List[InferenceJobResponse])
async def list_inference_jobs(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """List inference jobs for the current user"""
    jobs = await InferenceJob.find(
        InferenceJob.user_id == str(current_user.id)
    ).limit(limit).to_list()
    
    return [
        InferenceJobResponse(
            id=str(job.id),
            request_id=job.request_id,
            model_id=job.model_id,
            status=job.status,
            result=job.result,
            inference_time_seconds=job.inference_time_seconds,
            error_message=job.error_message,
            user_id=job.user_id,
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        for job in jobs
    ]


@router.get("/batch-jobs", response_model=List[BatchInferenceJobResponse])
async def list_batch_inference_jobs(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """List batch inference jobs for the current user"""
    jobs = await BatchInferenceJob.find(
        BatchInferenceJob.user_id == str(current_user.id)
    ).limit(limit).to_list()
    
    return [
        BatchInferenceJobResponse(
            id=str(job.id),
            batch_id=job.batch_id,
            model_id=job.model_id,
            status=job.status,
            progress=job.progress,
            completed_requests=job.completed_requests,
            failed_requests=job.failed_requests,
            total_requests=job.total_requests,
            total_inference_time_seconds=job.total_inference_time_seconds,
            user_id=job.user_id,
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}", response_model=InferenceJobResponse)
async def get_inference_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific inference job"""
    job = await InferenceJob.get(job_id)
    
    if not job or job.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Inference job not found")
    
    return InferenceJobResponse(
        id=str(job.id),
        request_id=job.request_id,
        model_id=job.model_id,
        status=job.status,
        result=job.result,
        inference_time_seconds=job.inference_time_seconds,
        error_message=job.error_message,
        user_id=job.user_id,
        created_at=job.created_at,
        completed_at=job.completed_at
    )


@router.get("/batch-jobs/{job_id}", response_model=BatchInferenceJobResponse)
async def get_batch_inference_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific batch inference job"""
    job = await BatchInferenceJob.get(job_id)
    
    if not job or job.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Batch inference job not found")
    
    return BatchInferenceJobResponse(
        id=str(job.id),
        batch_id=job.batch_id,
        model_id=job.model_id,
        status=job.status,
        progress=job.progress,
        completed_requests=job.completed_requests,
        failed_requests=job.failed_requests,
        total_requests=job.total_requests,
        total_inference_time_seconds=job.total_inference_time_seconds,
        user_id=job.user_id,
        created_at=job.created_at,
        completed_at=job.completed_at
    )


@router.get("/models/loaded")
async def list_loaded_models(
    current_user: User = Depends(get_current_active_user)
):
    """List currently loaded models"""
    inference_service = InferenceService()
    return await inference_service.list_loaded_models()


@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get information about a model"""
    inference_service = InferenceService()
    try:
        return await inference_service.get_model_info(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Unload a model from memory"""
    inference_service = InferenceService()
    success = await inference_service.unload_model(model_id)
    
    if success:
        return {"message": f"Model {model_id} unloaded successfully"}
    else:
        return {"message": f"Model {model_id} was not loaded"}


# Model Endpoints Management
@router.post("/endpoints", response_model=ModelEndpointResponse)
async def create_model_endpoint(
    endpoint_data: ModelEndpointCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new model endpoint"""
    endpoint_id = str(uuid.uuid4())
    endpoint_url = f"/api/v1/inference/endpoints/{endpoint_id}/predict"
    
    endpoint = ModelEndpoint(
        endpoint_id=endpoint_id,
        model_id=endpoint_data.model_id,
        name=endpoint_data.name,
        description=endpoint_data.description,
        endpoint_url=endpoint_url,
        max_concurrent_requests=endpoint_data.max_concurrent_requests,
        timeout_seconds=endpoint_data.timeout_seconds,
        min_instances=endpoint_data.min_instances,
        max_instances=endpoint_data.max_instances,
        auto_scaling_enabled=endpoint_data.auto_scaling_enabled,
        is_public=endpoint_data.is_public,
        owner_id=str(current_user.id),
        deployed_at=datetime.utcnow()
    )
    
    await endpoint.insert()
    
    return ModelEndpointResponse(
        id=str(endpoint.id),
        endpoint_id=endpoint.endpoint_id,
        model_id=endpoint.model_id,
        name=endpoint.name,
        description=endpoint.description,
        endpoint_url=endpoint.endpoint_url,
        status=endpoint.status,
        total_requests=endpoint.total_requests,
        successful_requests=endpoint.successful_requests,
        failed_requests=endpoint.failed_requests,
        average_response_time_ms=endpoint.average_response_time_ms,
        owner_id=endpoint.owner_id,
        is_public=endpoint.is_public,
        created_at=endpoint.created_at,
        deployed_at=endpoint.deployed_at
    )
