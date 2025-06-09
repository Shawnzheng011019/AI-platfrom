from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional
from datetime import datetime

from app.models.user import User
from app.models.model import Model, ModelCreate, ModelUpdate, ModelResponse, ModelStatus
from app.api.auth import get_current_active_user
from app.services.model_service import ModelService

router = APIRouter()


@router.post("/", response_model=ModelResponse)
async def create_model(
    model_data: ModelCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new model"""
    model_service = ModelService()
    model = await model_service.create_model(model_data, str(current_user.id))
    
    return ModelResponse(
        id=str(model.id),
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        base_model=model.base_model,
        framework=model.framework,
        status=model.status,
        version=model.version,
        metrics=model.metrics,
        model_size=model.model_size,
        tags=model.tags,
        owner_id=model.owner_id,
        is_public=model.is_public,
        parameters_count=model.parameters_count,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


@router.get("/", response_model=List[ModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """List models accessible to the current user"""
    model_service = ModelService()
    models = await model_service.list_models(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        model_type=model_type,
        status=status
    )
    
    return [
        ModelResponse(
            id=str(model.id),
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            base_model=model.base_model,
            framework=model.framework,
            status=model.status,
            version=model.version,
            metrics=model.metrics,
            model_size=model.model_size,
            tags=model.tags,
            owner_id=model.owner_id,
            is_public=model.is_public,
            parameters_count=model.parameters_count,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        for model in models
    ]


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific model"""
    model_service = ModelService()
    model = await model_service.get_model(model_id, str(current_user.id))
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelResponse(
        id=str(model.id),
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        base_model=model.base_model,
        framework=model.framework,
        status=model.status,
        version=model.version,
        metrics=model.metrics,
        model_size=model.model_size,
        tags=model.tags,
        owner_id=model.owner_id,
        is_public=model.is_public,
        parameters_count=model.parameters_count,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update a model"""
    model_service = ModelService()
    model = await model_service.update_model(model_id, model_update, str(current_user.id))
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelResponse(
        id=str(model.id),
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        base_model=model.base_model,
        framework=model.framework,
        status=model.status,
        version=model.version,
        metrics=model.metrics,
        model_size=model.model_size,
        tags=model.tags,
        owner_id=model.owner_id,
        is_public=model.is_public,
        parameters_count=model.parameters_count,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a model"""
    model_service = ModelService()
    success = await model_service.delete_model(model_id, str(current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": "Model deleted successfully"}


@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Deploy a model"""
    model_service = ModelService()
    deployment_url = await model_service.deploy_model(model_id, str(current_user.id))
    
    if not deployment_url:
        raise HTTPException(status_code=404, detail="Model not found or deployment failed")
    
    return {"message": "Model deployed successfully", "deployment_url": deployment_url}


@router.post("/{model_id}/undeploy")
async def undeploy_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Undeploy a model"""
    model_service = ModelService()
    success = await model_service.undeploy_model(model_id, str(current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": "Model undeployed successfully"}


@router.get("/{model_id}/download")
async def download_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get model download URL"""
    model_service = ModelService()
    download_url = await model_service.get_download_url(model_id, str(current_user.id))
    
    if not download_url:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"download_url": download_url}


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = None,
    description: str = None,
    model_type: str = None,
    framework: str = None,
    current_user: User = Depends(get_current_active_user)
):
    """Upload a pre-trained model"""
    model_service = ModelService()
    model = await model_service.upload_model(
        file=file,
        name=name or file.filename,
        description=description,
        model_type=model_type,
        framework=framework,
        owner_id=str(current_user.id)
    )
    
    return ModelResponse(
        id=str(model.id),
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        base_model=model.base_model,
        framework=model.framework,
        status=model.status,
        version=model.version,
        metrics=model.metrics,
        model_size=model.model_size,
        tags=model.tags,
        owner_id=model.owner_id,
        is_public=model.is_public,
        parameters_count=model.parameters_count,
        created_at=model.created_at,
        updated_at=model.updated_at
    )
