from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime

from app.models.user import User
from app.models.dataset import Dataset, DatasetCreate, DatasetUpdate, DatasetResponse, DatasetStatus
from app.api.auth import get_current_active_user
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    dataset_type: str = Form(...),
    format: str = Form(...),
    tags: str = Form(""),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_active_user)
):
    """Upload and create a new dataset"""
    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    
    dataset_data = DatasetCreate(
        name=name,
        description=description,
        dataset_type=dataset_type,
        format=format,
        tags=tags_list,
        is_public=is_public
    )
    
    dataset_service = DatasetService()
    dataset = await dataset_service.create_dataset(dataset_data, file, str(current_user.id))
    
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        format=dataset.format,
        status=dataset.status,
        file_size=dataset.file_size,
        metadata=dataset.metadata,
        tags=dataset.tags,
        owner_id=dataset.owner_id,
        is_public=dataset.is_public,
        version=dataset.version,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    dataset_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """List datasets accessible to the current user"""
    dataset_service = DatasetService()
    datasets = await dataset_service.list_datasets(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        dataset_type=dataset_type,
        status=status
    )
    
    return [
        DatasetResponse(
            id=str(dataset.id),
            name=dataset.name,
            description=dataset.description,
            dataset_type=dataset.dataset_type,
            format=dataset.format,
            status=dataset.status,
            file_size=dataset.file_size,
            metadata=dataset.metadata,
            tags=dataset.tags,
            owner_id=dataset.owner_id,
            is_public=dataset.is_public,
            version=dataset.version,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at
        )
        for dataset in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific dataset"""
    dataset_service = DatasetService()
    dataset = await dataset_service.get_dataset(dataset_id, str(current_user.id))
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        format=dataset.format,
        status=dataset.status,
        file_size=dataset.file_size,
        metadata=dataset.metadata,
        tags=dataset.tags,
        owner_id=dataset.owner_id,
        is_public=dataset.is_public,
        version=dataset.version,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    dataset_update: DatasetUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update a dataset"""
    dataset_service = DatasetService()
    dataset = await dataset_service.update_dataset(dataset_id, dataset_update, str(current_user.id))
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        format=dataset.format,
        status=dataset.status,
        file_size=dataset.file_size,
        metadata=dataset.metadata,
        tags=dataset.tags,
        owner_id=dataset.owner_id,
        is_public=dataset.is_public,
        version=dataset.version,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a dataset"""
    dataset_service = DatasetService()
    success = await dataset_service.delete_dataset(dataset_id, str(current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/process")
async def process_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Trigger dataset processing"""
    dataset_service = DatasetService()
    success = await dataset_service.process_dataset(dataset_id, str(current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"message": "Dataset processing started"}
