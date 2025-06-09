import os
import shutil
import pandas as pd
import json
from typing import List, Optional
from fastapi import UploadFile
from datetime import datetime

from app.models.dataset import Dataset, DatasetCreate, DatasetUpdate, DatasetStatus, DatasetMetadata
from app.core.config import settings
from app.services.storage_service import StorageService


class DatasetService:
    def __init__(self):
        self.storage_service = StorageService()

    async def create_dataset(self, dataset_data: DatasetCreate, file: UploadFile, owner_id: str) -> Dataset:
        """Create a new dataset with file upload"""
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(settings.datasets_dir, filename)
        
        # Save uploaded file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create dataset document
        dataset = Dataset(
            name=dataset_data.name,
            description=dataset_data.description,
            dataset_type=dataset_data.dataset_type,
            format=dataset_data.format,
            status=DatasetStatus.PROCESSING,
            file_path=file_path,
            original_filename=file.filename,
            file_size=file_size,
            tags=dataset_data.tags,
            owner_id=owner_id,
            is_public=dataset_data.is_public
        )
        
        await dataset.insert()
        
        # Process dataset asynchronously
        await self._process_dataset_async(dataset)
        
        return dataset

    async def _process_dataset_async(self, dataset: Dataset):
        """Process dataset to extract metadata"""
        try:
            metadata = await self._extract_metadata(dataset)
            dataset.metadata = metadata
            dataset.status = DatasetStatus.READY
            dataset.updated_at = datetime.utcnow()
            await dataset.save()
        except Exception as e:
            dataset.status = DatasetStatus.ERROR
            dataset.updated_at = datetime.utcnow()
            await dataset.save()
            raise e

    async def _extract_metadata(self, dataset: Dataset) -> DatasetMetadata:
        """Extract metadata from dataset file"""
        metadata = DatasetMetadata()
        
        try:
            if dataset.format.lower() == "csv":
                df = pd.read_csv(dataset.file_path)
                metadata.total_samples = len(df)
                metadata.columns = df.columns.tolist()
                metadata.data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                metadata.statistics = {
                    "numeric_columns": df.select_dtypes(include=['number']).describe().to_dict(),
                    "categorical_columns": {col: df[col].value_counts().head(10).to_dict() 
                                          for col in df.select_dtypes(include=['object']).columns}
                }
            
            elif dataset.format.lower() in ["json", "jsonl"]:
                with open(dataset.file_path, 'r') as f:
                    if dataset.format.lower() == "json":
                        data = json.load(f)
                        if isinstance(data, list):
                            metadata.total_samples = len(data)
                        else:
                            metadata.total_samples = 1
                    else:  # jsonl
                        lines = f.readlines()
                        metadata.total_samples = len(lines)
            
            elif dataset.dataset_type == "image":
                # For image datasets, count files in directory or extract from archive
                if os.path.isdir(dataset.file_path):
                    image_files = [f for f in os.listdir(dataset.file_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                    metadata.total_samples = len(image_files)
        
        except Exception as e:
            # If metadata extraction fails, set basic info
            metadata.total_samples = 0
            
        return metadata

    async def list_datasets(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100,
        dataset_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dataset]:
        """List datasets accessible to user"""
        
        # Build query
        query = {"$or": [{"owner_id": user_id}, {"is_public": True}]}
        
        if dataset_type:
            query["dataset_type"] = dataset_type
        if status:
            query["status"] = status
        
        datasets = await Dataset.find(query).skip(skip).limit(limit).to_list()
        return datasets

    async def get_dataset(self, dataset_id: str, user_id: str) -> Optional[Dataset]:
        """Get a specific dataset"""
        dataset = await Dataset.get(dataset_id)
        
        if not dataset:
            return None
        
        # Check access permissions
        if dataset.owner_id != user_id and not dataset.is_public:
            return None
        
        return dataset

    async def update_dataset(self, dataset_id: str, dataset_update: DatasetUpdate, user_id: str) -> Optional[Dataset]:
        """Update a dataset"""
        dataset = await Dataset.get(dataset_id)
        
        if not dataset or dataset.owner_id != user_id:
            return None
        
        # Update fields
        update_data = dataset_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(dataset, field, value)
        
        dataset.updated_at = datetime.utcnow()
        await dataset.save()
        
        return dataset

    async def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        """Delete a dataset"""
        dataset = await Dataset.get(dataset_id)
        
        if not dataset or dataset.owner_id != user_id:
            return False
        
        # Delete file
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete document
        await dataset.delete()
        
        return True

    async def process_dataset(self, dataset_id: str, user_id: str) -> bool:
        """Trigger dataset processing"""
        dataset = await Dataset.get(dataset_id)
        
        if not dataset or dataset.owner_id != user_id:
            return False
        
        dataset.status = DatasetStatus.PROCESSING
        await dataset.save()
        
        # Process asynchronously
        await self._process_dataset_async(dataset)
        
        return True
