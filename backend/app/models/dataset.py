from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DatasetType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"


class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DatasetFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    IMAGES = "images"
    AUDIO_FILES = "audio_files"


class DatasetMetadata(BaseModel):
    total_samples: Optional[int] = None
    file_size: Optional[int] = None
    columns: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None
    statistics: Optional[Dict[str, Any]] = None


class Dataset(Document):
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType
    format: DatasetFormat
    status: DatasetStatus = DatasetStatus.UPLOADING
    
    # File information
    file_path: str
    original_filename: str
    file_size: int = 0
    
    # Metadata
    metadata: Optional[DatasetMetadata] = None
    tags: List[str] = []
    
    # Ownership and access
    owner_id: str
    is_public: bool = False
    
    # Versioning
    version: str = "1.0.0"
    parent_dataset_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    
    class Settings:
        collection = "datasets"
        indexes = [
            "name",
            "owner_id",
            "dataset_type",
            "status",
            "created_at",
        ]


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType
    format: DatasetFormat
    tags: List[str] = []
    is_public: bool = False


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType
    format: DatasetFormat
    status: DatasetStatus
    file_size: int
    metadata: Optional[DatasetMetadata] = None
    tags: List[str]
    owner_id: str
    is_public: bool
    version: str
    created_at: datetime
    updated_at: datetime
