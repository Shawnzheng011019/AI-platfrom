from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from .training_job import ModelType


class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    ERROR = "error"


class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    custom_metrics: Dict[str, float] = {}


class ModelArtifacts(BaseModel):
    model_path: str
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    weights_path: Optional[str] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None


class Model(Document):
    name: str
    description: Optional[str] = None
    
    # Model information
    model_type: ModelType
    base_model: Optional[str] = None
    framework: str  # pytorch, tensorflow, etc.
    
    # Training information
    training_job_id: Optional[str] = None
    dataset_id: Optional[str] = None
    
    # Status and version
    status: ModelStatus = ModelStatus.TRAINING
    version: str = "1.0.0"
    
    # Performance metrics
    metrics: Optional[ModelMetrics] = None
    
    # Model artifacts
    artifacts: Optional[ModelArtifacts] = None
    model_size: Optional[int] = None  # in bytes
    
    # Deployment information
    deployment_url: Optional[str] = None
    deployment_config: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = []
    parameters_count: Optional[int] = None
    
    # Ownership
    owner_id: str
    is_public: bool = False
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    
    class Settings:
        collection = "models"
        indexes = [
            "name",
            "owner_id",
            "model_type",
            "status",
            "created_at",
        ]


class ModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: ModelType
    base_model: Optional[str] = None
    framework: str
    tags: List[str] = []
    is_public: bool = False


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    status: Optional[ModelStatus] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    model_type: ModelType
    base_model: Optional[str] = None
    framework: str
    status: ModelStatus
    version: str
    metrics: Optional[ModelMetrics] = None
    model_size: Optional[int] = None
    tags: List[str]
    owner_id: str
    is_public: bool
    parameters_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
