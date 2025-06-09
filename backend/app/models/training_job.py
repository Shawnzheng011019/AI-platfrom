from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    LLM = "llm"
    DIFFUSION = "diffusion"
    NLP_CLASSIFICATION = "nlp_classification"
    NLP_NER = "nlp_ner"
    CV_CLASSIFICATION = "cv_classification"
    CV_DETECTION = "cv_detection"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SPEECH_RECOGNITION = "speech_recognition"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    # Model configuration
    model_name: str
    model_type: ModelType
    base_model: Optional[str] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: float = 0.01
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Hardware configuration
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Additional parameters
    custom_params: Dict[str, Any] = {}


class TrainingMetrics(BaseModel):
    epoch: int
    step: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = {}
    timestamp: datetime = datetime.utcnow()


class TrainingJob(Document):
    name: str
    description: Optional[str] = None
    
    # Configuration
    config: TrainingConfig
    
    # Data
    dataset_id: str
    
    # Status and progress
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0  # 0-100
    current_epoch: int = 0
    current_step: int = 0
    
    # Results
    metrics: List[TrainingMetrics] = []
    final_metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    
    # Logs and errors
    logs: List[str] = []
    error_message: Optional[str] = None
    
    # Resource usage
    gpu_memory_used: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Ownership and scheduling
    owner_id: str
    priority: int = 0
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Settings:
        collection = "training_jobs"
        indexes = [
            "name",
            "owner_id",
            "status",
            "created_at",
            "priority",
        ]


class TrainingJobCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: TrainingConfig
    dataset_id: str
    priority: int = 0


class TrainingJobUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None


class TrainingJobResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    config: TrainingConfig
    dataset_id: str
    status: TrainingStatus
    progress: float
    current_epoch: int
    current_step: int
    final_metrics: Optional[Dict[str, float]] = None
    owner_id: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
