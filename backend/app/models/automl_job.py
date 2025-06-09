from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from .training_job import ModelType


class AutoMLStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationType(str, Enum):
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    FEATURE_ENGINEERING = "feature_engineering"
    FULL_PIPELINE = "full_pipeline"


class AutoMLConfig(BaseModel):
    name: str
    description: Optional[str] = None
    
    # Model configuration
    model_type: ModelType
    base_model: Optional[str] = None
    dataset_id: str
    
    # Optimization configuration
    optimization_type: OptimizationType = OptimizationType.HYPERPARAMETER
    optimization_metric: str = "accuracy"  # accuracy, f1_score, precision, recall, loss
    optimization_direction: str = "maximize"  # maximize or minimize
    
    # Search configuration
    max_trials: int = 20
    max_time_minutes: int = 120
    early_stopping_patience: int = 5
    
    # Resource constraints
    max_concurrent_trials: int = 2
    use_gpu: bool = True
    
    # Search space (optional - will use defaults if not provided)
    search_space: Optional[Dict[str, Any]] = None
    
    # Advanced options
    cross_validation_folds: int = 3
    test_split: float = 0.2
    random_seed: int = 42


class TrialResult(BaseModel):
    trial_id: int
    parameters: Dict[str, Any]
    score: float
    training_job_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    metrics: Optional[Dict[str, float]] = None


class AutoMLJob(Document):
    name: str
    description: Optional[str] = None
    
    # Configuration
    config: AutoMLConfig
    
    # Status and progress
    status: AutoMLStatus = AutoMLStatus.PENDING
    progress: float = 0.0  # 0-100
    
    # Results
    trial_results: List[TrialResult] = []
    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_trial_id: Optional[int] = None
    
    # Model artifacts
    best_model_path: Optional[str] = None
    
    # Logs and errors
    logs: List[str] = []
    error_message: Optional[str] = None
    
    # Resource usage
    total_compute_time_minutes: Optional[float] = None
    total_trials_completed: int = 0
    
    # Ownership
    owner_id: str
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Settings:
        collection = "automl_jobs"
        indexes = [
            "name",
            "owner_id",
            "status",
            "created_at",
        ]


class AutoMLJobCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: AutoMLConfig


class AutoMLJobUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AutoMLStatus] = None


class AutoMLJobResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    config: AutoMLConfig
    status: AutoMLStatus
    progress: float
    trial_results: List[TrialResult]
    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_trial_id: Optional[int] = None
    total_trials_completed: int
    owner_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AutoMLRecommendation(BaseModel):
    """Recommendations for AutoML configuration"""
    recommended_model_type: ModelType
    recommended_optimization_type: OptimizationType
    recommended_max_trials: int
    recommended_search_space: Dict[str, Any]
    reasoning: str
    estimated_time_minutes: int


class AutoMLSummary(BaseModel):
    """Summary of AutoML results"""
    job_id: str
    job_name: str
    status: AutoMLStatus
    best_score: Optional[float]
    improvement_over_baseline: Optional[float]
    total_trials: int
    successful_trials: int
    total_time_minutes: Optional[float]
    best_parameters: Optional[Dict[str, Any]]
    model_insights: List[str]
    recommendations: List[str]
