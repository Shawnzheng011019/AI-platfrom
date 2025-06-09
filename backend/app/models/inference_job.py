from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class InferenceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceRequest(BaseModel):
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class InferenceResponse(BaseModel):
    request_id: str
    model_id: str
    result: Optional[Dict[str, Any]] = None
    inference_time_seconds: float
    status: str
    error_message: Optional[str] = None
    timestamp: datetime


class BatchInferenceRequest(BaseModel):
    batch_id: str
    model_id: str
    inputs: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class BatchInferenceResponse(BaseModel):
    batch_id: str
    model_id: str
    results: List[InferenceResponse]
    total_inference_time_seconds: float
    status: str
    completed_requests: int
    failed_requests: int
    timestamp: datetime


class InferenceJob(Document):
    """Represents a single inference job"""
    request_id: str
    model_id: str
    
    # Input and output
    input_data: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    # Status and timing
    status: InferenceStatus = InferenceStatus.PENDING
    inference_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Metadata
    user_id: Optional[str] = None
    priority: int = 0
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Settings:
        collection = "inference_jobs"
        indexes = [
            "request_id",
            "model_id",
            "user_id",
            "status",
            "created_at",
        ]


class BatchInferenceJob(Document):
    """Represents a batch inference job"""
    batch_id: str
    model_id: str
    
    # Input and output
    inputs: List[Dict[str, Any]]
    results: List[Dict[str, Any]] = []
    parameters: Optional[Dict[str, Any]] = None
    
    # Status and progress
    status: InferenceStatus = InferenceStatus.PENDING
    progress: float = 0.0  # 0-100
    completed_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    
    # Timing
    total_inference_time_seconds: Optional[float] = None
    average_inference_time_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    failed_request_ids: List[str] = []
    
    # Metadata
    user_id: Optional[str] = None
    priority: int = 0
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Settings:
        collection = "batch_inference_jobs"
        indexes = [
            "batch_id",
            "model_id",
            "user_id",
            "status",
            "created_at",
        ]


class ModelEndpoint(Document):
    """Represents a deployed model endpoint"""
    endpoint_id: str
    model_id: str
    name: str
    description: Optional[str] = None
    
    # Endpoint configuration
    endpoint_url: str
    api_key: Optional[str] = None
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    
    # Load balancing and scaling
    min_instances: int = 1
    max_instances: int = 5
    auto_scaling_enabled: bool = True
    target_cpu_utilization: int = 70
    
    # Status and health
    status: str = "active"  # active, inactive, error
    health_check_url: Optional[str] = None
    last_health_check: Optional[datetime] = None
    
    # Metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: Optional[float] = None
    
    # Ownership
    owner_id: str
    is_public: bool = False
    
    # Timestamps
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    deployed_at: Optional[datetime] = None
    
    class Settings:
        collection = "model_endpoints"
        indexes = [
            "endpoint_id",
            "model_id",
            "owner_id",
            "status",
            "created_at",
        ]


class InferenceMetrics(BaseModel):
    """Metrics for inference performance"""
    model_id: str
    endpoint_id: Optional[str] = None
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing metrics
    average_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    requests_per_minute: float = 0.0
    
    # Resource metrics
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    
    # Error metrics
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0
    
    # Time window
    start_time: datetime
    end_time: datetime


class InferenceJobCreate(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 0


class BatchInferenceJobCreate(BaseModel):
    model_id: str
    inputs: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 0


class ModelEndpointCreate(BaseModel):
    model_id: str
    name: str
    description: Optional[str] = None
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    min_instances: int = 1
    max_instances: int = 5
    auto_scaling_enabled: bool = True
    is_public: bool = False


class ModelEndpointUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    max_concurrent_requests: Optional[int] = None
    timeout_seconds: Optional[int] = None
    min_instances: Optional[int] = None
    max_instances: Optional[int] = None
    auto_scaling_enabled: Optional[bool] = None
    is_public: Optional[bool] = None
    status: Optional[str] = None


class InferenceJobResponse(BaseModel):
    id: str
    request_id: str
    model_id: str
    status: InferenceStatus
    result: Optional[Dict[str, Any]] = None
    inference_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class BatchInferenceJobResponse(BaseModel):
    id: str
    batch_id: str
    model_id: str
    status: InferenceStatus
    progress: float
    completed_requests: int
    failed_requests: int
    total_requests: int
    total_inference_time_seconds: Optional[float] = None
    user_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ModelEndpointResponse(BaseModel):
    id: str
    endpoint_id: str
    model_id: str
    name: str
    description: Optional[str] = None
    endpoint_url: str
    status: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: Optional[float] = None
    owner_id: str
    is_public: bool
    created_at: datetime
    deployed_at: Optional[datetime] = None
