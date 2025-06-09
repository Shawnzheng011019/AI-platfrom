from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application
    app_name: str = "AI Training Platform"
    version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "ai_platform"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_secure: bool = False
    
    # Storage
    upload_dir: str = "./uploads"
    models_dir: str = "./models"
    datasets_dir: str = "./datasets"
    
    # Training
    max_concurrent_trainings: int = 2
    training_timeout_hours: int = 24
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8001
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


# Create directories if they don't exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.models_dir, exist_ok=True)
os.makedirs(settings.datasets_dir, exist_ok=True)
