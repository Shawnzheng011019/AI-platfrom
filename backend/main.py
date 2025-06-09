from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.api import auth, datasets, training_jobs, models, monitoring


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up AI Training Platform...")
    await connect_to_mongo()
    logger.info("Database connected successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Training Platform...")
    await close_mongo_connection()
    logger.info("Database disconnected")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI Model Training Platform API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=f"{settings.api_v1_prefix}/auth", tags=["authentication"])
app.include_router(datasets.router, prefix=f"{settings.api_v1_prefix}/datasets", tags=["datasets"])
app.include_router(training_jobs.router, prefix=f"{settings.api_v1_prefix}/training-jobs", tags=["training"])
app.include_router(models.router, prefix=f"{settings.api_v1_prefix}/models", tags=["models"])
app.include_router(monitoring.router, prefix=f"{settings.api_v1_prefix}/monitoring", tags=["monitoring"])

# Import and include AutoML router
from app.api import automl
app.include_router(automl.router, prefix=f"{settings.api_v1_prefix}/automl", tags=["automl"])

# Import and include Inference router
from app.api import inference
app.include_router(inference.router, prefix=f"{settings.api_v1_prefix}/inference", tags=["inference"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Training Platform API",
        "version": settings.version,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    from datetime import datetime, timezone
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.version
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
