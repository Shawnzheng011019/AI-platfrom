import os
import shutil
from typing import List, Optional
from datetime import datetime
from fastapi import UploadFile

from app.models.model import Model, ModelCreate, ModelUpdate, ModelStatus
from app.core.config import settings
from app.services.storage_service import StorageService


class ModelService:
    def __init__(self):
        self.storage_service = StorageService()

    async def create_model(self, model_data: ModelCreate, owner_id: str) -> Model:
        """Create a new model"""
        
        model = Model(
            name=model_data.name,
            description=model_data.description,
            model_type=model_data.model_type,
            base_model=model_data.base_model,
            framework=model_data.framework,
            tags=model_data.tags,
            owner_id=owner_id,
            is_public=model_data.is_public
        )
        
        await model.insert()
        return model

    async def list_models(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Model]:
        """List models accessible to user"""
        
        # Build query
        query = {"$or": [{"owner_id": user_id}, {"is_public": True}]}
        
        if model_type:
            query["model_type"] = model_type
        if status:
            query["status"] = status
        
        models = await Model.find(query).skip(skip).limit(limit).to_list()
        return models

    async def get_model(self, model_id: str, user_id: str) -> Optional[Model]:
        """Get a specific model"""
        model = await Model.get(model_id)
        
        if not model:
            return None
        
        # Check access permissions
        if model.owner_id != user_id and not model.is_public:
            return None
        
        return model

    async def update_model(self, model_id: str, model_update: ModelUpdate, user_id: str) -> Optional[Model]:
        """Update a model"""
        model = await Model.get(model_id)
        
        if not model or model.owner_id != user_id:
            return None
        
        # Update fields
        update_data = model_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(model, field, value)
        
        model.updated_at = datetime.utcnow()
        await model.save()
        
        return model

    async def delete_model(self, model_id: str, user_id: str) -> bool:
        """Delete a model"""
        model = await Model.get(model_id)
        
        if not model or model.owner_id != user_id:
            return False
        
        # Undeploy if deployed
        if model.status == ModelStatus.DEPLOYED:
            await self.undeploy_model(model_id, user_id)
        
        # Delete model files
        if model.artifacts and model.artifacts.get('model_path'):
            model_path = model.artifacts['model_path']
            if os.path.exists(model_path):
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path, ignore_errors=True)
                else:
                    os.remove(model_path)
        
        # Delete from storage service
        if self.storage_service.is_available():
            await self.storage_service.delete_file(f"models/{model_id}")
        
        # Delete document
        await model.delete()
        
        return True

    async def deploy_model(self, model_id: str, user_id: str) -> Optional[str]:
        """Deploy a model"""
        model = await Model.get(model_id)
        
        if not model or model.owner_id != user_id:
            return None
        
        if model.status != ModelStatus.READY:
            raise ValueError("Model is not ready for deployment")
        
        # TODO: Implement actual deployment logic
        # This would typically involve:
        # 1. Creating a deployment endpoint
        # 2. Loading the model into a serving framework
        # 3. Configuring load balancing and scaling
        
        # For now, simulate deployment
        deployment_url = f"http://localhost:8080/models/{model_id}/predict"
        
        model.status = ModelStatus.DEPLOYED
        model.deployment_url = deployment_url
        model.deployment_config = {
            "endpoint": deployment_url,
            "deployed_at": datetime.utcnow().isoformat(),
            "instance_type": "cpu",
            "replicas": 1
        }
        model.updated_at = datetime.utcnow()
        
        await model.save()
        
        return deployment_url

    async def undeploy_model(self, model_id: str, user_id: str) -> bool:
        """Undeploy a model"""
        model = await Model.get(model_id)
        
        if not model or model.owner_id != user_id:
            return False
        
        if model.status != ModelStatus.DEPLOYED:
            return False
        
        # TODO: Implement actual undeployment logic
        
        model.status = ModelStatus.READY
        model.deployment_url = None
        model.deployment_config = None
        model.updated_at = datetime.utcnow()
        
        await model.save()
        
        return True

    async def get_download_url(self, model_id: str, user_id: str) -> Optional[str]:
        """Get model download URL"""
        model = await Model.get(model_id)
        
        if not model:
            return None
        
        # Check access permissions
        if model.owner_id != user_id and not model.is_public:
            return None
        
        if not model.artifacts or not model.artifacts.get('model_path'):
            return None
        
        # Generate presigned URL if using storage service
        if self.storage_service.is_available():
            return await self.storage_service.get_file_url(f"models/{model_id}")
        
        # Return local file path (in production, this should be a proper download endpoint)
        return f"/api/v1/models/{model_id}/download"

    async def upload_model(
        self, 
        file: UploadFile, 
        name: str,
        description: Optional[str] = None,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        owner_id: str = None
    ) -> Model:
        """Upload a pre-trained model"""
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        model_dir = os.path.join(settings.models_dir, filename.replace('.', '_'))
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(model_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create model document
        model = Model(
            name=name,
            description=description,
            model_type=model_type or "custom",
            framework=framework or "unknown",
            status=ModelStatus.READY,
            model_size=file_size,
            artifacts={
                "model_path": model_dir,
                "original_filename": file.filename
            },
            owner_id=owner_id
        )
        
        await model.insert()
        
        # Upload to storage service if available
        if self.storage_service.is_available():
            await self.storage_service.upload_file(file_path, f"models/{model.id}/{file.filename}")
        
        return model

    async def evaluate_model(self, model_id: str, dataset_id: str, user_id: str) -> Optional[dict]:
        """Evaluate a model on a dataset"""
        model = await Model.get(model_id)
        
        if not model or model.owner_id != user_id:
            return None
        
        # TODO: Implement model evaluation logic
        # This would involve:
        # 1. Loading the model
        # 2. Loading the evaluation dataset
        # 3. Running inference
        # 4. Computing metrics
        
        # For now, return mock evaluation results
        evaluation_results = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.96,
            "f1_score": 0.95,
            "evaluated_at": datetime.utcnow().isoformat(),
            "dataset_id": dataset_id
        }
        
        return evaluation_results

    async def compare_models(self, model_ids: List[str], user_id: str) -> Optional[dict]:
        """Compare multiple models"""
        models = []
        
        for model_id in model_ids:
            model = await Model.get(model_id)
            if model and (model.owner_id == user_id or model.is_public):
                models.append(model)
        
        if not models:
            return None
        
        # TODO: Implement model comparison logic
        
        comparison_results = {
            "models": [
                {
                    "id": str(model.id),
                    "name": model.name,
                    "metrics": model.metrics.dict() if model.metrics else {},
                    "model_size": model.model_size,
                    "parameters_count": model.parameters_count
                }
                for model in models
            ],
            "compared_at": datetime.utcnow().isoformat()
        }
        
        return comparison_results

    async def get_model_info(self, model_id: str, user_id: str) -> Optional[dict]:
        """Get detailed model information"""
        model = await Model.get(model_id)
        
        if not model:
            return None
        
        # Check access permissions
        if model.owner_id != user_id and not model.is_public:
            return None
        
        model_info = {
            "id": str(model.id),
            "name": model.name,
            "description": model.description,
            "model_type": model.model_type,
            "framework": model.framework,
            "status": model.status,
            "metrics": model.metrics.dict() if model.metrics else {},
            "model_size": model.model_size,
            "parameters_count": model.parameters_count,
            "artifacts": model.artifacts,
            "deployment_info": {
                "url": model.deployment_url,
                "config": model.deployment_config
            } if model.status == ModelStatus.DEPLOYED else None,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }
        
        return model_info
