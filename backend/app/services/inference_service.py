import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np
from datetime import datetime
from app.models.model import Model, ModelStatus
from app.models.inference_job import InferenceJob, InferenceRequest, InferenceResponse
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and caching of trained models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_configs = {}
    
    async def load_model(self, model_id: str) -> Any:
        """Load a model into memory"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # Get model from database
        model = await Model.get(model_id)
        if not model or model.status != ModelStatus.READY:
            raise ValueError(f"Model {model_id} not found or not ready")
        
        # Load model based on type
        if model.model_type.value == "llm":
            loaded_model = await self._load_llm_model(model)
        elif model.model_type.value == "cv_classification":
            loaded_model = await self._load_cv_model(model)
        elif model.model_type.value == "time_series":
            loaded_model = await self._load_time_series_model(model)
        elif model.model_type.value == "recommendation":
            loaded_model = await self._load_recommendation_model(model)
        else:
            loaded_model = await self._load_generic_model(model)
        
        # Cache the loaded model
        self.loaded_models[model_id] = loaded_model
        self.model_configs[model_id] = model
        
        logger.info(f"Model {model_id} loaded successfully")
        return loaded_model
    
    async def _load_llm_model(self, model: Model) -> Any:
        """Load LLM model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = model.artifacts.model_path if model.artifacts else None
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Model path not found: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_obj = AutoModelForCausalLM.from_pretrained(model_path)
            
            return {
                "model": model_obj,
                "tokenizer": tokenizer,
                "type": "llm"
            }
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    async def _load_cv_model(self, model: Model) -> Any:
        """Load computer vision model"""
        try:
            model_path = model.artifacts.model_path if model.artifacts else None
            if not model_path:
                raise ValueError("Model path not found")
            
            # Load PyTorch model
            model_file = os.path.join(model_path, "model.pth")
            if os.path.exists(model_file):
                model_obj = torch.load(model_file, map_location='cpu')
                return {
                    "model": model_obj,
                    "type": "cv_classification"
                }
            else:
                raise ValueError(f"Model file not found: {model_file}")
        except Exception as e:
            logger.error(f"Failed to load CV model: {e}")
            raise
    
    async def _load_time_series_model(self, model: Model) -> Any:
        """Load time series model"""
        try:
            model_path = model.artifacts.model_path if model.artifacts else None
            if not model_path:
                raise ValueError("Model path not found")
            
            model_file = os.path.join(model_path, "model.pth")
            scaler_file = os.path.join(model_path, "scaler.pkl")
            
            if os.path.exists(model_file):
                import pickle
                model_obj = torch.load(model_file, map_location='cpu')
                scaler = None
                if os.path.exists(scaler_file):
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                
                return {
                    "model": model_obj,
                    "scaler": scaler,
                    "type": "time_series"
                }
            else:
                raise ValueError(f"Model file not found: {model_file}")
        except Exception as e:
            logger.error(f"Failed to load time series model: {e}")
            raise
    
    async def _load_recommendation_model(self, model: Model) -> Any:
        """Load recommendation model"""
        try:
            model_path = model.artifacts.model_path if model.artifacts else None
            if not model_path:
                raise ValueError("Model path not found")
            
            model_file = os.path.join(model_path, "model.pth")
            mappings_file = os.path.join(model_path, "mappings.json")
            
            if os.path.exists(model_file) and os.path.exists(mappings_file):
                model_obj = torch.load(model_file, map_location='cpu')
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
                
                return {
                    "model": model_obj,
                    "mappings": mappings,
                    "type": "recommendation"
                }
            else:
                raise ValueError("Model or mappings file not found")
        except Exception as e:
            logger.error(f"Failed to load recommendation model: {e}")
            raise
    
    async def _load_generic_model(self, model: Model) -> Any:
        """Load generic PyTorch model"""
        try:
            model_path = model.artifacts.model_path if model.artifacts else None
            if not model_path:
                raise ValueError("Model path not found")
            
            model_file = os.path.join(model_path, "model.pth")
            if os.path.exists(model_file):
                model_obj = torch.load(model_file, map_location='cpu')
                return {
                    "model": model_obj,
                    "type": "generic"
                }
            else:
                raise ValueError(f"Model file not found: {model_file}")
        except Exception as e:
            logger.error(f"Failed to load generic model: {e}")
            raise
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.model_configs[model_id]
            logger.info(f"Model {model_id} unloaded")


class InferenceService:
    """Handles model inference requests"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
    
    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a model"""
        start_time = datetime.utcnow()
        
        try:
            # Load model if not already loaded
            loaded_model = await self.model_loader.load_model(request.model_id)
            
            # Run inference based on model type
            if loaded_model["type"] == "llm":
                result = await self._run_llm_inference(loaded_model, request.input_data)
            elif loaded_model["type"] == "cv_classification":
                result = await self._run_cv_inference(loaded_model, request.input_data)
            elif loaded_model["type"] == "time_series":
                result = await self._run_time_series_inference(loaded_model, request.input_data)
            elif loaded_model["type"] == "recommendation":
                result = await self._run_recommendation_inference(loaded_model, request.input_data)
            else:
                result = await self._run_generic_inference(loaded_model, request.input_data)
            
            end_time = datetime.utcnow()
            inference_time = (end_time - start_time).total_seconds()
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                result=result,
                inference_time_seconds=inference_time,
                status="success",
                timestamp=end_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            inference_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Inference failed for model {request.model_id}: {e}")
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                result=None,
                inference_time_seconds=inference_time,
                status="error",
                error_message=str(e),
                timestamp=end_time
            )
    
    async def _run_llm_inference(self, loaded_model: Dict, input_data: Dict) -> Dict:
        """Run inference for LLM models"""
        model = loaded_model["model"]
        tokenizer = loaded_model["tokenizer"]
        
        prompt = input_data.get("prompt", "")
        max_length = input_data.get("max_length", 100)
        temperature = input_data.get("temperature", 0.7)
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": response,
            "prompt": prompt,
            "tokens_generated": len(outputs[0]) - len(inputs[0])
        }
    
    async def _run_cv_inference(self, loaded_model: Dict, input_data: Dict) -> Dict:
        """Run inference for computer vision models"""
        model = loaded_model["model"]
        
        # Expect image data as base64 or numpy array
        image_data = input_data.get("image")
        if not image_data:
            raise ValueError("No image data provided")
        
        # Convert to tensor (simplified - would need proper preprocessing)
        if isinstance(image_data, list):
            image_tensor = torch.FloatTensor(image_data).unsqueeze(0)
        else:
            raise ValueError("Invalid image data format")
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }
    
    async def _run_time_series_inference(self, loaded_model: Dict, input_data: Dict) -> Dict:
        """Run inference for time series models"""
        model = loaded_model["model"]
        scaler = loaded_model.get("scaler")
        
        # Expect sequence data
        sequence = input_data.get("sequence")
        if not sequence:
            raise ValueError("No sequence data provided")
        
        # Preprocess data
        if scaler:
            sequence = scaler.transform([sequence])[0]
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            prediction = model(sequence_tensor)
            
        # Post-process prediction
        if scaler:
            prediction = scaler.inverse_transform(prediction.numpy())[0]
        else:
            prediction = prediction.numpy()[0]
        
        return {
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else float(prediction),
            "input_sequence_length": len(sequence)
        }
    
    async def _run_recommendation_inference(self, loaded_model: Dict, input_data: Dict) -> Dict:
        """Run inference for recommendation models"""
        model = loaded_model["model"]
        mappings = loaded_model["mappings"]
        
        user_id = input_data.get("user_id")
        num_recommendations = input_data.get("num_recommendations", 10)
        
        if user_id is None:
            raise ValueError("No user_id provided")
        
        # Map user ID to index
        user_idx = mappings["user_to_idx"].get(str(user_id))
        if user_idx is None:
            raise ValueError(f"Unknown user_id: {user_id}")
        
        # Get recommendations (simplified)
        model.eval()
        with torch.no_grad():
            # This would depend on the specific recommendation model architecture
            # For now, return dummy recommendations
            recommendations = list(range(num_recommendations))
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "num_recommendations": len(recommendations)
        }
    
    async def _run_generic_inference(self, loaded_model: Dict, input_data: Dict) -> Dict:
        """Run inference for generic models"""
        model = loaded_model["model"]
        
        # Expect input as tensor data
        input_tensor = torch.FloatTensor(input_data.get("input", []))
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        return {
            "output": output.tolist(),
            "input_shape": list(input_tensor.shape)
        }
    
    async def get_model_info(self, model_id: str) -> Dict:
        """Get information about a loaded model"""
        if model_id in self.model_loader.model_configs:
            model_config = self.model_loader.model_configs[model_id]
            return {
                "model_id": model_id,
                "name": model_config.name,
                "type": model_config.model_type.value,
                "status": "loaded",
                "framework": model_config.framework
            }
        else:
            model = await Model.get(model_id)
            if model:
                return {
                    "model_id": model_id,
                    "name": model.name,
                    "type": model.model_type.value,
                    "status": "not_loaded",
                    "framework": model.framework
                }
            else:
                raise ValueError(f"Model {model_id} not found")
    
    async def list_loaded_models(self) -> List[Dict]:
        """List all currently loaded models"""
        return [
            await self.get_model_info(model_id)
            for model_id in self.model_loader.loaded_models.keys()
        ]
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self.model_loader.loaded_models:
            self.model_loader.unload_model(model_id)
            return True
        return False
