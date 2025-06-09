import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from app.models.training_job import TrainingJob, TrainingConfig, TrainingStatus, ModelType
from app.models.automl_job import AutoMLJob, AutoMLConfig, AutoMLStatus
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)


class AutoMLService:
    def __init__(self):
        self.training_service = TrainingService()
        
    async def create_automl_job(self, config: AutoMLConfig, user_id: str) -> AutoMLJob:
        """Create a new AutoML job"""
        automl_job = AutoMLJob(
            name=config.name,
            description=config.description,
            config=config,
            owner_id=user_id,
            status=AutoMLStatus.PENDING
        )
        
        await automl_job.insert()
        logger.info(f"Created AutoML job: {automl_job.id}")
        return automl_job
    
    async def start_automl_job(self, job_id: str) -> bool:
        """Start an AutoML job"""
        job = await AutoMLJob.get(job_id)
        if not job:
            return False
        
        if job.status != AutoMLStatus.PENDING:
            return False
        
        job.status = AutoMLStatus.RUNNING
        job.started_at = datetime.utcnow()
        await job.save()
        
        # Start AutoML process in background
        asyncio.create_task(self._run_automl_process(job))
        return True
    
    async def _run_automl_process(self, job: AutoMLJob):
        """Run the AutoML optimization process"""
        try:
            logger.info(f"Starting AutoML process for job {job.id}")
            
            if job.config.optimization_type == "hyperparameter":
                await self._run_hyperparameter_optimization(job)
            elif job.config.optimization_type == "architecture":
                await self._run_architecture_search(job)
            elif job.config.optimization_type == "feature_engineering":
                await self._run_feature_engineering(job)
            else:
                raise ValueError(f"Unknown optimization type: {job.config.optimization_type}")
            
            job.status = AutoMLStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            await job.save()
            
            logger.info(f"AutoML job {job.id} completed successfully")
            
        except Exception as e:
            logger.error(f"AutoML job {job.id} failed: {e}")
            job.status = AutoMLStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await job.save()
    
    async def _run_hyperparameter_optimization(self, job: AutoMLJob):
        """Run hyperparameter optimization"""
        config = job.config
        
        # Define hyperparameter search space
        param_grid = self._get_hyperparameter_grid(config)
        
        best_score = float('-inf')
        best_params = None
        trial_results = []
        
        total_trials = len(list(ParameterGrid(param_grid)))
        completed_trials = 0
        
        for params in ParameterGrid(param_grid):
            try:
                # Create training config with current parameters
                training_config = self._create_training_config(config, params)
                
                # Run training job
                training_job = await self._run_training_trial(training_config, job)
                
                # Wait for completion and get results
                score = await self._wait_for_training_completion(training_job)
                
                trial_result = {
                    "trial_id": len(trial_results) + 1,
                    "parameters": params,
                    "score": score,
                    "training_job_id": str(training_job.id)
                }
                trial_results.append(trial_result)
                
                # Update best parameters
                if score > best_score:
                    best_score = score
                    best_params = params
                
                completed_trials += 1
                job.progress = (completed_trials / total_trials) * 100
                job.trial_results = trial_results
                job.best_parameters = best_params
                job.best_score = best_score
                await job.save()
                
                logger.info(f"Trial {completed_trials}/{total_trials} completed. Score: {score}")
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                continue
        
        # Save final results
        job.trial_results = trial_results
        job.best_parameters = best_params
        job.best_score = best_score
        await job.save()
    
    async def _run_architecture_search(self, job: AutoMLJob):
        """Run neural architecture search"""
        # Simplified architecture search - in practice, this would be more sophisticated
        config = job.config
        
        architectures = [
            {"hidden_dims": [64, 32], "num_layers": 2},
            {"hidden_dims": [128, 64], "num_layers": 2},
            {"hidden_dims": [256, 128, 64], "num_layers": 3},
            {"hidden_dims": [512, 256, 128], "num_layers": 3},
        ]
        
        best_score = float('-inf')
        best_architecture = None
        trial_results = []
        
        for i, arch in enumerate(architectures):
            try:
                # Create training config with current architecture
                training_config = self._create_training_config(config, arch)
                
                # Run training job
                training_job = await self._run_training_trial(training_config, job)
                
                # Wait for completion and get results
                score = await self._wait_for_training_completion(training_job)
                
                trial_result = {
                    "trial_id": i + 1,
                    "architecture": arch,
                    "score": score,
                    "training_job_id": str(training_job.id)
                }
                trial_results.append(trial_result)
                
                if score > best_score:
                    best_score = score
                    best_architecture = arch
                
                job.progress = ((i + 1) / len(architectures)) * 100
                job.trial_results = trial_results
                job.best_parameters = best_architecture
                job.best_score = best_score
                await job.save()
                
            except Exception as e:
                logger.error(f"Architecture trial failed: {e}")
                continue
    
    async def _run_feature_engineering(self, job: AutoMLJob):
        """Run automated feature engineering"""
        # Simplified feature engineering - in practice, this would analyze the dataset
        # and automatically generate features
        config = job.config
        
        feature_configs = [
            {"feature_selection": "variance", "n_features": 100},
            {"feature_selection": "mutual_info", "n_features": 100},
            {"feature_selection": "chi2", "n_features": 100},
            {"feature_selection": "rfe", "n_features": 100},
        ]
        
        best_score = float('-inf')
        best_features = None
        trial_results = []
        
        for i, feat_config in enumerate(feature_configs):
            try:
                # Create training config with current feature engineering
                training_config = self._create_training_config(config, feat_config)
                
                # Run training job
                training_job = await self._run_training_trial(training_config, job)
                
                # Wait for completion and get results
                score = await self._wait_for_training_completion(training_job)
                
                trial_result = {
                    "trial_id": i + 1,
                    "feature_config": feat_config,
                    "score": score,
                    "training_job_id": str(training_job.id)
                }
                trial_results.append(trial_result)
                
                if score > best_score:
                    best_score = score
                    best_features = feat_config
                
                job.progress = ((i + 1) / len(feature_configs)) * 100
                job.trial_results = trial_results
                job.best_parameters = best_features
                job.best_score = best_score
                await job.save()
                
            except Exception as e:
                logger.error(f"Feature engineering trial failed: {e}")
                continue
    
    def _get_hyperparameter_grid(self, config: AutoMLConfig) -> Dict:
        """Get hyperparameter search space based on model type"""
        if config.model_type == ModelType.LLM:
            return {
                "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
                "batch_size": [8, 16, 32],
                "num_epochs": [3, 5, 10],
                "warmup_steps": [0, 100, 500]
            }
        elif config.model_type == ModelType.CV_CLASSIFICATION:
            return {
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "batch_size": [16, 32, 64],
                "num_epochs": [10, 20, 50],
                "weight_decay": [0.01, 0.001, 0.0001]
            }
        else:
            # Default grid
            return {
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "batch_size": [16, 32, 64],
                "num_epochs": [10, 20, 30]
            }
    
    def _create_training_config(self, automl_config: AutoMLConfig, params: Dict) -> TrainingConfig:
        """Create training configuration from AutoML config and parameters"""
        return TrainingConfig(
            model_name=f"{automl_config.name}_trial",
            model_type=automl_config.model_type,
            base_model=automl_config.base_model,
            learning_rate=params.get("learning_rate", 1e-4),
            batch_size=params.get("batch_size", 32),
            num_epochs=params.get("num_epochs", 10),
            warmup_steps=params.get("warmup_steps", 0),
            weight_decay=params.get("weight_decay", 0.01),
            custom_params=params
        )
    
    async def _run_training_trial(self, training_config: TrainingConfig, automl_job: AutoMLJob) -> TrainingJob:
        """Run a single training trial"""
        training_job_data = {
            "name": f"{automl_job.name}_trial_{len(automl_job.trial_results) + 1}",
            "description": f"AutoML trial for job {automl_job.id}",
            "config": training_config,
            "dataset_id": automl_job.config.dataset_id,
            "priority": 1  # High priority for AutoML trials
        }
        
        training_job = await self.training_service.create_training_job(
            training_job_data, automl_job.owner_id
        )
        
        # Start training
        await self.training_service.start_training_job(str(training_job.id))
        
        return training_job
    
    async def _wait_for_training_completion(self, training_job: TrainingJob) -> float:
        """Wait for training job to complete and return score"""
        max_wait_time = 3600  # 1 hour max wait
        check_interval = 30  # Check every 30 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            # Refresh job status
            updated_job = await TrainingJob.get(training_job.id)
            
            if updated_job.status == TrainingStatus.COMPLETED:
                # Extract score from final metrics
                if updated_job.final_metrics:
                    return updated_job.final_metrics.get("accuracy", 0.0)
                else:
                    return 0.0
            elif updated_job.status == TrainingStatus.FAILED:
                return 0.0
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Timeout
        return 0.0
    
    async def get_automl_job(self, job_id: str, user_id: str) -> Optional[AutoMLJob]:
        """Get AutoML job by ID"""
        job = await AutoMLJob.get(job_id)
        if job and job.owner_id == user_id:
            return job
        return None
    
    async def list_automl_jobs(self, user_id: str) -> List[AutoMLJob]:
        """List AutoML jobs for a user"""
        return await AutoMLJob.find(AutoMLJob.owner_id == user_id).to_list()
    
    async def stop_automl_job(self, job_id: str, user_id: str) -> bool:
        """Stop a running AutoML job"""
        job = await AutoMLJob.get(job_id)
        if not job or job.owner_id != user_id:
            return False
        
        if job.status == AutoMLStatus.RUNNING:
            job.status = AutoMLStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            await job.save()
            return True
        
        return False
