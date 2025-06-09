#!/usr/bin/env python3
"""
Diffusion Model Training Script
Supports training of diffusion models for image generation
"""

import os
import json
import argparse
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline
from datasets import load_dataset
from PIL import Image
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.noise_scheduler = None
        self.optimizer = None
        self.train_dataloader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_dataset(self):
        """Prepare the dataset"""
        dataset_path = self.config['dataset_path']
        image_size = self.config.get('image_size', 64)
        batch_size = self.config.get('batch_size', 16)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Load dataset
        if os.path.isdir(dataset_path):
            # Load from directory
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            
            def transform_images(examples):
                images = [transform(image.convert("RGB")) for image in examples["image"]]
                return {"images": images}
            
            dataset = dataset.with_transform(transform_images)
            train_dataset = dataset["train"]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        logger.info(f"Dataset loaded with {len(train_dataset)} images")
        
    def build_model(self):
        """Build the diffusion model"""
        image_size = self.config.get('image_size', 64)
        
        # Create UNet model
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
        
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-6)
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_step(self, batch):
        """Single training step"""
        clean_images = batch["images"].to(self.device)
        
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
        
    def train(self):
        """Train the model"""
        num_epochs = self.config.get('num_epochs', 50)
        save_steps = self.config.get('save_steps', 1000)
        log_steps = self.config.get('log_steps', 100)
        
        global_step = 0
        
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log progress
                if global_step % log_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(f"Step {global_step}, Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(global_step)
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Generate sample images
            if (epoch + 1) % 10 == 0:
                self.generate_samples(epoch + 1)
        
        # Save final model
        self.save_final_model()
        
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")
        
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['output_dir'], f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'model.pt'))
        
    def save_final_model(self):
        """Save final trained model"""
        # Create pipeline
        pipeline = DDPMPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler
        )
        
        # Save pipeline
        pipeline.save_pretrained(self.config['output_dir'])
        
        # Also save raw model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, os.path.join(self.config['output_dir'], 'final_model.pt'))
        
    def generate_samples(self, epoch):
        """Generate sample images"""
        self.model.eval()
        
        # Create pipeline for generation
        pipeline = DDPMPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler
        )
        
        # Generate images
        with torch.no_grad():
            images = pipeline(
                batch_size=4,
                generator=torch.manual_seed(42)
            ).images
        
        # Save images
        sample_dir = os.path.join(self.config['output_dir'], 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            image.save(os.path.join(sample_dir, f'epoch_{epoch}_sample_{i}.png'))
        
        self.model.train()


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    trainer = DiffusionTrainer(config)
    trainer.prepare_dataset()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
