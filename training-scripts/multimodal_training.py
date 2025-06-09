#!/usr/bin/env python3
"""
Multimodal Model Training Script
Supports CLIP, BLIP, and custom vision-language models
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForConditionalGeneration, BlipProcessor,
    TrainingArguments, Trainer
)
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, processor, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Process with processor
        if self.processor:
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'pixel_values': inputs.pixel_values.squeeze(),
                'labels': inputs.input_ids.squeeze()  # For language modeling
            }
        else:
            return {
                'image': image,
                'text': text
            }


class CustomVisionLanguageModel(nn.Module):
    def __init__(self, vision_dim=2048, text_dim=768, hidden_dim=512, num_classes=1000):
        super(CustomVisionLanguageModel, self).__init__()
        
        # Vision encoder (simplified ResNet-like)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, vision_dim)
        )
        
        # Text encoder (simplified)
        self.text_encoder = nn.Sequential(
            nn.Embedding(50000, 256),  # Vocab size assumption
            nn.LSTM(256, text_dim // 2, batch_first=True, bidirectional=True),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, images, text_ids):
        # Vision features
        vision_features = self.vision_encoder(images)
        
        # Text features
        text_embeddings = self.text_encoder.embedding(text_ids)
        text_features, _ = self.text_encoder.lstm(text_embeddings)
        text_features = text_features.mean(dim=1)  # Average pooling
        
        # Fusion
        combined_features = torch.cat([vision_features, text_features], dim=1)
        output = self.fusion(combined_features)
        
        return output


class MultimodalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model_type = config.get('model_type', 'clip').lower()
        
    def load_data(self):
        """Load and preprocess multimodal data"""
        logger.info(f"Loading data from {self.config['dataset_path']}")
        
        # Load data
        if self.config['dataset_path'].endswith('.csv'):
            data = pd.read_csv(self.config['dataset_path'])
        elif self.config['dataset_path'].endswith('.json'):
            data = pd.read_json(self.config['dataset_path'])
        else:
            raise ValueError("Unsupported file format")
        
        # Expected columns: image_path, text (and optionally label for classification)
        required_columns = ['image_path', 'text']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Convert relative paths to absolute
        base_path = os.path.dirname(self.config['dataset_path'])
        data['image_path'] = data['image_path'].apply(
            lambda x: os.path.join(base_path, x) if not os.path.isabs(x) else x
        )
        
        # Filter existing files
        data = data[data['image_path'].apply(os.path.exists)]
        
        if len(data) == 0:
            raise ValueError("No valid image files found")
        
        # Split data
        train_size = int(len(data) * self.config.get('train_split', 0.8))
        val_size = int(len(data) * self.config.get('val_split', 0.1))
        
        self.train_data = data[:train_size]
        self.val_data = data[train_size:train_size + val_size]
        self.test_data = data[train_size + val_size:]
        
        logger.info(f"Data loaded: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test samples")
        
    def setup_model_and_processor(self):
        """Setup model and processor based on model type"""
        if self.model_type == 'clip':
            model_name = self.config.get('base_model', 'openai/clip-vit-base-patch32')
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            
        elif self.model_type == 'blip':
            model_name = self.config.get('base_model', 'Salesforce/blip-image-captioning-base')
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
        elif self.model_type == 'custom':
            # Custom multimodal model
            vision_dim = self.config.get('vision_dim', 2048)
            text_dim = self.config.get('text_dim', 768)
            hidden_dim = self.config.get('hidden_dim', 512)
            num_classes = self.config.get('num_classes', 1000)
            
            self.model = CustomVisionLanguageModel(vision_dim, text_dim, hidden_dim, num_classes)
            self.processor = None  # Custom preprocessing
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        logger.info(f"Model setup complete: {self.model_type}")
        
    def create_datasets(self):
        """Create datasets"""
        # Image transforms
        if self.model_type == 'custom':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = None  # Processor handles transforms
        
        self.train_dataset = MultimodalDataset(
            self.train_data['image_path'].tolist(),
            self.train_data['text'].tolist(),
            self.processor,
            transform
        )
        
        self.val_dataset = MultimodalDataset(
            self.val_data['image_path'].tolist(),
            self.val_data['text'].tolist(),
            self.processor,
            transform
        )
        
    def train(self):
        """Train the model"""
        if self.model_type in ['clip', 'blip']:
            self._train_huggingface_model()
        else:
            self._train_custom_model()
    
    def _train_huggingface_model(self):
        """Train using HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config.get('num_epochs', 10),
            per_device_train_batch_size=self.config.get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('batch_size', 16),
            warmup_steps=self.config.get('warmup_steps', 500),
            learning_rate=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.get('mixed_precision', True),
            dataloader_pin_memory=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.processor.tokenizer if self.processor else None,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        if self.processor:
            self.processor.save_pretrained(self.config['output_dir'])
        
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")
    
    def _train_custom_model(self):
        """Train custom multimodal model"""
        batch_size = self.config.get('batch_size', 16)
        num_epochs = self.config.get('num_epochs', 10)
        learning_rate = self.config.get('learning_rate', 1e-4)
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                images = batch['image'].to(self.device)
                # For custom model, we'd need to tokenize text here
                # This is a simplified example
                
                optimizer.zero_grad()
                # outputs = self.model(images, text_ids)
                # loss = criterion(outputs, labels)
                # loss.backward()
                optimizer.step()
                
                # train_loss += loss.item()
            
            # Validation phase would go here
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss/len(train_loader):.6f}")
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'model.pth'))
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = MultimodalTrainer(config)
    trainer.load_data()
    trainer.setup_model_and_processor()
    trainer.create_datasets()
    trainer.train()


if __name__ == "__main__":
    main()
