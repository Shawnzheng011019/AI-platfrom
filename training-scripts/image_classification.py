#!/usr/bin/env python3
"""
Image Classification Training Script
Supports training of CNN models for image classification tasks
"""

import os
import json
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        """Prepare data loaders"""
        data_dir = self.config['dataset_path']
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4
        )
        
        self.num_classes = len(train_dataset.classes)
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
    def build_model(self):
        """Build the model"""
        model_name = self.config.get('model_name', 'resnet18')
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(self.device)
        
        # Freeze early layers if specified
        if self.config.get('freeze_backbone', False):
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            if hasattr(self.model, 'fc'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, 'classifier'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                    
    def train(self):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get('lr_step_size', 7),
            gamma=self.config.get('lr_gamma', 0.1)
        )
        
        num_epochs = self.config.get('num_epochs', 10)
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            
            logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
            
            val_epoch_loss = val_running_loss / len(self.val_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(self.val_loader.dataset)
            
            logger.info(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Save best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config['output_dir'], 'best_model.pth'))
            
            scheduler.step()
        
        logger.info(f'Best validation accuracy: {best_acc:.4f}')
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(self.config['output_dir'], 'final_model.pth'))


def main():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    trainer = ImageClassificationTrainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
