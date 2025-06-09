#!/usr/bin/env python3
"""
Recommendation System Training Script
Supports Collaborative Filtering, Matrix Factorization, and Deep Learning approaches
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, dropout=0.2):
        super(MatrixFactorization, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.user_bias.weight, std=0.1)
        nn.init.normal_(self.item_bias.weight, std=0.1)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.dropout(self.user_embedding(user_ids))
        item_emb = self.dropout(self.item_embedding(item_ids))
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        dot_product = (user_emb * item_emb).sum(dim=1)
        rating = dot_product + user_bias + item_bias + self.global_bias
        
        return rating


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64], dropout=0.2):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        rating = self.mlp(x).squeeze()
        
        return rating


class RecommendationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        
    def load_data(self):
        """Load and preprocess recommendation data"""
        logger.info(f"Loading data from {self.config['dataset_path']}")
        
        # Load data
        if self.config['dataset_path'].endswith('.csv'):
            self.data = pd.read_csv(self.config['dataset_path'])
        elif self.config['dataset_path'].endswith('.json'):
            self.data = pd.read_json(self.config['dataset_path'])
        else:
            raise ValueError("Unsupported file format")
        
        # Expected columns: user_id, item_id, rating
        required_columns = ['user_id', 'item_id', 'rating']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Create user and item mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.data['user_id'].unique())}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.data['item_id'].unique())}
        
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Map IDs to indices
        self.data['user_idx'] = self.data['user_id'].map(self.user_to_idx)
        self.data['item_idx'] = self.data['item_id'].map(self.item_to_idx)
        
        self.num_users = len(self.user_to_idx)
        self.num_items = len(self.item_to_idx)
        
        logger.info(f"Data loaded: {len(self.data)} interactions, {self.num_users} users, {self.num_items} items")
        
        # Split data
        train_data, temp_data = train_test_split(
            self.data, test_size=0.3, random_state=42, stratify=self.data['user_idx']
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42, stratify=temp_data['user_idx']
        )
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
    def create_datasets(self):
        """Create PyTorch datasets"""
        self.train_dataset = RecommendationDataset(
            self.train_data['user_idx'].values,
            self.train_data['item_idx'].values,
            self.train_data['rating'].values
        )
        
        self.val_dataset = RecommendationDataset(
            self.val_data['user_idx'].values,
            self.val_data['item_idx'].values,
            self.val_data['rating'].values
        )
        
        self.test_dataset = RecommendationDataset(
            self.test_data['user_idx'].values,
            self.test_data['item_idx'].values,
            self.test_data['rating'].values
        )
        
        batch_size = self.config.get('batch_size', 1024)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    def build_model(self):
        """Build the recommendation model"""
        model_type = self.config.get('model_type', 'matrix_factorization').lower()
        embedding_dim = self.config.get('embedding_dim', 50)
        dropout = self.config.get('dropout', 0.2)
        
        if model_type == 'matrix_factorization':
            self.model = MatrixFactorization(
                self.num_users, self.num_items, embedding_dim, dropout
            )
        elif model_type == 'neural_cf':
            hidden_dims = self.config.get('hidden_dims', [128, 64])
            self.model = NeuralCollaborativeFiltering(
                self.num_users, self.num_items, embedding_dim, hidden_dims, dropout
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        logger.info(f"Model built: {model_type} with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train(self):
        """Train the model"""
        num_epochs = self.config.get('num_epochs', 100)
        best_val_loss = float('inf')
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for user_ids, item_ids, ratings in self.train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in self.val_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    
                    predictions = self.model(user_ids, item_ids)
                    loss = self.criterion(predictions, ratings)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model and mappings
        torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'final_model.pth'))
        
        # Save user and item mappings
        mappings = {
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'num_users': self.num_users,
            'num_items': self.num_items
        }
        
        with open(os.path.join(self.config['output_dir'], 'mappings.json'), 'w') as f:
            json.dump(mappings, f, indent=2)
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train Recommendation System')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = RecommendationTrainer(config)
    trainer.load_data()
    trainer.create_datasets()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
