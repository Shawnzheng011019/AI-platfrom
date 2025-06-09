#!/usr/bin/env python3
"""
Time Series Forecasting Model Training Script
Supports LSTM, GRU, Transformer models for time series prediction
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, target_column):
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length].drop(columns=[self.target_column]).values
        y = self.data[idx + self.sequence_length][self.target_column]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class TimeSeriesTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and preprocess time series data"""
        logger.info(f"Loading data from {self.config['dataset_path']}")
        
        # Load data
        if self.config['dataset_path'].endswith('.csv'):
            self.data = pd.read_csv(self.config['dataset_path'])
        elif self.config['dataset_path'].endswith('.json'):
            self.data = pd.read_json(self.config['dataset_path'])
        else:
            raise ValueError("Unsupported file format")
        
        # Parse datetime column if specified
        if 'datetime_column' in self.config:
            self.data[self.config['datetime_column']] = pd.to_datetime(
                self.data[self.config['datetime_column']]
            )
            self.data = self.data.sort_values(self.config['datetime_column'])
        
        # Select features and target
        feature_columns = self.config.get('feature_columns', [])
        if not feature_columns:
            feature_columns = [col for col in self.data.columns 
                             if col != self.config['target_column']]
        
        self.feature_columns = feature_columns
        self.target_column = self.config['target_column']
        
        # Prepare data for scaling
        feature_data = self.data[feature_columns + [self.target_column]]
        
        # Scale data
        scaled_data = self.scaler.fit_transform(feature_data)
        self.scaled_df = pd.DataFrame(scaled_data, columns=feature_columns + [self.target_column])
        
        # Split data
        train_size = int(len(self.scaled_df) * self.config.get('train_split', 0.8))
        val_size = int(len(self.scaled_df) * self.config.get('val_split', 0.1))
        
        self.train_data = self.scaled_df[:train_size]
        self.val_data = self.scaled_df[train_size:train_size + val_size]
        self.test_data = self.scaled_df[train_size + val_size:]
        
        logger.info(f"Data loaded: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test samples")
        
    def create_datasets(self):
        """Create PyTorch datasets"""
        sequence_length = self.config.get('sequence_length', 10)
        
        self.train_dataset = TimeSeriesDataset(self.train_data, sequence_length, self.target_column)
        self.val_dataset = TimeSeriesDataset(self.val_data, sequence_length, self.target_column)
        self.test_dataset = TimeSeriesDataset(self.test_data, sequence_length, self.target_column)
        
        batch_size = self.config.get('batch_size', 32)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    def build_model(self):
        """Build the time series model"""
        model_type = self.config.get('model_architecture', 'lstm').lower()
        input_size = len(self.feature_columns)
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        output_size = 1
        dropout = self.config.get('dropout', 0.2)
        
        if model_type == 'lstm':
            self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
        elif model_type == 'gru':
            self.model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
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
        
        logger.info(f"Model built: {model_type.upper()} with {sum(p.numel() for p in self.model.parameters())} parameters")
        
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
            
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
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
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'final_model.pth'))
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train Time Series Forecasting Model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = TimeSeriesTrainer(config)
    trainer.load_data()
    trainer.create_datasets()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
