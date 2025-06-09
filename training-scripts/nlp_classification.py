#!/usr/bin/env python3
"""
NLP Text Classification Training Script
Supports training of text classification models using various architectures
"""

import os
import json
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NLPClassificationTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_labels = 0
        
    def load_data(self):
        """Load and prepare the dataset"""
        dataset_path = self.config['dataset_path']
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        # Assume columns are 'text' and 'label'
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        
        # Encode labels
        unique_labels = df['label'].unique()
        self.num_labels = len(unique_labels)
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        df['label_id'] = df['label'].map(label_to_id)
        
        logger.info(f"Number of classes: {self.num_labels}")
        logger.info(f"Label mapping: {label_to_id}")
        
        # Split data
        train_size = int(len(df) * self.config.get('train_split', 0.8))
        val_size = int(len(df) * self.config.get('val_split', 0.1))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        # Load tokenizer
        model_name = self.config.get('base_model', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create datasets
        max_length = self.config.get('max_length', 512)
        self.train_dataset = TextClassificationDataset(
            train_df['text'].tolist(),
            train_df['label_id'].tolist(),
            self.tokenizer,
            max_length
        )
        
        self.val_dataset = TextClassificationDataset(
            val_df['text'].tolist(),
            val_df['label_id'].tolist(),
            self.tokenizer,
            max_length
        )
        
    def load_model(self):
        """Load the model"""
        model_name = self.config.get('base_model', 'bert-base-uncased')
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels
        )
        
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def train(self):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('batch_size', 16),
            warmup_steps=self.config.get('warmup_steps', 500),
            weight_decay=self.config.get('weight_decay', 0.01),
            learning_rate=self.config.get('learning_rate', 2e-5),
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            fp16=self.config.get('mixed_precision', True),
            dataloader_pin_memory=False,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(os.path.join(self.config['output_dir'], 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train NLP classification model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    trainer = NLPClassificationTrainer(config)
    trainer.load_data()
    trainer.load_model()
    trainer.train()


if __name__ == "__main__":
    main()
