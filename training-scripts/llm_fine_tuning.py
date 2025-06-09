#!/usr/bin/env python3
"""
LLM Fine-tuning Script Template
Supports fine-tuning of large language models using Hugging Face Transformers
"""

import os
import json
import argparse
from typing import Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFineTuner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        model_name = self.config.get('base_model', 'gpt2')
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.get('mixed_precision', True) else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def prepare_dataset(self):
        """Prepare dataset for training"""
        dataset_path = self.config['dataset_path']
        
        if dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=dataset_path)
        elif dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, field='data')
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.get('max_length', 512)
            )
        
        self.dataset = dataset.map(tokenize_function, batched=True)
        
        # Split dataset
        train_size = self.config.get('train_split', 0.8)
        if 'train' not in self.dataset:
            split_dataset = self.dataset['train'].train_test_split(train_size=train_size)
            self.dataset['train'] = split_dataset['train']
            self.dataset['validation'] = split_dataset['test']
            
    def train(self):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('batch_size', 4),
            warmup_steps=self.config.get('warmup_steps', 0),
            learning_rate=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            logging_steps=10,
            evaluation_strategy="steps" if 'validation' in self.dataset else "no",
            eval_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.get('mixed_precision', True),
            gradient_checkpointing=self.config.get('gradient_checkpointing', False),
            dataloader_pin_memory=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset.get('validation'),
            data_collator=data_collator,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    fine_tuner = LLMFineTuner(config)
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.prepare_dataset()
    fine_tuner.train()


if __name__ == "__main__":
    main()
