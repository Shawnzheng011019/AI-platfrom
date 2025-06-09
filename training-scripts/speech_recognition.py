#!/usr/bin/env python3
"""
Speech Recognition Model Training Script
Supports Wav2Vec2, Whisper fine-tuning, and custom ASR models
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer,
    WhisperForConditionalGeneration, WhisperProcessor,
    TrainingArguments, Trainer
)
import librosa
from datasets import Dataset as HFDataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechDataset(Dataset):
    def __init__(self, audio_paths, transcripts, processor, max_length=16000*30):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        
        # Load audio
        speech, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            speech = resampler(speech)
        
        # Convert to mono if stereo
        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)
        
        speech = speech.squeeze()
        
        # Truncate or pad
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        
        # Process with processor
        inputs = self.processor(
            speech.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process transcript
        with self.processor.as_target_processor():
            labels = self.processor(transcript, return_tensors="pt").input_ids
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': labels.squeeze()
        }


class CustomASRModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6):
        super(CustomASRModel, self).__init__()
        
        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            256, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        
        # CTC head
        self.ctc_head = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        # x shape: (batch, time)
        x = x.unsqueeze(1)  # (batch, 1, time)
        
        # Feature extraction
        features = self.feature_extractor(x)  # (batch, 256, time')
        features = features.transpose(1, 2)  # (batch, time', 256)
        
        # LSTM encoding
        encoded, _ = self.encoder(features)  # (batch, time', hidden*2)
        
        # CTC prediction
        logits = self.ctc_head(encoded)  # (batch, time', vocab_size)
        
        return logits


class SpeechRecognitionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model_type = config.get('model_type', 'wav2vec2').lower()
        
    def load_data(self):
        """Load and preprocess speech data"""
        logger.info(f"Loading data from {self.config['dataset_path']}")
        
        # Load data
        if self.config['dataset_path'].endswith('.csv'):
            data = pd.read_csv(self.config['dataset_path'])
        elif self.config['dataset_path'].endswith('.json'):
            data = pd.read_json(self.config['dataset_path'])
        else:
            raise ValueError("Unsupported file format")
        
        # Expected columns: audio_path, transcript
        required_columns = ['audio_path', 'transcript']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Convert relative paths to absolute
        base_path = os.path.dirname(self.config['dataset_path'])
        data['audio_path'] = data['audio_path'].apply(
            lambda x: os.path.join(base_path, x) if not os.path.isabs(x) else x
        )
        
        # Filter existing files
        data = data[data['audio_path'].apply(os.path.exists)]
        
        if len(data) == 0:
            raise ValueError("No valid audio files found")
        
        # Split data
        train_size = int(len(data) * self.config.get('train_split', 0.8))
        val_size = int(len(data) * self.config.get('val_split', 0.1))
        
        self.train_data = data[:train_size]
        self.val_data = data[train_size:train_size + val_size]
        self.test_data = data[train_size + val_size:]
        
        logger.info(f"Data loaded: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test samples")
        
    def setup_model_and_processor(self):
        """Setup model and processor based on model type"""
        if self.model_type == 'wav2vec2':
            model_name = self.config.get('base_model', 'facebook/wav2vec2-base-960h')
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
        elif self.model_type == 'whisper':
            model_name = self.config.get('base_model', 'openai/whisper-small')
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
        elif self.model_type == 'custom':
            # Create custom vocabulary from transcripts
            all_transcripts = list(self.train_data['transcript']) + list(self.val_data['transcript'])
            vocab = set()
            for transcript in all_transcripts:
                vocab.update(transcript.lower())
            
            vocab = sorted(list(vocab))
            vocab_size = len(vocab) + 1  # +1 for blank token
            
            self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
            self.char_to_idx['<blank>'] = len(vocab)
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            
            # Create custom model
            hidden_dim = self.config.get('hidden_dim', 512)
            num_layers = self.config.get('num_layers', 6)
            self.model = CustomASRModel(vocab_size, hidden_dim, num_layers)
            self.processor = None  # Custom preprocessing
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        logger.info(f"Model setup complete: {self.model_type}")
        
    def create_datasets(self):
        """Create datasets"""
        if self.model_type in ['wav2vec2', 'whisper']:
            self.train_dataset = SpeechDataset(
                self.train_data['audio_path'].tolist(),
                self.train_data['transcript'].tolist(),
                self.processor
            )
            
            self.val_dataset = SpeechDataset(
                self.val_data['audio_path'].tolist(),
                self.val_data['transcript'].tolist(),
                self.processor
            )
        else:
            # Custom dataset handling would go here
            pass
        
    def train(self):
        """Train the model"""
        if self.model_type in ['wav2vec2', 'whisper']:
            self._train_huggingface_model()
        else:
            self._train_custom_model()
    
    def _train_huggingface_model(self):
        """Train using HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config.get('num_epochs', 10),
            per_device_train_batch_size=self.config.get('batch_size', 8),
            per_device_eval_batch_size=self.config.get('batch_size', 8),
            warmup_steps=self.config.get('warmup_steps', 500),
            learning_rate=self.config.get('learning_rate', 1e-4),
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
            tokenizer=self.processor.feature_extractor,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        if self.processor:
            self.processor.save_pretrained(self.config['output_dir'])
        
        logger.info(f"Training completed. Model saved to {self.config['output_dir']}")
    
    def _train_custom_model(self):
        """Train custom model with CTC loss"""
        # This would implement custom training loop for the custom ASR model
        logger.info("Custom model training not fully implemented in this example")
        pass


def main():
    parser = argparse.ArgumentParser(description='Train Speech Recognition Model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = SpeechRecognitionTrainer(config)
    trainer.load_data()
    trainer.setup_model_and_processor()
    trainer.create_datasets()
    trainer.train()


if __name__ == "__main__":
    main()
