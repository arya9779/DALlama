"""Sentiment analysis models with LoRA fine-tuning support."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from loguru import logger

from ..config import config

class SentimentDataset:
    """Dataset class for sentiment analysis."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """Prepare dataset for training."""
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Convert labels to integers
        label_ids = [self.label_map[label] for label in labels]
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": label_ids
        })
        
        return dataset

class FinBERTBaseline:
    """FinBERT baseline model for comparison."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dataset_handler = None
        
    def load_model(self):
        """Load pre-trained FinBERT model."""
        
        logger.info(f"Loading FinBERT model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=config.model.num_labels,
            problem_type="single_label_classification"
        )
        
        self.dataset_handler = SentimentDataset(self.tokenizer)
        
        logger.info("FinBERT model loaded successfully")
    
    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: List[str], val_labels: List[str]) -> Dict:
        """Train FinBERT model."""
        
        if not self.model:
            self.load_model()
        
        # Prepare datasets
        train_dataset = self.dataset_handler.prepare_dataset(train_texts, train_labels)
        val_dataset = self.dataset_handler.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.training.output_dir / "finbert_baseline",
            num_train_epochs=config.model.num_epochs,
            per_device_train_batch_size=config.model.batch_size,
            per_device_eval_batch_size=config.model.batch_size,
            gradient_accumulation_steps=config.model.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.model.weight_decay,
            logging_dir=config.training.logging_dir,
            logging_steps=config.training.logging_steps,
            evaluation_strategy=config.training.eval_strategy,
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            fp16=config.training.fp16,
            dataloader_num_workers=config.training.dataloader_num_workers,
            seed=config.training.seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience,
                early_stopping_threshold=config.training.early_stopping_threshold
            )]
        )
        
        # Train model
        logger.info("Starting FinBERT training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info("FinBERT training completed")
        
        return train_result.metrics
    
    def _compute_metrics(self, eval_pred) -> Dict:
        """Compute evaluation metrics."""
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
        }
        
        # Add per-class metrics
        for i, label in enumerate(["negative", "neutral", "positive"]):
            metrics[f"f1_{label}"] = f1_per_class[i]
            metrics[f"precision_{label}"] = precision_per_class[i]
            metrics[f"recall_{label}"] = recall_per_class[i]
        
        return metrics

class LoRASentimentModel:
    """LoRA fine-tuned model for sentiment analysis."""
    
    def __init__(self, base_model_name: str = None):
        self.base_model_name = base_model_name or config.model.base_model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.dataset_handler = None
        
    def load_base_model(self):
        """Load base model and apply LoRA configuration."""
        
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if using QLoRA
        model_kwargs = {}
        if config.model.use_qlora:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=config.model.num_labels,
            problem_type="single_label_classification",
            torch_dtype=torch.float16 if config.training.fp16 else torch.float32,
            **model_kwargs
        )
        
        # Configure model for training
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.dataset_handler = SentimentDataset(self.tokenizer)
        
        logger.info("Base model loaded successfully")
    
    def apply_lora(self):
        """Apply LoRA configuration to the model."""
        
        if not config.model.use_lora:
            logger.info("LoRA not enabled, using full fine-tuning")
            return
        
        logger.info("Applying LoRA configuration...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: List[str], val_labels: List[str]) -> Dict:
        """Train the LoRA model."""
        
        if not self.model:
            self.load_base_model()
            self.apply_lora()
        
        # Use LoRA model if available, otherwise base model
        model_to_train = self.peft_model if self.peft_model else self.model
        
        # Prepare datasets
        train_dataset = self.dataset_handler.prepare_dataset(train_texts, train_labels)
        val_dataset = self.dataset_handler.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        output_dir = config.training.output_dir / config.get_model_name()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.model.num_epochs,
            per_device_train_batch_size=config.model.batch_size,
            per_device_eval_batch_size=config.model.batch_size,
            gradient_accumulation_steps=config.model.gradient_accumulation_steps,
            learning_rate=config.model.learning_rate,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.model.weight_decay,
            max_grad_norm=config.model.max_grad_norm,
            logging_dir=config.training.logging_dir,
            logging_steps=config.training.logging_steps,
            evaluation_strategy=config.training.eval_strategy,
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            fp16=config.training.fp16,
            dataloader_num_workers=config.training.dataloader_num_workers,
            seed=config.training.seed,
            report_to="wandb" if config.experiment.use_wandb else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience,
                early_stopping_threshold=config.training.early_stopping_threshold
            )]
        )
        
        # Train model
        logger.info("Starting LoRA model training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapters separately if using LoRA
        if self.peft_model:
            self.peft_model.save_pretrained(output_dir / "lora_adapters")
        
        logger.info("LoRA model training completed")
        
        return train_result.metrics
    
    def _compute_metrics(self, eval_pred) -> Dict:
        """Compute evaluation metrics."""
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
        }
        
        # Add per-class metrics
        for i, label in enumerate(["negative", "neutral", "positive"]):
            metrics[f"f1_{label}"] = f1_per_class[i]
            metrics[f"precision_{label}"] = precision_per_class[i]
            metrics[f"recall_{label}"] = recall_per_class[i]
        
        return metrics
    
    def load_trained_model(self, model_path: Path):
        """Load a trained model for inference."""
        
        logger.info(f"Loading trained model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load LoRA adapters if they exist
        lora_path = model_path / "lora_adapters"
        if lora_path.exists():
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info("LoRA adapters loaded")
        
        self.dataset_handler = SentimentDataset(self.tokenizer)
        
        logger.info("Trained model loaded successfully")
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Make predictions on new texts."""
        
        model_to_use = self.peft_model if self.peft_model else self.model
        
        if not model_to_use:
            raise ValueError("No model loaded. Call load_trained_model() first.")
        
        model_to_use.eval()
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=config.data.max_text_length,
                    return_tensors="pt"
                )
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    model_to_use = model_to_use.cuda()
                
                # Predict
                outputs = model_to_use(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs.max().item()
                
                # Convert to label
                label = self.dataset_handler.reverse_label_map[predicted_class]
                
                predictions.append({
                    "text": text,
                    "predicted_label": label,
                    "confidence": confidence,
                    "probabilities": {
                        "negative": probs[0][0].item(),
                        "neutral": probs[0][1].item(),
                        "positive": probs[0][2].item()
                    }
                })
        
        return predictions

class ModelComparison:
    """Compare different model approaches."""
    
    def __init__(self):
        self.finbert_baseline = FinBERTBaseline()
        self.lora_model = LoRASentimentModel()
        self.results = {}
    
    def run_comparison(self, train_texts: List[str], train_labels: List[str],
                      val_texts: List[str], val_labels: List[str],
                      test_texts: List[str], test_labels: List[str]) -> Dict:
        """Run comparison between models."""
        
        logger.info("Starting model comparison...")
        
        # Train FinBERT baseline
        logger.info("Training FinBERT baseline...")
        finbert_metrics = self.finbert_baseline.train(
            train_texts, train_labels, val_texts, val_labels
        )
        self.results["finbert_baseline"] = finbert_metrics
        
        # Train LoRA model
        logger.info("Training LoRA model...")
        lora_metrics = self.lora_model.train(
            train_texts, train_labels, val_texts, val_labels
        )
        self.results["lora_model"] = lora_metrics
        
        # Evaluate on test set
        # (Implementation would include test set evaluation)
        
        logger.info("Model comparison completed")
        
        return self.results
    
    def save_comparison_results(self, output_path: Path):
        """Save comparison results."""
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_path}")

# Utility functions
def load_model_for_inference(model_path: Path) -> LoRASentimentModel:
    """Load a trained model for inference."""
    
    model = LoRASentimentModel()
    model.load_trained_model(model_path)
    return model

def batch_predict(model: LoRASentimentModel, texts: List[str], 
                 batch_size: int = 32) -> List[Dict]:
    """Make predictions in batches for efficiency."""
    
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = model.predict(batch_texts)
        all_predictions.extend(batch_predictions)
    
    return all_predictions