from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTSentimentTrainer:
    """
    Fine-tune BERT for 5-class sentiment analysis
    Maps ratings 1-5 to terrible/negative/neutral/positive/amazing
    """
    
    def __init__(self, 
                 model_name="nlptown/bert-base-multilingual-uncased-sentiment",
                 output_dir="./bert-finetuned"):
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        self.sentiment_labels = ['terrible', 'negative', 'neutral', 'positive', 'amazing']
        self.num_labels = 5
        
        logger.info(f"Initialized BERT trainer with model: {model_name}")
    
    def prepare_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Convert numpy arrays to HuggingFace datasets
        
        Args:
            X_train, y_train: Training text and labels (ratings 1-5)
            X_val, y_val: Validation text and labels
            X_test, y_test: Test text and labels
        """
        logger.info("Preparing datasets for BERT...")
        
        # Convert ratings (1-5) to labels (0-4)
        y_train_labels = [int(rating) - 1 for rating in y_train]
        y_val_labels = [int(rating) - 1 for rating in y_val]
        y_test_labels = [int(rating) - 1 for rating in y_test]
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': X_train,
            'label': y_train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': X_val,
            'label': y_val_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': X_test,
            'label': y_test_labels
        })
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_dataset(self, dataset):
        """Tokenize text for BERT"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics during training"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_dataset, val_dataset, 
              epochs=3, 
              batch_size=16,
              learning_rate=2e-5,
              warmup_steps=500):
        """
        Fine-tune BERT on your data
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
        """
        logger.info("Starting BERT fine-tuning...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        val_dataset = self.tokenize_dataset(val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        logger.info("Training started...")
        train_result = self.trainer.train()
        
        logger.info("✓ Training completed!")
        logger.info(f"Best model saved to: {self.output_dir}")
        
        return train_result
    
    def evaluate(self, test_dataset):
        """Evaluate on test set"""
        
        if not self.trainer:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Evaluating on test set...")
        
        # Tokenize test set
        test_dataset = self.tokenize_dataset(test_dataset)
        
        # Evaluate
        test_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions for confusion matrix
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=-1)
        y_true = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_labels)
        )
        
        results = {
            'test_accuracy': test_results['eval_accuracy'],
            'test_f1': test_results['eval_f1'],
            'test_precision': test_results['eval_precision'],
            'test_recall': test_results['eval_recall'],
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.sentiment_labels[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(self.num_labels)
            }
        }
        
        logger.info("="*60)
        logger.info("TEST SET RESULTS")
        logger.info("="*60)
        logger.info(f"Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Weighted F1: {results['test_f1']:.4f}")
        logger.info(f"Precision: {results['test_precision']:.4f}")
        logger.info(f"Recall: {results['test_recall']:.4f}")
        logger.info("="*60)
        
        return results
    
    def save_model(self, save_path=None):
        """Save fine-tuned model"""
        
        if save_path is None:
            save_path = self.output_dir
        else:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"✓ Model saved to {save_path}")
    
    def predict(self, text):
        """Quick prediction on single text"""
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        pred_class = torch.argmax(probs).item()
        confidence = probs[0][pred_class].item()
        
        return {
            'sentiment': self.sentiment_labels[pred_class],
            'confidence': confidence,
            'rating': pred_class + 1
        }


def main():
    """Test BERT trainer"""
    import sys
    sys.path.append('..')
    from data_loader import prepare_data_for_training
    
    # Load data
    data_splits, stats = prepare_data_for_training()
    X_train, y_train, _ = data_splits['train']
    X_val, y_val, _ = data_splits['val']
    X_test, y_test, _ = data_splits['test']
    
    # Initialize trainer
    trainer = BERTSentimentTrainer(output_dir='../models/bert-finetuned')
    
    # Prepare datasets
    train_ds, val_ds, test_ds = trainer.prepare_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Train
    trainer.train(train_ds, val_ds, epochs=3, batch_size=16)
    
    # Evaluate
    results = trainer.evaluate(test_ds)
    
    # Save
    trainer.save_model('../models/bert-finetuned')
    
    # Save results
    import json
    with open('../results/bert_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ BERT training complete!")
    print(f"Test F1: {results['test_f1']:.4f}")
    print(f"Results saved to results/bert_training_results.json")


if __name__ == "__main__":
    main()