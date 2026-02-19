"""
BERT Training Module for EchoAI
Orchestrates BERT fine-tuning and evaluation
"""
import logging
import json
from pathlib import Path
from datetime import datetime

from sentiment.bert_trainer import BERTSentimentTrainer
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTTrainingPipeline:
    """
    Orchestrates BERT model training
    """
    
    def __init__(self):
        self.trainer = None
        self.results = {}
        self.model_dir = MODEL_DIR / 'bert-finetuned'
        
    def train_bert_model(self, data_splits, epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train BERT model on your data
        
        Args:
            data_splits: Dict with train/val/test splits from data_loader
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        
        logger.info("="*70)
        logger.info("  BERT FINE-TUNING")
        logger.info("="*70)
        
        try:
            # Extract data
            X_train, y_train, _ = data_splits['train']
            X_val, y_val, _ = data_splits['val']
            X_test, y_test, _ = data_splits['test']
            
            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Validation samples: {len(X_val)}")
            logger.info(f"Test samples: {len(X_test)}")
            
            # Initialize trainer
            self.trainer = BERTSentimentTrainer(output_dir=str(self.model_dir))
            
            # Prepare datasets
            logger.info("Preparing datasets for BERT...")
            train_dataset, val_dataset, test_dataset = self.trainer.prepare_datasets(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Train
            logger.info(f"Starting fine-tuning (epochs={epochs}, batch_size={batch_size})...")
            logger.info("This may take 30-60 minutes depending on your hardware...")
            
            train_result = self.trainer.train(
                train_dataset, 
                val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = self.trainer.evaluate(test_dataset)
            
            # Save model
            logger.info("Saving fine-tuned model...")
            self.trainer.save_model(self.model_dir)
            
            # Store results
            self.results = {
                'status': 'success',
                'model_name': 'BERT-finetuned',
                'base_model': self.trainer.model_name,
                'training_args': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                },
                'data_sizes': {
                    'train': len(X_train),
                    'val': len(X_val),
                    'test': len(X_test)
                },
                'test_metrics': test_results,
                'model_path': str(self.model_dir),
                'trained_at': datetime.now().isoformat()
            }
            
            # Save results
            results_file = RESULTS_DIR / 'bert_training_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info("="*70)
            logger.info("  BERT TRAINING COMPLETE")
            logger.info("="*70)
            logger.info(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_results['test_f1']:.4f}")
            logger.info(f"Model saved to: {self.model_dir}")
            logger.info(f"Results saved to: {results_file}")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"BERT training failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def quick_test(self):
        """Quick test of trained model"""
        
        if not self.trainer:
            raise ValueError("Model not trained yet")
        
        test_cases = [
            "Terrible food and service",
            "Disappointing experience",
            "It was okay",
            "Good food!",
            "Amazing! Best ever!"
        ]
        
        logger.info("\nQuick Test:")
        logger.info("-"*60)
        
        for text in test_cases:
            result = self.trainer.predict(text)
            logger.info(f"{text:40} → {result['sentiment']:8} ({result['confidence']:.2f})")


def main():
    """Main function for standalone BERT training"""
    from data_loader import prepare_data_for_training
    
    logger.info("Loading data...")
    data_splits, stats = prepare_data_for_training()
    
    logger.info(f"Dataset statistics: {stats}")
    
    # Train BERT
    pipeline = BERTTrainingPipeline()
    results = pipeline.train_bert_model(
        data_splits,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Quick test
    pipeline.quick_test()
    
    logger.info("\n✓ BERT training pipeline completed successfully!")
    
    return results


if __name__ == "__main__":
    main()  