"""
BERT-based sentiment analysis model
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BERTSentimentModel:
    """
    BERT model for 5-class sentiment analysis
    """
    
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentiment_labels = ['terrible', 'negative', 'neutral', 'positive', 'amazing']
    
    def load_pretrained(self):
        """Load pre-trained model (no fine-tuning)"""
        logger.info(f"Loading pre-trained BERT: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        logger.info("✓ BERT loaded")
    
    def predict(self, text: str):
        """Predict sentiment for single review"""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get results
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][pred_class].item()
        
        return {
            'sentiment': self.sentiment_labels[pred_class],
            'sentiment_score': pred_class + 1,  # 1-5 rating
            'confidence': confidence,
            'probabilities': {
                label: probabilities[0][i].item()
                for i, label in enumerate(self.sentiment_labels)
            }
        }
    
    def save(self, path: str):
        """Save model to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        load_path = Path(path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        
        logger.info(f"✓ Model loaded from {path}")