"""
Abstract base class for sentiment models
"""
from abc import ABC, abstractmethod
from typing import Dict

class BaseSentimentModel(ABC):
    """Base class all sentiment models inherit from"""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for text
        Returns: {'sentiment': str, 'confidence': float, 'probabilities': dict}
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass