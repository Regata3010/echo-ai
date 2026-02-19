"""
Sarcasm Detection Module
Detects sarcastic reviews that appear positive but are actually negative
"""
import logging
from typing import Dict, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SarcasmDetector:
    """
    Detects sarcasm in review text
    Combines pattern matching + AI model
    """
    
    def __init__(self):
        # Sarcasm indicators (heuristic patterns)
        self.sarcasm_patterns = [
            r'\boh (great|wonderful|fantastic|amazing)\b',
            r'\bjust (great|wonderful|perfect)\b',
            r'\byeah,? (right|sure)\b',
            r'\bof course\b.*\bnot\b',
            r'\bobviously\b',
            r'\bclearly\b.*\bnot\b'
        ]
        
        # Context clues for sarcasm
        self.negative_context = [
            'cold', 'waited', 'hour', 'terrible', 'worst', 'never', 'again',
            'disappointed', 'waste', 'avoid', 'regret', 'mistake'
        ]
        
        # Try to load AI model
        self.use_ai_model = self._load_ai_model()
    
    def _load_ai_model(self):
        """
        Load pre-trained sarcasm detection model
        Falls back to pattern-based if model unavailable
        """
        try:
            from transformers import pipeline
            
            self.ai_detector = pipeline(
                "text-classification",
                model="helinivan/english-sarcasm-detector"
            )
            logger.info("Sarcasm AI model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not load sarcasm AI model: {e}")
            logger.info("Using pattern-based sarcasm detection")
            return False
    
    def detect_pattern_based(self, text: str) -> Tuple[bool, float]:
        """
        Pattern-based sarcasm detection (fallback)
        """
        text_lower = text.lower()
        
        # Check for sarcasm patterns
        pattern_matches = sum(1 for pattern in self.sarcasm_patterns 
                             if re.search(pattern, text_lower))
        
        # Check for negative context with positive words
        has_positive_words = any(word in text_lower for word in ['great', 'wonderful', 'perfect', 'amazing'])
        has_negative_context = sum(1 for word in self.negative_context if word in text_lower)
        
        # Heuristic scoring
        sarcasm_score = 0.0
        
        if pattern_matches > 0:
            sarcasm_score += 0.4 * pattern_matches
        
        if has_positive_words and has_negative_context >= 2:
            sarcasm_score += 0.5
        
        # Exclamation marks with negative context
        exclamations = text.count('!')
        if exclamations > 0 and has_negative_context > 0:
            sarcasm_score += 0.2
        
        is_sarcastic = sarcasm_score > 0.6
        
        return is_sarcastic, min(sarcasm_score, 1.0)
    
    def detect_ai_based(self, text: str) -> Tuple[bool, float]:
        """
        AI model-based sarcasm detection
        """
        try:
            result = self.ai_detector(text)[0]
            
            is_sarcastic = result['label'].lower() == 'sarcasm' and result['score'] > 0.65
            confidence = result['score'] if result['label'].lower() == 'sarcasm' else 1 - result['score']
            
            return is_sarcastic, confidence
        
        except Exception as e:
            logger.error(f"AI sarcasm detection failed: {e}")
            return self.detect_pattern_based(text)
    
    def detect(self, text: str) -> Dict:
        """
        Main detection method - use pattern-based (AI model is unreliable)
        """
        
        # Use pattern-based detection (more reliable)
        is_sarcastic, confidence = self.detect_pattern_based(text)
        method = 'pattern_based'
        
        # Boost confidence if both methods agree
        if self.use_ai_model and confidence > 0.5:
            try:
                ai_sarcastic, ai_conf = self.detect_ai_based(text)
                if ai_sarcastic == is_sarcastic:
                    confidence = min((confidence + ai_conf) / 2, 1.0)
                    method = 'pattern_and_ai'
            except:
                pass
        
        return {
            'is_sarcastic': is_sarcastic,
            'confidence': round(confidence, 3),
            'method': method
        }
    
    def correct_sentiment(self, text: str, predicted_sentiment: str) -> Dict:
        """
        Correct sentiment if sarcasm detected
        
        Args:
            text: Review text
            predicted_sentiment: Initial sentiment prediction
        
        Returns:
            Corrected sentiment and sarcasm info
        """
        
        sarcasm_result = self.detect(text)
        
        corrected_sentiment = predicted_sentiment
        
        # If sarcastic and predicted positive, flip to negative
        if sarcasm_result['is_sarcastic']:
            if predicted_sentiment in ['positive', 'amazing']:
                corrected_sentiment = 'negative'
                logger.info(f"Sarcasm detected - flipped {predicted_sentiment} to negative")
            elif predicted_sentiment in ['negative', 'terrible']:
                # Sarcastic negative might actually be positive (rare)
                pass
        
        return {
            'original_sentiment': predicted_sentiment,
            'corrected_sentiment': corrected_sentiment,
            'sarcasm_detected': sarcasm_result['is_sarcastic'],
            'sarcasm_confidence': sarcasm_result['confidence'],
            'was_corrected': corrected_sentiment != predicted_sentiment
        }


def test_sarcasm_detector():
    """Test sarcasm detection"""
    
    detector = SarcasmDetector()
    
    test_cases = [
        ("Oh great, another cold burger. Just wonderful.", True),
        ("The food was truly amazing!", False),
        ("Yeah right, best service ever.", True),
        ("Excellent food and service!", False),
    ]
    
    print("Sarcasm Detection Test")
    print("="*60)
    
    for text, expected_sarcastic in test_cases:
        result = detector.detect(text)
        match = "PASS" if result['is_sarcastic'] == expected_sarcastic else "FAIL"
        
        print(f"\n{text}")
        print(f"  Sarcastic: {result['is_sarcastic']} (confidence: {result['confidence']:.2f})")
        print(f"  Expected: {expected_sarcastic} [{match}]")


if __name__ == "__main__":
    test_sarcasm_detector()