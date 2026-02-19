"""
Enhanced Inference Pipeline for EchoAI
Combines BERT sentiment + Enhanced ABSA + Smart responses
"""
import joblib
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path

from config import *
from response_generator import ResponseGenerator
from AspectSA.hybrid_absa import HybridABSA
from advanced.sarcasm_detector import SarcasmDetector
from advanced.critical_issues import CriticalIssueDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInferencePipeline:
    """
    Complete inference pipeline with BERT + Enhanced ABSA
    """
    
    def __init__(self, 
                 sentiment_model_path: Path = None,
                 vectorizer_path: Path = None,
                 llm_model: str = 'google/flan-t5-base',
                 use_bert: bool = True,
                 use_enhanced_absa: bool = True):
        
        self.sentiment_model_path = sentiment_model_path or BEST_MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        self.llm_model_name = llm_model
        self.use_bert = use_bert
        self.use_enhanced_absa = use_enhanced_absa
        
        self.sentiment_model = None
        self.vectorizer = None
        self.bert_model = None
        self.absa_analyzer = None
        self.response_generator = None
        self.sarcasm_detector = None
        self.critical_issues = None
        
        self.sentiment_labels = ['terrible', 'negative', 'neutral', 'positive', 'amazing']
        
        self.inference_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0
        }
    
    def load_models(self, load_llm: bool = True):
        """Load all required models"""
        logger.info("Loading models for inference...")
        
        # Load sentiment model
        if self.use_bert:
            try:
                from sentiment.bert_model import BERTSentimentModel
                self.bert_model = BERTSentimentModel()
                
                finetuned_path = MODEL_DIR / 'bert-finetuned'
                if finetuned_path.exists():
                    logger.info("Loading fine-tuned BERT model...")
                    self.bert_model.load(str(finetuned_path))
                else:
                    logger.info("Loading pre-trained BERT model...")
                    self.bert_model.load_pretrained()
                
                logger.info("BERT sentiment model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BERT: {e}")
                raise
        else:
            try:
                self.sentiment_model = joblib.load(self.sentiment_model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info("TF-IDF sentiment model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF: {e}")
                raise
        
        # Load Enhanced ABSA
        if self.use_enhanced_absa:
            try:
                self.absa_analyzer = HybridABSA()
                logger.info("Enhanced ABSA loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ABSA: {e}")
                self.absa_analyzer = None
        
        # Load Advanced Detectors
        try:
            from advanced.sarcasm_detector import SarcasmDetector
            from advanced.critical_issues import CriticalIssueDetector
            
            self.sarcasm_detector = SarcasmDetector()
            self.critical_detector = CriticalIssueDetector()
            logger.info("Advanced detectors loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load advanced detectors: {e}")
            self.sarcasm_detector = None
            self.critical_detector = None
        
        # Load LLM
        if load_llm:
            try:
                self.response_generator = ResponseGenerator(self.llm_model_name)
                self.response_generator.load_model()
                logger.info("Response generation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                self.response_generator = None
    def predict_sentiment(self, text: str) -> Dict:
        """Predict overall sentiment"""
        
        try:
            if self.use_bert:
                if not self.bert_model:
                    raise ValueError("BERT model not loaded")
                return self.bert_model.predict(text)
            else:
                if not self.sentiment_model or not self.vectorizer:
                    raise ValueError("Sentiment model not loaded")
                
                text_tfidf = self.vectorizer.transform([text])
                prediction = self.sentiment_model.predict(text_tfidf)[0]
                
                prediction_int = int(prediction)
                if prediction_int >= 1 and prediction_int <= 5:
                    sentiment_idx = prediction_int - 1
                else:
                    sentiment_idx = prediction_int
                
                sentiment_idx = max(0, min(4, sentiment_idx))
                sentiment_label = self.sentiment_labels[sentiment_idx]
                
                confidence = None
                if hasattr(self.sentiment_model, 'predict_proba'):
                    probabilities = self.sentiment_model.predict_proba(text_tfidf)[0]
                    confidence = float(max(probabilities))
                    class_probabilities = {
                        label: float(prob) 
                        for label, prob in zip(self.sentiment_labels, probabilities)
                    }
                else:
                    class_probabilities = {}
                
                return {
                    'sentiment': sentiment_label,
                    'sentiment_score': prediction_int,
                    'confidence': confidence,
                    'probabilities': class_probabilities
                }
        
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise
    
    def _generate_aspect_aware_response(self, sentiment: str, absa_analysis: Dict) -> str:
        """Generate response addressing specific aspects"""
        
        patterns = absa_analysis['patterns']
        negative_aspects = patterns['negative_aspects']
        positive_aspects = patterns['positive_aspects']
        has_conflict = patterns['has_conflict']
        
        response_parts = []
        
        # Opening
        if sentiment in ['terrible', 'negative'] or len(negative_aspects) >= 2:
            response_parts.append("We sincerely apologize for your experience.")
        elif sentiment in ['amazing', 'positive']:
            response_parts.append("Thank you for your wonderful feedback!")
        else:
            response_parts.append("Thank you for taking the time to share your thoughts.")
        
        # Address negatives specifically
        if negative_aspects:
            if len(negative_aspects) == 1:
                response_parts.append(f"We are particularly concerned about the {negative_aspects[0]} issues you mentioned and will address them immediately.")
            else:
                aspects_list = ', '.join(negative_aspects[:-1]) + f' and {negative_aspects[-1]}'
                response_parts.append(f"We are particularly concerned about the {aspects_list} issues you experienced and will work to improve in these areas.")
        
        # Acknowledge positives if there are also negatives
        if positive_aspects and (has_conflict or sentiment in ['terrible', 'negative']):
            if len(positive_aspects) == 1:
                response_parts.append(f"We are glad you appreciated our {positive_aspects[0]}.")
            else:
                aspects_list = ' and '.join(positive_aspects[:2])
                response_parts.append(f"We are pleased you enjoyed our {aspects_list}.")
        
        # Closing
        if sentiment in ['terrible', 'negative'] or len(negative_aspects) >= 2:
            response_parts.append("Please contact our manager directly so we can make this right and restore your faith in our service.")
        elif negative_aspects:
            response_parts.append("We value your feedback and will use it to improve.")
        else:
            response_parts.append("We look forward to welcoming you back soon!")
        
        return " ".join(response_parts)
    
    def _get_template_response(self, sentiment: str) -> str:
        """Fallback template responses"""
        templates = {
            'amazing': "We are absolutely thrilled by your amazing review! Your incredible feedback means everything to us, and we cannot wait to exceed your expectations again.",
            'positive': "Thank you for your positive feedback! We are delighted to hear about your experience and look forward to serving you again.",
            'neutral': "Thank you for taking the time to share your feedback. We value your input and are always working to improve our service.",
            'negative': "We sincerely apologize for your experience. Your feedback is important to us, and we would like to make things right. Please contact us directly.",
            'terrible': "We are deeply sorry for the completely unacceptable experience you had. Please contact our management immediately so we can resolve this urgently."
        }
        return templates.get(sentiment, templates['neutral'])
    
    def process_review(self, 
                   review: Union[str, Dict],
                   generate_response: bool = True) -> Dict:
        """
        Process review through complete enhanced pipeline
        Now includes: BERT + ABSA + Sarcasm + Critical Issues
        """
        # Parse input
        if isinstance(review, str):
            review_text = review
            metadata = {}
        else:
            review_text = review.get('reviewText', review.get('text', ''))
            metadata = {
                'placeName': review.get('placeName'),
                'placeAddress': review.get('placeAddress'),
                'provider': review.get('provider'),
                'reviewRating': review.get('reviewRating'),
                'authorName': review.get('authorName'),
                'reviewDate': review.get('reviewDate')
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
        
        result = {
            'input': review_text,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # STEP 1: Check for Critical Issues (highest priority)
            critical_result = None
            if self.critical_detector:
                critical_result = self.critical_detector.detect(review_text)
                result['critical_issues'] = critical_result
            
            # STEP 2: Overall Sentiment (BERT)
            sentiment_result = self.predict_sentiment(review_text)
            
            # STEP 3: Sarcasm Detection and Correction
            if self.sarcasm_detector:
                sarcasm_result = self.sarcasm_detector.correct_sentiment(
                    review_text,
                    sentiment_result['sentiment']
                )
                result['sarcasm_analysis'] = sarcasm_result
                
                # Apply correction if sarcasm detected
                if sarcasm_result['was_corrected']:
                    sentiment_result['sentiment'] = sarcasm_result['corrected_sentiment']
                    sentiment_result['sarcasm_corrected'] = True
                    logger.info(f"Sarcasm corrected: {sarcasm_result['original_sentiment']} -> {sarcasm_result['corrected_sentiment']}")
            
            # STEP 4: Override Sentiment if Critical Issue Detected
            if critical_result and critical_result['should_override_sentiment']:
                original_sentiment = sentiment_result['sentiment']
                sentiment_result['sentiment'] = 'terrible'
                sentiment_result['overridden_by_critical_issue'] = True
                logger.warning(f"Critical issue detected - forcing sentiment to terrible (was {original_sentiment})")
            
            result['sentiment_analysis'] = sentiment_result
            
            # STEP 5: Aspect-Based Analysis
            if self.absa_analyzer:
                absa_result = self.absa_analyzer.analyze(
                    review_text,
                    overall_sentiment=sentiment_result['sentiment']
                )
                result['aspect_analysis'] = absa_result
            else:
                absa_result = None
            
            # STEP 6: Generate Response
            if generate_response:
                # Special handling for critical issues
                if critical_result and critical_result['has_critical_issue']:
                    response = self._generate_critical_issue_response(
                        sentiment_result['sentiment'],
                        absa_result,
                        critical_result
                    )
                # Aspect-aware response for mixed sentiment
                elif absa_result and absa_result['aspects_mentioned']:
                    response = self._generate_aspect_aware_response(
                        sentiment_result['sentiment'],
                        absa_result
                    )
                # Simple template fallback
                else:
                    response = self._get_template_response(sentiment_result['sentiment'])
                
                result['generated_response'] = response
            
            # Update stats
            self.inference_stats['total_processed'] += 1
            self.inference_stats['successful'] += 1
            if sentiment_result.get('confidence'):
                self.inference_stats['avg_confidence'] = (
                    (self.inference_stats['avg_confidence'] * 
                    (self.inference_stats['successful'] - 1) +
                    sentiment_result['confidence']) / 
                    self.inference_stats['successful']
                )
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            self.inference_stats['failed'] += 1
        
        return result
    
    def _generate_critical_issue_response(self, sentiment: str, absa_analysis: Dict, critical_result: Dict) -> str:

        issue_categories = [issue['category'] for issue in critical_result['issues']]
        response_parts = []

        # Tone varies if sentiment wasn't already terrible before override
        if sentiment == 'terrible':
            response_parts.append("We are extremely concerned about your experience and take these issues very seriously.")
        else:
            response_parts.append("Despite some positive aspects of your visit, the issues you raised require our immediate attention.")

        if 'health_violation' in issue_categories:
            response_parts.append("The health and safety matter you mentioned is unacceptable and requires immediate investigation.")

        if 'safety' in issue_categories:
            response_parts.append("Your safety concern is our top priority and we will address this immediately.")

        if 'discrimination' in issue_categories:
            response_parts.append("We have zero tolerance for discrimination and will investigate this matter urgently.")

        response_parts.append("Please contact our management team via the contact form so we can address this situation personally and make it right.")

        return " ".join(response_parts)
    
    def process_batch(self, reviews: List[Union[str, Dict]],
                     generate_responses: bool = True) -> List[Dict]:
        """Process multiple reviews"""
        
        logger.info(f"Processing batch of {len(reviews)} reviews...")
        
        results = []
        for i, review in enumerate(reviews, 1):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(reviews)} reviews")
            
            result = self.process_review(review, generate_responses)
            results.append(result)
        
        return results


def main():
    """Test the enhanced pipeline"""
    
    pipeline = EnhancedInferencePipeline(use_bert=True, use_enhanced_absa=True)
    pipeline.load_models(load_llm=False)
    
    test_reviews = [
        "The food was good. Service was not great particularly.",
        "Amazing ambiance and delicious food! Will definitely return!",
        "Terrible everything. Dirty place, rude staff, cold food.",
        "Great food but way overpriced for the portion size."
    ]
    
    print("="*70)
    print("ENHANCED INFERENCE PIPELINE TEST")
    print("="*70)
    
    for review in test_reviews:
        print(f"\nReview: {review}")
        print("-"*70)
        
        result = pipeline.process_review(review, generate_response=True)
        
        if result['status'] == 'success':
            print(f"Overall: {result['sentiment_analysis']['sentiment']} ({result['sentiment_analysis']['confidence']:.2f})")
            
            if 'aspect_analysis' in result:
                absa = result['aspect_analysis']
                print(f"Mixed sentiment: {absa['has_mixed_sentiment']}")
                print(f"Pattern: {absa['patterns']['pattern']}")
                
                print("\nAspects:")
                for aspect, data in absa['aspect_sentiments'].items():
                    if data['sentiment'] != 'not_mentioned':
                        print(f"  {aspect}: {data['sentiment']} (score: {data['score']})")
            
            print(f"\nResponse:\n{result['generated_response']}")


if __name__ == "__main__":
    main()