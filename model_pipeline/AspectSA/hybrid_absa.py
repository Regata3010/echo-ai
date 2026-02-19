"""
Hybrid ABSA: Combines Rule-Based + BERT Approaches
Provides comprehensive aspect-level sentiment analysis
"""
from .bert_absa import BERTAspectSentiment
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridABSA:
    """
    Hybrid approach combining:
    1. Rule-based aspect detection (fast, reliable)
    2. BERT-based sentiment scoring (accurate, context-aware)
    """
    
    def __init__(self):
        self.bert_absa = BERTAspectSentiment()
        self.aspects = ['food', 'service', 'ambiance', 'price', 'cleanliness']
    
    def analyze(self, review_text: str, overall_sentiment: str = None) -> Dict:
        """
        Complete hybrid ABSA analysis
        
        Args:
            review_text: The review to analyze
            overall_sentiment: Overall sentiment from BERT (optional)
        
        Returns:
            Comprehensive aspect analysis with conflict detection
        """
        
        # Step 1: Extract aspect sentences
        aspect_sentences = self.bert_absa.extract_aspect_sentences(review_text)
        
        # Step 2: Analyze sentiment for each aspect
        aspect_results = {}
        
        for aspect in self.aspects:
            if aspect in aspect_sentences:
                # Aspect is mentioned - analyze its sentiment
                sentiment_result = self.bert_absa.analyze_aspect_sentiment(
                    aspect_sentences[aspect]
                )
                aspect_results[aspect] = sentiment_result
            else:
                # Aspect not mentioned
                aspect_results[aspect] = {
                    'sentiment': 'not_mentioned',
                    'score': 0,
                    'confidence': 0,
                    'num_mentions': 0
                }
        
        # Step 3: Detect patterns and conflicts
        analysis = self._analyze_patterns(aspect_results, overall_sentiment)
        
        return {
            'aspect_sentiments': aspect_results,
            'patterns': analysis,
            'aspects_mentioned': [k for k, v in aspect_results.items() if v['sentiment'] != 'not_mentioned'],
            'has_mixed_sentiment': analysis['has_conflict'],
            'dominant_aspect': analysis['dominant_aspect'],
            'priority_aspects': analysis['priority_aspects']
        }
    
    def _analyze_patterns(self, aspect_results: Dict, overall_sentiment: str = None) -> Dict:
        """
        Detect patterns in aspect sentiments
        """
        
        mentioned_aspects = {
            k: v for k, v in aspect_results.items() 
            if v['sentiment'] != 'not_mentioned'
        }
        
        if not mentioned_aspects:
            return {
                'has_conflict': False,
                'dominant_aspect': None,
                'priority_aspects': [],
                'pattern': 'no_aspects_mentioned'
            }
        
        # Categorize aspects by sentiment
        positive_aspects = [k for k, v in mentioned_aspects.items() if v['sentiment'] == 'positive']
        negative_aspects = [k for k, v in mentioned_aspects.items() if v['sentiment'] == 'negative']
        neutral_aspects = [k for k, v in mentioned_aspects.items() if v['sentiment'] == 'neutral']
        
        # Detect conflicts
        has_conflict = len(positive_aspects) > 0 and len(negative_aspects) > 0
        
        # Find dominant aspect (most strongly expressed)
        if mentioned_aspects:
            dominant_aspect = max(
                mentioned_aspects.items(),
                key=lambda x: abs(x[1]['score']) * x[1]['confidence']
            )[0]
        else:
            dominant_aspect = None
        
        # Priority aspects (negative ones first, by intensity)
        priority = sorted(
            [(k, v) for k, v in mentioned_aspects.items() if v['sentiment'] == 'negative'],
            key=lambda x: x[1]['score'],  # More negative = higher priority
            reverse=False
        )
        priority_aspects = [k for k, v in priority]
        
        # Determine pattern
        if has_conflict:
            pattern = 'mixed_sentiment'
        elif len(negative_aspects) >= 2:
            pattern = 'multiple_issues'
        elif len(positive_aspects) >= 3:
            pattern = 'highly_positive'
        elif len(negative_aspects) == 1:
            pattern = 'single_issue'
        else:
            pattern = 'neutral_or_positive'
        
        return {
            'has_conflict': has_conflict,
            'dominant_aspect': dominant_aspect,
            'priority_aspects': priority_aspects,
            'positive_aspects': positive_aspects,
            'negative_aspects': negative_aspects,
            'neutral_aspects': neutral_aspects,
            'pattern': pattern,
            'overall_vs_aspects_mismatch': self._detect_mismatch(
                overall_sentiment, positive_aspects, negative_aspects
            )
        }
    
    def _detect_mismatch(self, overall_sentiment, positive_aspects, negative_aspects):
        """
        Detect when overall sentiment doesn't match aspect breakdown
        Example: Overall=positive but service=negative (hidden issue!)
        """
        if not overall_sentiment:
            return False
        
        # Overall positive but has negative aspects
        if overall_sentiment in ['positive', 'amazing'] and negative_aspects:
            return True
        
        # Overall negative but has positive aspects
        if overall_sentiment in ['negative', 'terrible'] and positive_aspects:
            return True
        
        return False
    
    def generate_response_strategy(self, analysis: Dict, overall_sentiment: str) -> Dict:
        """
        Determine response strategy based on analysis
        
        Returns strategy for response generation
        """
        
        pattern = analysis['patterns']['pattern']
        negative_aspects = analysis['patterns']['negative_aspects']
        positive_aspects = analysis['patterns']['positive_aspects']
        
        # Determine tone
        if overall_sentiment in ['terrible', 'negative'] or len(negative_aspects) >= 2:
            tone = 'apologetic_urgent'
        elif len(negative_aspects) == 1:
            tone = 'apologetic_standard'
        elif overall_sentiment == 'neutral':
            tone = 'appreciative_improving'
        else:
            tone = 'enthusiastic'
        
        # Determine what to address
        address_aspects = []
        
        # Always address negative aspects
        for aspect in negative_aspects:
            address_aspects.append({
                'aspect': aspect,
                'type': 'apologize',
                'priority': 'high'
            })
        
        # Acknowledge positive if overall is negative (show you care about what worked)
        if tone.startswith('apologetic') and positive_aspects:
            for aspect in positive_aspects[:2]:  # Max 2 to keep response concise
                address_aspects.append({
                    'aspect': aspect,
                    'type': 'acknowledge',
                    'priority': 'medium'
                })
        
        return {
            'tone': tone,
            'pattern': pattern,
            'address_aspects': address_aspects,
            'has_mixed': analysis['has_mixed_sentiment'],
            'priority': 'urgent' if tone == 'apologetic_urgent' else 'normal'
        }


def test_hybrid_absa():
    """Test the hybrid ABSA system"""
    
    absa = HybridABSA()
    
    test_reviews = [
        "The food was good. Service was not great particularly.",
        "Amazing ambiance and delicious food! Will definitely return!",
        "Terrible everything. Dirty place, rude staff, cold food.",
        "Great food but way overpriced for the portion size.",
    ]
    
    print("="*70)
    print("HYBRID ABSA TESTING")
    print("="*70)
    
    for review in test_reviews:
        print(f"\nReview: {review}")
        print("-"*70)
        
        analysis = absa.analyze(review)
        
        print(f"Aspects mentioned: {analysis['aspects_mentioned']}")
        print(f"Mixed sentiment: {analysis['has_mixed_sentiment']}")
        print(f"Pattern: {analysis['patterns']['pattern']}")
        print(f"Dominant aspect: {analysis['dominant_aspect']}")
        
        print("\nAspect Breakdown:")
        for aspect, data in analysis['aspect_sentiments'].items():
            if data['sentiment'] != 'not_mentioned':
                print(f"  {aspect}: {data['sentiment']} (score: {data['score']}, confidence: {data['confidence']})")
        
        if analysis['patterns']['priority_aspects']:
            print(f"\nPriority issues: {analysis['patterns']['priority_aspects']}")


if __name__ == "__main__":
    test_hybrid_absa()