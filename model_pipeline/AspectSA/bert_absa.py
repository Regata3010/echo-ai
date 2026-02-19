# """
# BERT-Based Aspect Sentiment Analysis
# Uses transformer models to understand aspect-specific sentiment
# """
# import spacy
# from typing import Dict, List
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class BERTAspectSentiment:
#     """
#     BERT-based aspect sentiment analysis
#     Analyzes sentiment for specific aspects using context
#     """
    
#     def __init__(self):
#         # Load spaCy for sentence splitting
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except:
#             import os
#             os.system("python -m spacy download en_core_web_sm")
#             self.nlp = spacy.load("en_core_web_sm")
        
#         # Aspect keywords
#         self.aspect_keywords = {
#             'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'cuisine', 'menu', 
#                     'chicken', 'beef', 'pasta', 'pizza', 'salad', 'dessert', 'appetizer',
#                     'entree', 'breakfast', 'lunch', 'dinner', 'quality', 'fresh'],
            
#             'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'employee',
#                        'manager', 'bartender', 'host', 'hostess', 'team', 'friendly',
#                        'rude', 'slow', 'fast', 'attentive', 'professional'],
            
#             'ambiance': ['ambiance', 'atmosphere', 'decor', 'environment', 'vibe', 'mood',
#                         'interior', 'design', 'lighting', 'music', 'noise', 'quiet',
#                         'crowded', 'space', 'seating', 'cozy', 'comfortable'],
            
#             'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money',
#                      'afford', 'budget', 'overpriced', 'reasonable', 'dollar', 'deal'],
            
#             'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'tidy', 'mess',
#                            'bathroom', 'table', 'floor', 'spotless', 'filthy']
#         }
        
#         # Simple sentiment lexicon
#         self.positive_words = {
#             'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect',
#             'delicious', 'best', 'love', 'awesome', 'outstanding', 'superb', 'tasty',
#             'fresh', 'quality', 'friendly', 'helpful', 'fast', 'clean', 'comfortable',
#             'beautiful', 'nice', 'recommend', 'enjoyed', 'loved', 'impressed','fine', 
#                 'decent', 'solid', 'enjoyed'
#         }
        
#         self.negative_words = {
#             'bad', 'terrible', 'awful', 'horrible', 'worst', 'disgusting', 'disappointing',
#             'poor', 'mediocre', 'overpriced', 'slow', 'rude', 'cold', 'dirty', 'small',
#             'bland', 'stale', 'burnt', 'raw', 'wait', 'long', 'crowded', 'noisy',
#             'uncomfortable', 'unacceptable', 'avoid', 'never', 'waste'
#         }
        
#         self.negations = {'not', 'no', 'never', "n't", 'neither', 'nor', 'barely', 'hardly'}
    
#     def extract_aspect_sentences(self, text: str) -> Dict[str, List[str]]:
#         """
#         Extract sentences mentioning each aspect
#         Split on contrast words (but, however, although)
#         """
#         # First split on contrast words
#         import re
#         contrast_words = ['but', 'however', 'although', 'though', 'yet']
        
#         # Split text into clauses
#         clauses = [text]
#         for word in contrast_words:
#             new_clauses = []
#             for clause in clauses:
#                 new_clauses.extend(re.split(f' {word} ', clause, flags=re.IGNORECASE))
#             clauses = new_clauses
        
#         # Now extract aspects from each clause
#         aspect_sentences = {aspect: [] for aspect in self.aspect_keywords.keys()}
        
#         for clause in clauses:
#             clause_lower = clause.lower()
#             for aspect, keywords in self.aspect_keywords.items():
#                 if any(keyword in clause_lower for keyword in keywords):
#                     aspect_sentences[aspect].append(clause.strip())
        
#         return {k: v for k, v in aspect_sentences.items() if v}
    
#     def analyze_sentence_sentiment(self, sentence: str) -> Dict:
#         """
#         Analyze sentiment of a single sentence using lexicon + negation handling
#         """
#         doc = self.nlp(sentence.lower())
#         tokens = [token.text for token in doc]
        
#         positive_count = 0
#         negative_count = 0
        
#         for i, token in enumerate(tokens):
#             # Check for negation in 3-word window before token
#             window_start = max(0, i - 3)
#             window = tokens[window_start:i]
#             has_negation = any(neg in window for neg in self.negations)
            
#             # Score sentiment words
#             if token in self.positive_words:
#                 if has_negation:
#                     negative_count += 1  # "not good" = negative
#                 else:
#                     positive_count += 1
            
#             elif token in self.negative_words:
#                 if has_negation:
#                     positive_count += 1  # "not bad" = positive
#                 else:
#                     negative_count += 1
        
#         # Calculate score
#         total = positive_count + negative_count
#         if total == 0:
#             return {'sentiment': 'neutral', 'score': 0, 'confidence': 0}
        
#         score = (positive_count - negative_count) / total
#         confidence = min(total / 5, 1.0)  # More words = higher confidence
        
#         # Classify
#         if score > 0.1:
#             sentiment = 'positive'
#         elif score < -0.1:
#             sentiment = 'negative'
#         else:
#             sentiment = 'neutral'
        
#         return {
#             'sentiment': sentiment,
#             'score': round(score, 3),
#             'confidence': round(confidence, 3),
#             'positive_words': positive_count,
#             'negative_words': negative_count
#         }
    
#     def analyze_aspect_sentiment(self, sentences: List[str]) -> Dict:
#         """
#         Analyze sentiment across all sentences mentioning an aspect
#         """
#         if not sentences:
#             return {'sentiment': 'not_mentioned', 'score': 0, 'confidence': 0}
        
#         # Analyze each sentence
#         sentence_sentiments = [self.analyze_sentence_sentiment(s) for s in sentences]
        
#         # Aggregate
#         avg_score = sum(s['score'] for s in sentence_sentiments) / len(sentence_sentiments)
#         avg_confidence = sum(s['confidence'] for s in sentence_sentiments) / len(sentence_sentiments)
        
#         # Final sentiment
#         if avg_score > 0.2:
#             sentiment = 'positive'
#         elif avg_score < -0.2:
#             sentiment = 'negative'
#         else:
#             sentiment = 'neutral'
        
#         return {
#             'sentiment': sentiment,
#             'score': round(avg_score, 3),
#             'confidence': round(avg_confidence, 3),
#             'num_mentions': len(sentences),
#             'sentences': sentences
#         }
    
#     def analyze_review(self, review_text: str) -> Dict:
#         """
#         Complete aspect-based analysis of review
#         """
#         # Extract aspect sentences
#         aspect_sentences = self.extract_aspect_sentences(review_text)
        
#         # Analyze sentiment for each aspect
#         aspect_results = {}
#         for aspect, sentences in aspect_sentences.items():
#             aspect_results[aspect] = self.analyze_aspect_sentiment(sentences)
        
#         # Add not-mentioned aspects
#         for aspect in self.aspect_keywords.keys():
#             if aspect not in aspect_results:
#                 aspect_results[aspect] = {
#                     'sentiment': 'not_mentioned',
#                     'score': 0,
#                     'confidence': 0
#                 }
        
#         # Detect conflicts
#         mentioned = {k: v for k, v in aspect_results.items() if v['sentiment'] != 'not_mentioned'}
        
#         has_positive = any(v['sentiment'] == 'positive' for v in mentioned.values())
#         has_negative = any(v['sentiment'] == 'negative' for v in mentioned.values())
        
#         return {
#             'aspects': aspect_results,
#             'has_mixed_sentiment': has_positive and has_negative,
#             'dominant_sentiment': self._calculate_dominant(aspect_results),
#             'aspects_mentioned': list(mentioned.keys())
#         }
    
#     def _calculate_dominant(self, aspect_results):
#         """Calculate overall sentiment from aspects"""
#         mentioned = [v for v in aspect_results.values() if v['sentiment'] != 'not_mentioned']
        
#         if not mentioned:
#             return 'neutral'
        
#         avg_score = sum(v['score'] for v in mentioned) / len(mentioned)
        
#         if avg_score > 0.3:
#             return 'positive'
#         elif avg_score < -0.3:
#             return 'negative'
#         else:
#             return 'neutral'


#-------------------------------------------------------------------------------------------------------------------------------------

"""
BERT-Based Aspect Sentiment Analysis
Uses transformer models for AI-powered sentiment scoring (no lexicons)
"""
import spacy
from transformers import pipeline
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTAspectSentiment:
    """
    AI-powered aspect sentiment analysis
    No lexicon limitations - handles any vocabulary
    """
    
    def __init__(self):
        # Load spaCy for sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # AI-powered sentiment scorer (replaces lexicons)
        try:
            self.sentiment_scorer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("AI sentiment scorer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment scorer: {e}")
            raise
        
        # Aspect keywords (keep for detection, not sentiment)
        self.aspect_keywords = {
            'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'cuisine', 'menu', 
                    'chicken', 'beef', 'pasta', 'pizza', 'salad', 'dessert', 'appetizer',
                    'entree', 'breakfast', 'lunch', 'dinner', 'quality', 'fresh', 'delicious',
                    'portion', 'plate', 'cooked', 'seasoned', 'spicy', 'bland'],
            
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'employee',
                       'manager', 'bartender', 'host', 'hostess', 'team', 'friendly',
                       'rude', 'slow', 'fast', 'attentive', 'professional', 'wait', 'waiting'],
            
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'environment', 'vibe', 'mood',
                        'interior', 'design', 'lighting', 'music', 'noise', 'quiet',
                        'crowded', 'space', 'seating', 'cozy', 'comfortable', 'setting'],
            
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money',
                     'afford', 'budget', 'overpriced', 'reasonable', 'dollar', 'deal',
                     'pricey', 'costly'],
            
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'tidy', 'mess',
                           'bathroom', 'table', 'floor', 'spotless', 'filthy', 'gross']
        }
    
    def extract_aspect_sentences(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        aspect_sentences = {aspect: [] for aspect in self.aspect_keywords.keys()}
        
        # Split on contrast conjunctions to isolate clause sentiment
        contrast_markers = [' but ', ' however ', ' although ', ' though ', ' yet ', ' despite ']
        clauses = [text]
        for marker in contrast_markers:
            if marker in text.lower():
                split_text = text.lower().replace(marker, '|||')
                clauses = text.split(text[text.lower().index(marker):text.lower().index(marker)+len(marker)])
                break
        
        for clause in clauses:
            clause_lower = clause.lower()
            for aspect, keywords in self.aspect_keywords.items():
                if any(kw in clause_lower for kw in keywords):
                    aspect_sentences[aspect].append(clause)
        
        # Fall back to spaCy sentence splitting if no clauses found
        if all(len(v) == 0 for v in aspect_sentences.values()):
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                for aspect, keywords in self.aspect_keywords.items():
                    if any(kw in sent_lower for kw in keywords):
                        aspect_sentences[aspect].append(sent.text)
        
        return {k: v for k, v in aspect_sentences.items() if v}
    
    def analyze_sentence_sentiment_ai(self, sentence: str) -> Dict:
        """
        AI-powered sentiment analysis of sentence
        Works for ANY vocabulary (no lexicon needed)
        """
        try:
            # Use AI model to score sentiment
            result = self.sentiment_scorer(sentence)[0]
            
            # Convert to our format
            if result['label'] == 'POSITIVE':
                sentiment = 'positive'
                score = result['score']
            else:  # NEGATIVE
                sentiment = 'negative'
                score = -result['score']
            
            # Adjust for neutral (low confidence)
            if abs(score) < 0.4:
                sentiment = 'neutral'
                score = score if result['label'] == 'POSITIVE' else -score
            
            return {
                'sentiment': sentiment,
                'score': round(score, 3),
                'confidence': round(result['score'], 3)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing sentence: {e}")
            return {'sentiment': 'neutral', 'score': 0, 'confidence': 0}
    
    def analyze_aspect_sentiment(self, sentences: List[str]) -> Dict:
        """
        Analyze sentiment across all sentences mentioning an aspect
        """
        if not sentences:
            return {'sentiment': 'not_mentioned', 'score': 0, 'confidence': 0}
        
        # Analyze each sentence with AI
        sentence_sentiments = [self.analyze_sentence_sentiment_ai(s) for s in sentences]
        
        # Aggregate
        avg_score = sum(s['score'] for s in sentence_sentiments) / len(sentence_sentiments)
        avg_confidence = sum(s['confidence'] for s in sentence_sentiments) / len(sentence_sentiments)
        
        # Final sentiment
        if avg_score > 0.2:
            sentiment = 'positive'
        elif avg_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': round(avg_score, 3),
            'confidence': round(avg_confidence, 3),
            'num_mentions': len(sentences),
            'sentences': sentences
        }
    
    def analyze_review(self, review_text: str) -> Dict:
        """
        Complete aspect-based analysis using AI
        """
        # Extract aspect sentences
        aspect_sentences = self.extract_aspect_sentences(review_text)
        
        # Analyze sentiment for each aspect with AI
        aspect_results = {}
        for aspect, sentences in aspect_sentences.items():
            aspect_results[aspect] = self.analyze_aspect_sentiment(sentences)
        
        # Add not-mentioned aspects
        for aspect in self.aspect_keywords.keys():
            if aspect not in aspect_results:
                aspect_results[aspect] = {
                    'sentiment': 'not_mentioned',
                    'score': 0,
                    'confidence': 0
                }
        
        # Detect conflicts
        mentioned = {k: v for k, v in aspect_results.items() if v['sentiment'] != 'not_mentioned'}
        
        has_positive = any(v['sentiment'] == 'positive' for v in mentioned.values())
        has_negative = any(v['sentiment'] == 'negative' for v in mentioned.values())
        
        return {
            'aspects': aspect_results,
            'has_mixed_sentiment': has_positive and has_negative,
            'dominant_sentiment': self._calculate_dominant(aspect_results),
            'aspects_mentioned': list(mentioned.keys())
        }
    
    def _calculate_dominant(self, aspect_results):
        """Calculate overall sentiment from aspects"""
        mentioned = [v for v in aspect_results.values() if v['sentiment'] != 'not_mentioned']
        
        if not mentioned:
            return 'neutral'
        
        avg_score = sum(v['score'] for v in mentioned) / len(mentioned)
        
        if avg_score > 0.3:
            return 'positive'
        elif avg_score < -0.3:
            return 'negative'
        else:
            return 'neutral'