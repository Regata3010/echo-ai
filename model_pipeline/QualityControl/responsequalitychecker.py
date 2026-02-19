"""
Response Quality Scorer
Evaluates quality of AI-generated responses across multiple dimensions
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseQualityScorer:
    """
    Score generated responses on multiple quality dimensions
    """
    
    def __init__(self):
        # Empathy indicators
        self.empathy_phrases = {
            'high': [
                'we understand', 'we appreciate', 'we hear you', 'we value',
                'thank you for sharing', 'we sincerely', 'deeply sorry',
                'we truly appreciate', 'means a lot', 'we care'
            ],
            'medium': [
                'thank you', 'we apologize', 'sorry', 'appreciate',
                'understand', 'hear', 'noted'
            ],
            'low': [
                'okay', 'noted', 'received'
            ]
        }
        
        # Specificity indicators
        self.specific_terms = [
            'manager', 'team', 'chef', 'kitchen', 'specifically',
            'particular', 'issue', 'concern', 'situation', 'experience',
            'meal', 'dish', 'service', 'staff member'
        ]
        
        # Action words
        self.action_words = [
            'will', 'contact', 'resolve', 'address', 'fix', 'improve',
            'investigate', 'follow up', 'reach out', 'call', 'email',
            'refund', 'compensate', 'make it right'
        ]
        
        # Generic filler phrases (bad)
        self.generic_phrases = [
            'thank you for your feedback',
            'we appreciate your review',
            'your feedback is important',
            'we value your input',
            'thank you for bringing this to our attention'
        ]
    
    def score_empathy(self, response: str) -> Dict:
        """
        Score empathy level (0-10)
        Higher = more empathetic and human
        """
        response_lower = response.lower()
        score = 0
        details = []
        
        # Check for high empathy phrases
        high_count = sum(1 for phrase in self.empathy_phrases['high'] 
                        if phrase in response_lower)
        score += high_count * 3
        if high_count > 0:
            details.append(f"High empathy phrases: {high_count}")
        
        # Medium empathy
        medium_count = sum(1 for phrase in self.empathy_phrases['medium'] 
                          if phrase in response_lower)
        score += medium_count * 1.5
        if medium_count > 0:
            details.append(f"Medium empathy phrases: {medium_count}")
        
        # Personal pronouns (we, our, us)
        personal_pronouns = len(re.findall(r'\b(we|our|us)\b', response_lower))
        score += personal_pronouns * 0.5
        details.append(f"Personal pronouns: {personal_pronouns}")
        
        # Exclamation marks (show enthusiasm, but too many is bad)
        exclamations = response.count('!')
        if exclamations == 1 or exclamations == 2:
            score += 1
            details.append("Appropriate enthusiasm")
        elif exclamations > 3:
            score -= 1
            details.append("Excessive exclamation marks")
        
        # Cap at 10
        score = min(score, 10)
        
        return {
            'score': round(score, 1),
            'max': 10,
            'details': details,
            'level': 'high' if score >= 7 else 'medium' if score >= 4 else 'low'
        }
    
    def score_specificity(self, response: str, original_review: str = None, sentiment: str = 'neutral') -> Dict:
        """
        Score specificity (0-10)
        Higher = addresses specific points from review, not generic
        
        ADJUSTED: Positive reviews can be more generic, negative reviews need specificity
        """
        response_lower = response.lower()
        score = 5  # Start at midpoint
        details = []
        
        # For POSITIVE/AMAZING reviews, generic is acceptable
        if sentiment in ['positive', 'amazing', 'excellent']:
            # Don't penalize generic phrases for positive reviews
            generic_count = sum(1 for phrase in self.generic_phrases 
                              if phrase in response_lower)
            if generic_count > 0:
                details.append(f"Generic phrases: {generic_count} (OK for positive reviews)")
            
            # Just check if response shows appreciation
            appreciation_words = ['thank', 'thrilled', 'delighted', 'happy', 'glad']
            has_appreciation = any(word in response_lower for word in appreciation_words)
            if has_appreciation:
                score += 3
                details.append("Shows appreciation")
        
        # For NEGATIVE/TERRIBLE reviews, specificity is CRITICAL
        else:
            # Penalty for generic phrases in negative responses
            generic_count = sum(1 for phrase in self.generic_phrases 
                              if phrase in response_lower)
            score -= generic_count * 1.5
            if generic_count > 0:
                details.append(f"Generic phrases: {generic_count} (penalty for negative review)")
            
            # Reward for specific terms
            specific_count = sum(1 for term in self.specific_terms 
                               if term in response_lower)
            score += specific_count * 2
            if specific_count > 0:
                details.append(f"Specific terms: {specific_count}")
            
            # Check if response references specific aspects from review
            if original_review:
                review_lower = original_review.lower()
                
                # Extract key nouns from review
                review_words = set(re.findall(r'\b[a-z]{4,}\b', review_lower))
                response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
                
                # How many review-specific words are in response?
                overlap = len(review_words & response_words)
                overlap_rate = overlap / max(len(review_words), 1)
                
                score += overlap_rate * 4
                details.append(f"Review word overlap: {overlap} words ({overlap_rate:.1%})")
        
        # Response length (adjusted by sentiment)
        word_count = len(response.split())
        if sentiment in ['positive', 'amazing', 'excellent']:
            # Positive can be shorter
            if word_count < 15:
                score -= 1
                details.append("A bit short")
            elif word_count > 60:
                score -= 0.5
                details.append("Slightly long")
            else:
                score += 1
                details.append(f"Good length ({word_count} words)")
        else:
            # Negative needs more detail
            if word_count < 25:
                score -= 2
                details.append("Too short for negative review")
            elif word_count > 100:
                score -= 1
                details.append("Too long")
            else:
                score += 1
                details.append(f"Good length ({word_count} words)")
        
        # Normalize to 0-10
        score = max(0, min(score, 10))
        
        return {
            'score': round(score, 1),
            'max': 10,
            'details': details,
            'level': 'high' if score >= 7 else 'medium' if score >= 4 else 'low'
        }
    
    def score_actionability(self, response: str, sentiment: str) -> Dict:
        """
        Score actionability (0-10)
        Does response offer concrete next steps?
        """
        response_lower = response.lower()
        score = 0
        details = []
        
        # Count action words
        action_count = sum(1 for word in self.action_words 
                          if word in response_lower)
        score += action_count * 2
        if action_count > 0:
            details.append(f"Action words: {action_count}")
        
        # Check for contact information request
        has_contact = any(phrase in response_lower for phrase in [
            'contact us', 'reach out', 'call us', 'email us',
            'phone', 'speak with', 'message us'
        ])
        if has_contact:
            score += 3
            details.append("Requests contact/follow-up")
        
        # For negative reviews, should offer resolution
        if sentiment in ['negative', 'terrible']:
            has_resolution = any(phrase in response_lower for phrase in [
                'make it right', 'resolve', 'refund', 'compensate',
                'manager', 'investigate', 'fix'
            ])
            
            if has_resolution:
                score += 3
                details.append("Offers concrete resolution")
            else:
                score -= 2
                details.append("Missing resolution offer (needed for negative review)")
        
        # For positive reviews, should encourage return visit
        if sentiment in ['positive', 'amazing', 'excellent']:
            has_encouragement = any(phrase in response_lower for phrase in [
                'come back', 'visit again', 'see you', 'return',
                'next time', 'welcome back'
            ])
            
            if has_encouragement:
                score += 2
                details.append("Encourages return visit")
        
        # Normalize
        score = max(0, min(score, 10))
        
        return {
            'score': round(score, 1),
            'max': 10,
            'details': details,
            'level': 'high' if score >= 7 else 'medium' if score >= 4 else 'low'
        }
    
    def score_professionalism(self, response: str) -> Dict:
        """
        Score professionalism (0-10)
        Grammar, tone, appropriateness
        """
        score = 10  # Start at perfect, deduct for issues
        details = []
        
        # Check for spelling/grammar issues
        blob = TextBlob(response)
        if blob.sentiment.polarity < -0.5:
            score -= 2
            details.append("Overly negative tone")
        
        # Check for inappropriate casual language
        casual_terms = ['lol', 'omg', 'hey', 'yeah', 'gonna', 'wanna', 'cool', 'awesome']
        casual_count = sum(1 for term in casual_terms if term in response.lower())
        if casual_count > 0:
            score -= casual_count * 1.5
            details.append(f"Overly casual language: {casual_count} instances")
        
        # Check capitalization (should start with capital)
        if response and not response[0].isupper():
            score -= 1
            details.append("Missing capitalization")
        
        # Check ending punctuation
        if response and response[-1] not in '.!?':
            score -= 0.5
            details.append("Missing ending punctuation")
        
        # Multiple exclamation marks (unprofessional)
        if '!!' in response or '!!!' in response:
            score -= 2
            details.append("Excessive exclamation marks")
        
        # ALL CAPS (unprofessional)
        if any(word.isupper() and len(word) > 3 for word in response.split()):
            score -= 2
            details.append("Contains all-caps words")
        
        # Length appropriateness
        word_count = len(response.split())
        if word_count < 10:
            score -= 2
            details.append("Too brief for professional response")
        
        # Normalize
        score = max(0, min(score, 10))
        
        return {
            'score': round(score, 1),
            'max': 10,
            'details': details if details else ["No issues detected"],
            'level': 'high' if score >= 8 else 'medium' if score >= 6 else 'low'
        }
    
    def score_response(self, response: str, original_review: str, 
                      sentiment: str, rating: int = None) -> Dict:
        """
        Complete quality scoring across all dimensions
        ADJUSTED: Different expectations for positive vs negative reviews
        """
        if not response or pd.isna(response):
            return {
                'overall_score': 0,
                'grade': 'F',
                'error': 'Empty response'
            }
        
        # Score each dimension (pass sentiment to specificity and actionability)
        empathy = self.score_empathy(response)
        specificity = self.score_specificity(response, original_review, sentiment)
        actionability = self.score_actionability(response, sentiment)
        professionalism = self.score_professionalism(response)
        
        # Adjusted weights based on sentiment
        if sentiment in ['negative', 'terrible']:
            # For negative reviews: actionability and specificity matter most
            weights = {
                'empathy': 0.25,
                'specificity': 0.30,
                'actionability': 0.35,  # Increased
                'professionalism': 0.10
            }
        elif sentiment in ['positive', 'amazing', 'excellent']:
            # For positive reviews: empathy and professionalism matter most
            weights = {
                'empathy': 0.40,  # Increased
                'specificity': 0.15,  # Decreased
                'actionability': 0.15,  # Decreased
                'professionalism': 0.30
            }
        else:
            # Neutral: balanced
            weights = {
                'empathy': 0.25,
                'specificity': 0.30,
                'actionability': 0.25,
                'professionalism': 0.20
            }
        
        overall_score = (
            empathy['score'] * weights['empathy'] +
            specificity['score'] * weights['specificity'] +
            actionability['score'] * weights['actionability'] +
            professionalism['score'] * weights['professionalism']
        )
        
        # Assign letter grade
        if overall_score >= 9:
            grade = 'A+'
        elif overall_score >= 8:
            grade = 'A'
        elif overall_score >= 7:
            grade = 'B'
        elif overall_score >= 6:
            grade = 'C'
        elif overall_score >= 5:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_score': round(overall_score, 2),
            'grade': grade,
            'empathy': empathy,
            'specificity': specificity,
            'actionability': actionability,
            'professionalism': professionalism,
            'weights_used': weights,
            'review_text': original_review[:100] + '...',
            'response_text': response[:100] + '...',
            'sentiment': sentiment,
            'rating': rating
        }
    
    def evaluate_batch_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate quality for all generated responses in a dataset
        """
        logger.info(f"Evaluating {len(df)} responses...")
        
        scores = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Evaluated {idx}/{len(df)} responses")
            
            score_result = self.score_response(
                response=row.get('generated_response', ''),
                original_review=row.get('reviewText', ''),
                sentiment=row.get('sentiment', 'neutral'),
                rating=row.get('reviewRating')
            )
            scores.append(score_result)
        
        # Add scores to dataframe
        df['response_quality_score'] = [s['overall_score'] for s in scores]
        df['response_grade'] = [s['grade'] for s in scores]
        df['empathy_score'] = [s['empathy']['score'] for s in scores]
        df['specificity_score'] = [s['specificity']['score'] for s in scores]
        df['actionability_score'] = [s['actionability']['score'] for s in scores]
        df['professionalism_score'] = [s['professionalism']['score'] for s in scores]
        
        return df, scores
    
    def generate_quality_report(self, scores: List[Dict]) -> Dict:
        """
        Generate summary report of response quality
        """
        valid_scores = [s for s in scores if 'overall_score' in s]
        
        if not valid_scores:
            return {'error': 'No valid scores'}
        
        report = {
            'total_responses': len(valid_scores),
            'average_quality_score': round(np.mean([s['overall_score'] for s in valid_scores]), 2),
            'grade_distribution': {},
            'dimension_averages': {
                'empathy': round(np.mean([s['empathy']['score'] for s in valid_scores]), 2),
                'specificity': round(np.mean([s['specificity']['score'] for s in valid_scores]), 2),
                'actionability': round(np.mean([s['actionability']['score'] for s in valid_scores]), 2),
                'professionalism': round(np.mean([s['professionalism']['score'] for s in valid_scores]), 2)
            }
        }
        
        # Grade distribution
        from collections import Counter
        grades = Counter([s['grade'] for s in valid_scores])
        report['grade_distribution'] = dict(grades)
        
        # Find best and worst responses
        sorted_scores = sorted(valid_scores, key=lambda x: x['overall_score'], reverse=True)
        
        report['best_response'] = {
            'score': sorted_scores[0]['overall_score'],
            'grade': sorted_scores[0]['grade'],
            'response': sorted_scores[0]['response_text'],
            'review': sorted_scores[0]['review_text']
        }
        
        report['worst_response'] = {
            'score': sorted_scores[-1]['overall_score'],
            'grade': sorted_scores[-1]['grade'],
            'response': sorted_scores[-1]['response_text'],
            'review': sorted_scores[-1]['review_text']
        }
        
        # Identify improvement areas
        improvement_areas = []
        for dimension, avg_score in report['dimension_averages'].items():
            if avg_score < 6:
                improvement_areas.append({
                    'dimension': dimension,
                    'current_score': avg_score,
                    'gap': 8 - avg_score  # Target score is 8
                })
        
        report['improvement_areas'] = sorted(improvement_areas, 
                                            key=lambda x: x['gap'], 
                                            reverse=True)
        
        return report
    
    def create_quality_visualizations(self, df: pd.DataFrame, save_dir='results/'):
        """
        Create visualizations comparing response quality
        """
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Figure 1: Grade Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        grade_counts = df['response_grade'].value_counts().sort_index()
        colors = {'A+': 'darkgreen', 'A': 'green', 'B': 'yellowgreen', 
                 'C': 'orange', 'D': 'orangered', 'F': 'red'}
        grade_counts.plot(kind='bar', ax=ax, 
                         color=[colors.get(g, 'gray') for g in grade_counts.index])
        ax.set_title('Response Quality Grade Distribution')
        ax.set_xlabel('Grade')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quality_grade_distribution.png', dpi=300)
        plt.close()
        
        # Figure 2: Dimension Scores
        fig, ax = plt.subplots(figsize=(10, 6))
        dimensions = ['empathy_score', 'specificity_score', 'actionability_score', 'professionalism_score']
        avg_scores = [df[dim].mean() for dim in dimensions]
        dim_names = ['Empathy', 'Specificity', 'Actionability', 'Professionalism']
        
        bars = ax.barh(dim_names, avg_scores, color='steelblue')
        ax.axvline(x=8, color='green', linestyle='--', label='Target (8.0)')
        ax.axvline(x=6, color='orange', linestyle='--', label='Acceptable (6.0)')
        ax.set_xlabel('Average Score (0-10)')
        ax.set_title('Response Quality by Dimension')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, avg_scores)):
            ax.text(score + 0.2, i, f'{score:.1f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quality_dimensions.png', dpi=300)
        plt.close()
        
        # Figure 3: Quality vs Sentiment
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiment_quality = df.groupby('sentiment')['response_quality_score'].mean().sort_values()
        sentiment_quality.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Average Response Quality by Review Sentiment')
        ax.set_xlabel('Review Sentiment')
        ax.set_ylabel('Average Quality Score')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=7, color='green', linestyle='--', label='Good Quality (7.0)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quality_by_sentiment.png', dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {save_dir}")


def main():
    """
    Main function to score responses and generate quality report
    """
    print("\n" + "="*70)
    print("  RESPONSE QUALITY SCORING SYSTEM")
    print("="*70)
    
    # Load data with generated responses
    logger.info("\nLoading data with generated responses...")
    
    # You'll need to generate responses first using inference_pipeline.py
    # For now, let's create sample data
    
    sample_data = [
        {
            'reviewText': 'The food was absolutely amazing but the service was terrible and slow.',
            'reviewRating': 3,
            'sentiment': 'neutral',
            'generated_response': 'Thank you for your feedback. We value your input.'
        },
        {
            'reviewText': 'Best restaurant in Boston! Everything was perfect!',
            'reviewRating': 5,
            'sentiment': 'amazing',
            'generated_response': 'We are thrilled to hear about your wonderful experience! Thank you so much for your kind words. We truly appreciate your support and can\'t wait to welcome you back soon!'
        },
        {
            'reviewText': 'Worst experience ever. Cold food, rude staff, never coming back.',
            'reviewRating': 1,
            'sentiment': 'terrible',
            'generated_response': 'We sincerely apologize for your disappointing experience. Please contact our manager directly at manager@restaurant.com so we can investigate this immediately and make it right.'
        }
    ]
    
    df_sample = pd.DataFrame(sample_data)
    
    # Initialize scorer
    scorer = ResponseQualityScorer()
    
    # Evaluate responses
    logger.info("\nScoring responses...")
    df_scored, all_scores = scorer.evaluate_batch_responses(df_sample)
    
    # Print individual scores
    print("\n\nINDIVIDUAL RESPONSE SCORES:")
    print("="*70)
    
    for i, score in enumerate(all_scores, 1):
        print(f"\nResponse {i}:")
        print(f"  Review: {score['review_text']}")
        print(f"  Response: {score['response_text']}")
        print(f"  Overall Score: {score['overall_score']}/10 (Grade: {score['grade']})")
        print(f"  Breakdown:")
        print(f"    - Empathy: {score['empathy']['score']}/10 ({score['empathy']['level']})")
        print(f"    - Specificity: {score['specificity']['score']}/10 ({score['specificity']['level']})")
        print(f"    - Actionability: {score['actionability']['score']}/10 ({score['actionability']['level']})")
        print(f"    - Professionalism: {score['professionalism']['score']}/10 ({score['professionalism']['level']})")
        
        # Show details for lowest scoring dimension
        lowest_dim = min(
            [('empathy', score['empathy']), 
             ('specificity', score['specificity']),
             ('actionability', score['actionability']),
             ('professionalism', score['professionalism'])],
            key=lambda x: x[1]['score']
        )
        print(f"  Needs Improvement: {lowest_dim[0]} ({lowest_dim[1]['score']}/10)")
        print(f"    Feedback: {'; '.join(lowest_dim[1]['details'])}")
    
    # Generate quality report
    logger.info("\nGenerating quality report...")
    quality_report = scorer.generate_quality_report(all_scores)
    
    print("\n\nQUALITY REPORT SUMMARY:")
    print("="*70)
    print(f"Total Responses Evaluated: {quality_report['total_responses']}")
    print(f"Average Quality Score: {quality_report['average_quality_score']}/10")
    print(f"\nGrade Distribution:")
    for grade, count in sorted(quality_report['grade_distribution'].items()):
        print(f"  {grade}: {count}")
    
    print(f"\nDimension Averages:")
    for dim, score in quality_report['dimension_averages'].items():
        status = "GOOD" if score >= 7 else "NEEDS WORK" if score < 6 else "OK"
        print(f"  {dim.capitalize():15} : {score:.1f}/10 [{status}]")
    
    if quality_report['improvement_areas']:
        print(f"\nPRIORITY IMPROVEMENTS:")
        for area in quality_report['improvement_areas']:
            print(f"  - {area['dimension'].capitalize()}: {area['current_score']:.1f}/10 (need +{area['gap']:.1f} points)")
    
    print("\n" + "="*70)
    
    return df_scored, quality_report


if __name__ == "__main__":
    main()