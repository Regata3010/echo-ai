"""
Aspect-Aware Response Generator
Generates personalized responses based on specific aspect sentiments
Much better than generic overall sentiment responses
"""
import random
from typing import Dict, List
import pandas as pd

class AspectAwareResponseGenerator:
    """
    Generate responses that address specific aspects mentioned in reviews
    """
    
    def __init__(self):
        # Aspect-specific response templates
        self.aspect_templates = {
            'food': {
                'positive': [
                    "We're so glad you enjoyed our food!",
                    "Thank you for the kind words about our cuisine!",
                    "We're delighted our food met your expectations!",
                    "Our chefs will be thrilled to hear you loved the dishes!"
                ],
                'negative': [
                    "We apologize that our food didn't meet your expectations.",
                    "We're sorry to hear the food was disappointing.",
                    "We take your concerns about our food quality seriously.",
                    "Our kitchen team is addressing the food quality issues you mentioned."
                ]
            },
            'service': {
                'positive': [
                    "Our team is thrilled to hear about your positive experience!",
                    "Thank you for recognizing our staff's hard work!",
                    "We're so happy our service made your visit special!",
                    "Our servers will be delighted by your kind feedback!"
                ],
                'negative': [
                    "We sincerely apologize for the poor service you received.",
                    "We're addressing the service issues with our team immediately.",
                    "Your feedback about our service is being taken very seriously.",
                    "We're sorry our staff didn't provide the service you deserved."
                ]
            },
            'ambiance': {
                'positive': [
                    "We're glad you enjoyed our atmosphere!",
                    "Thank you for appreciating our ambiance!",
                    "We're happy the environment added to your experience!"
                ],
                'negative': [
                    "We apologize for the uncomfortable atmosphere.",
                    "We're working on improving our dining environment.",
                    "Thank you for the feedback about our ambiance - we're making changes."
                ]
            },
            'price': {
                'positive': [
                    "We're glad you found our pricing fair!",
                    "Thank you for recognizing our value!",
                    "We strive to offer great value for money!"
                ],
                'negative': [
                    "We understand your concerns about pricing.",
                    "We appreciate your feedback on our pricing structure.",
                    "We're reviewing our prices to ensure we provide good value."
                ]
            },
            'cleanliness': {
                'positive': [
                    "We're pleased our cleanliness standards impressed you!",
                    "Thank you for noticing our attention to hygiene!"
                ],
                'negative': [
                    "We sincerely apologize for the cleanliness issues - this is unacceptable.",
                    "We're addressing the hygiene concerns immediately with our team.",
                    "Cleanliness is our top priority and we failed you - we're very sorry."
                ]
            }
        }
        
        # Closing statements based on overall sentiment
        self.closings = {
            'mostly_positive': [
                "We look forward to welcoming you back soon!",
                "Thank you for choosing us, and we hope to see you again!",
                "We can't wait to serve you again!"
            ],
            'mixed': [
                "We hope to serve you better next time. Please contact us at manager@restaurant.com if you'd like to discuss further.",
                "We value your feedback and hope you'll give us another chance to provide a better experience.",
                "Please reach out to us directly so we can address your concerns."
            ],
            'mostly_negative': [
                "Please contact our manager at manager@restaurant.com or (555) 123-4567 so we can make this right.",
                "We'd like to resolve this personally. Please call us at (555) 123-4567.",
                "Please email us at support@restaurant.com - we want to restore your trust in us."
            ]
        }
    
    def generate_aspect_based_response(self, 
                                      aspect_sentiments: Dict[str, str],
                                      overall_rating: int = None,
                                      place_name: str = None) -> str:
        """
        Generate response based on specific aspect sentiments
        
        Args:
            aspect_sentiments: Dict like {'food': 'positive', 'service': 'negative', ...}
            overall_rating: Overall rating (1-5)
            place_name: Restaurant name
        
        Returns:
            Personalized response addressing specific aspects
        """
        response_parts = []
        
        # Count positive vs negative aspects
        positive_aspects = [asp for asp, sent in aspect_sentiments.items() 
                           if sent == 'positive' and asp != 'not_mentioned']
        negative_aspects = [asp for asp, sent in aspect_sentiments.items() 
                           if sent == 'negative' and asp != 'not_mentioned']
        
        # Determine overall tone
        if len(positive_aspects) > len(negative_aspects):
            overall_tone = 'mostly_positive'
        elif len(negative_aspects) > len(positive_aspects):
            overall_tone = 'mostly_negative'
        else:
            overall_tone = 'mixed'
        
        # Opening based on overall tone
        if overall_tone == 'mostly_positive':
            response_parts.append("Thank you so much for your wonderful review!")
        elif overall_tone == 'mostly_negative':
            response_parts.append("We sincerely apologize for your disappointing experience.")
        else:
            response_parts.append("Thank you for taking the time to share your honest feedback.")
        
        # Address positive aspects first
        for aspect in positive_aspects[:2]:  # Mention top 2
            if aspect in self.aspect_templates:
                template = random.choice(self.aspect_templates[aspect]['positive'])
                response_parts.append(template)
        
        # Address negative aspects with solutions
        for aspect in negative_aspects:
            if aspect in self.aspect_templates:
                template = random.choice(self.aspect_templates[aspect]['negative'])
                response_parts.append(template)
        
        # Add appropriate closing
        closing = random.choice(self.closings[overall_tone])
        response_parts.append(closing)
        
        # Combine into coherent response
        response = ' '.join(response_parts)
        
        return response
    
    def generate_from_dataframe_row(self, row: pd.Series) -> str:
        """
        Generate response from a DataFrame row with aspect columns
        """
        # Extract aspect sentiments
        aspect_sentiments = {}
        aspects = ['food', 'service', 'ambiance', 'price', 'cleanliness']
        
        for aspect in aspects:
            col_name = f'{aspect}_sentiment'
            if col_name in row:
                aspect_sentiments[aspect] = row[col_name]
        
        # Generate response
        response = self.generate_aspect_based_response(
            aspect_sentiments=aspect_sentiments,
            overall_rating=row.get('reviewRating'),
            place_name=row.get('placeName')
        )
        
        return response
    
    def batch_generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate aspect-aware responses for entire dataset
        """
        print("Generating aspect-aware responses...")
        
        responses = []
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Generated {idx}/{len(df)} responses")
            
            response = self.generate_from_dataframe_row(row)
            responses.append(response)
        
        df['aspect_aware_response'] = responses
        
        return df


def compare_traditional_vs_aspect_responses(df: pd.DataFrame):
    """
    Compare quality of traditional vs aspect-aware responses
    """
    from QualityControl.responsequalitychecker import ResponseQualityScorer
    
    scorer = ResponseQualityScorer()
    
    print("\n" + "="*70)
    print("  COMPARING TRADITIONAL VS ASPECT-AWARE RESPONSES")
    print("="*70)
    
    # Score traditional responses
    print("\nScoring traditional responses...")
    traditional_scores = []
    for idx, row in df.iterrows():
        score = scorer.score_response(
            response=row.get('generated_response', ''),
            original_review=row.get('reviewText', ''),
            sentiment=row.get('sentiment', 'neutral'),
            rating=row.get('reviewRating')
        )
        traditional_scores.append(score['overall_score'])
    
    # Score aspect-aware responses
    print("Scoring aspect-aware responses...")
    aspect_scores = []
    for idx, row in df.iterrows():
        score = scorer.score_response(
            response=row.get('aspect_aware_response', ''),
            original_review=row.get('reviewText', ''),
            sentiment=row.get('sentiment', 'neutral'),
            rating=row.get('reviewRating')
        )
        aspect_scores.append(score['overall_score'])
    
    # Calculate improvements
    df['traditional_quality'] = traditional_scores
    df['aspect_quality'] = aspect_scores
    df['quality_improvement'] = df['aspect_quality'] - df['traditional_quality']
    
    # Summary statistics
    print("\n\nCOMPARISON RESULTS:")
    print("="*70)
    print(f"Traditional Average: {df['traditional_quality'].mean():.2f}/10")
    print(f"Aspect-Aware Average: {df['aspect_quality'].mean():.2f}/10")
    print(f"Average Improvement: +{df['quality_improvement'].mean():.2f} points")
    print(f"Reviews with Better Quality: {(df['quality_improvement'] > 0).sum()}/{len(df)} ({(df['quality_improvement'] > 0).mean()*100:.1f}%)")
    
    # Show examples
    print("\n\nEXAMPLE IMPROVEMENTS:")
    print("="*70)
    
    # Get top 3 improvements
    top_improvements = df.nlargest(3, 'quality_improvement')
    
    for i, (idx, row) in enumerate(top_improvements.iterrows(), 1):
        print(f"\nExample {i} (Improvement: +{row['quality_improvement']:.1f} points):")
        print(f"  Review: {row['reviewText'][:150]}...")
        print(f"  Rating: {row['reviewRating']}/5")
        print(f"  Aspects: Food={row.get('food_sentiment')}, Service={row.get('service_sentiment')}")
        print(f"\n  Traditional ({row['traditional_quality']:.1f}/10):")
        print(f"    {row['generated_response'][:200]}...")
        print(f"\n  Aspect-Aware ({row['aspect_quality']:.1f}/10):")
        print(f"    {row['aspect_aware_response'][:200]}...")
    
    return df


def main():
    """
    Complete comparison pipeline
    """
    # Load data with aspects
    df = pd.read_csv('data/processed/reviews_with_aspects.csv').head(100)
    
    # Generate aspect-aware responses
    generator = AspectAwareResponseGenerator()
    df = generator.batch_generate(df)
    
    # Compare quality
    df = compare_traditional_vs_aspect_responses(df)
    
    # Save results
    df.to_csv('model_pipeline/results/response_comparison.csv', index=False)
    
    print("\n" + "="*70)
    print("  Analysis complete!")
    print(f"  Results saved to: model_pipeline/results/response_comparison.csv")
    print("="*70)


if __name__ == "__main__":
    main()