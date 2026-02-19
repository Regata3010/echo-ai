"""
Complete Response Quality Analysis Pipeline
1. Generate responses for reviews
2. Score response quality
3. Create comparison visualizations
4. Generate actionable insights
"""
import sys
sys.path.append('..')

import pandas as pd
import logging
from pathlib import Path
sys.path.insert(0,"..")
sys.path.insert(0,"QualityControl")
from responsequalitychecker import ResponseQualityScorer
from inference_pipeline import EchoAIInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_and_score_responses(input_csv='../../data/raw/featurecleaned_apify_processed.csv',
                                output_csv='model_pipeline/results/responses_with_quality.csv'):
    """
    Complete pipeline: generate responses → score quality → save results
    """
    
    print("\n" + "="*70)
    print("  RESPONSE GENERATION AND QUALITY SCORING PIPELINE")
    print("="*70)
    
    # Load reviews
    logger.info("\nStep 1: Loading reviews...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} reviews")
    
    # Take sample for testing (remove .head() for full dataset)
    df_sample = df.head(100)  # Start with 100 reviews
    logger.info(f"Processing {len(df_sample)} reviews for demo")
    
    # Step 2: Generate responses
    logger.info("\nStep 2: Generating responses...")
    pipeline = EchoAIInference()
    pipeline.load_models(load_llm=False)  # Use templates, not LLM
    
    df_with_responses = pipeline.process_dataframe(df_sample, generate_responses=True)
    
    # Step 3: Score response quality
    logger.info("\nStep 3: Scoring response quality...")
    scorer = ResponseQualityScorer()
    df_scored, all_scores = scorer.evaluate_batch_responses(df_with_responses)
    
    # Step 4: Generate quality report
    logger.info("\nStep 4: Generating quality report...")
    quality_report = scorer.generate_quality_report(all_scores)
    
    # Step 5: Create visualizations
    logger.info("\nStep 5: Creating visualizations...")
    scorer.create_quality_visualizations(df_scored, save_dir='model_pipeline/results/')
    
    # Step 6: Save results
    df_scored.to_csv(output_csv, index=False)
    
    import json
    with open('model_pipeline/results/response_quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    # Print detailed analysis
    print("\n\n" + "="*70)
    print("  RESPONSE QUALITY ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nTotal Responses Analyzed: {quality_report['total_responses']}")
    print(f"Average Quality Score: {quality_report['average_quality_score']}/10")
    
    print(f"\nGrade Distribution:")
    for grade in ['A+', 'A', 'B', 'C', 'D', 'F']:
        count = quality_report['grade_distribution'].get(grade, 0)
        pct = (count / quality_report['total_responses']) * 100 if count > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {grade:3} : {count:3} ({pct:5.1f}%) {bar}")
    
    print(f"\nDimension Performance:")
    for dim, score in quality_report['dimension_averages'].items():
        status = "[EXCELLENT]" if score >= 8 else "[GOOD]" if score >= 7 else "[NEEDS WORK]"
        bar_length = int(score)
        bar = '█' * bar_length + '░' * (10 - bar_length)
        print(f"  {dim.capitalize():15} : {score:4.1f}/10 {bar} {status}")
    
    if quality_report['improvement_areas']:
        print(f"\n\nPRIORITY IMPROVEMENTS:")
        for i, area in enumerate(quality_report['improvement_areas'], 1):
            print(f"  {i}. {area['dimension'].capitalize()}: {area['current_score']:.1f}/10")
            print(f"     Need to improve by {area['gap']:.1f} points to reach target (8.0)")
    
    print(f"\n\nBEST RESPONSE EXAMPLE (Score: {quality_report['best_response']['score']}/10, Grade: {quality_report['best_response']['grade']}):")
    print(f"  Review: {quality_report['best_response']['review']}")
    print(f"  Response: {quality_report['best_response']['response']}")
    
    print(f"\n\nWORST RESPONSE EXAMPLE (Score: {quality_report['worst_response']['score']}/10, Grade: {quality_report['worst_response']['grade']}):")
    print(f"  Review: {quality_report['worst_response']['review']}")
    print(f"  Response: {quality_report['worst_response']['response']}")
    print(f"  NEEDS: More specific, actionable, and empathetic")
    
    print("\n" + "="*70)
    print(f"  Results saved to: {output_csv}")
    print(f"  Visualizations: model_pipeline/results/quality_*.png")
    print("="*70 + "\n")
    
    return df_scored, quality_report


if __name__ == "__main__":
    generate_and_score_responses()