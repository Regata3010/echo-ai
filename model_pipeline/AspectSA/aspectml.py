# """
# Run complete pipeline with ABSA comparison
# Shows difference between traditional and aspect-based approaches
# """
# import sys
# sys.path.append('..')

# from aspectsentiment import AspectSentimentAnalyzer, compare_overall_vs_aspect_sentiment
# from comparesentiments import create_comparison_visualizations, find_interesting_insights
# import pandas as pd
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def run_complete_absa_analysis():
#     """
#     Complete ABSA pipeline with comparison
#     """
    
#     print("\n" + "="*70)
#     print("  ASPECT-BASED SENTIMENT ANALYSIS PIPELINE")
#     print("="*70)
    
#     # Step 1: Load processed data
#     logger.info("\nStep 1: Loading processed reviews...")
#     df = pd.read_csv('/Users/Sem End MLOPS/echo-ai/data/raw/featurecleaned_apify_processed.csv')
#     logger.info(f"Loaded {len(df)} reviews")
    
#     # Step 2: Run traditional sentiment analysis summary
#     logger.info("\nStep 2: Traditional Sentiment Analysis Summary...")
#     traditional_summary = {
#         'total_reviews': len(df),
#         'rating_distribution': df['reviewRating'].value_counts().sort_index().to_dict(),
#         'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
#         'avg_rating': df['reviewRating'].mean()
#     }
    
#     print("\nTRADITIONAL SENTIMENT ANALYSIS:")
#     print(f"  Total Reviews: {traditional_summary['total_reviews']}")
#     print(f"  Average Rating: {traditional_summary['avg_rating']:.2f}")
#     print(f"  Sentiment Distribution:")
#     for sent, count in traditional_summary['sentiment_distribution'].items():
#         pct = (count / traditional_summary['total_reviews']) * 100
#         print(f"    {sent}: {count} ({pct:.1f}%)")
    
#     # Step 3: Run ABSA
#     logger.info("\nStep 3: Running Aspect-Based Sentiment Analysis...")
#     analyzer = AspectSentimentAnalyzer()
#     df_with_aspects = analyzer.analyze_batch(df)
    
#     # Step 4: Generate aspect report
#     logger.info("\nStep 4: Generating aspect analysis report...")
#     aspect_report = analyzer.generate_aspect_report(df_with_aspects)
    
#     print("\n\nASPECT-BASED SENTIMENT ANALYSIS:")
#     for aspect, data in aspect_report['aspect_analysis'].items():
#         print(f"\n{aspect.upper()}:")
#         print(f"  Mentioned: {data['mention_rate']}% of reviews")
#         print(f"  Positive: {data['positive_count']}")
#         print(f"  Negative: {data['negative_count']}")
#         print(f"  Neutral: {data['neutral_count']}")
#         print(f"  Avg Score: {data['avg_sentiment_score']}")
    
#     # Step 5: Compare approaches
#     logger.info("\nStep 5: Comparing traditional vs aspect-based...")
#     comparison = compare_overall_vs_aspect_sentiment(df_with_aspects)
    
#     print("\n\nCOMPARISON ANALYSIS:")
#     print(f"  Agreement Rate: {comparison['agreement_rate']}%")
#     print(f"  Total Disagreements: {comparison['disagreements']}")
    
#     # Step 6: Find insights
#     logger.info("\nStep 6: Extracting insights...")
#     insights = find_interesting_insights(df_with_aspects)
    
#     print("\n\nKEY INSIGHTS:")
#     for insight in insights:
#         print(f"\n  {insight['description']}")
#         if 'count' in insight:
#             print(f"    Found in {insight['count']} reviews")
    
#     # Show specific examples
#     if comparison['disagreement_examples']:
#         print("\n\nEXAMPLE DISAGREEMENTS (Why ABSA is Better):")
#         print("="*70)
        
#         for i, ex in enumerate(comparison['disagreement_examples'][:3], 1):
#             print(f"\nExample {i}:")
#             print(f"  Review: \"{ex['review'][:200]}...\"")
#             print(f"  Overall Rating: {ex['overall_rating']} stars → {ex['overall_sentiment']}")
#             print(f"  ABSA Aggregate: {ex['aspect_sentiment']}")
#             print(f"  Breakdown:")
#             print(f"    - Food: {ex.get('food', 'not mentioned')}")
#             print(f"    - Service: {ex.get('service', 'not mentioned')}")
#             print(f"    - Price: {ex.get('price', 'not mentioned')}")
#             print(f"  WHY ABSA HELPS: Can see specific pain points even in mixed reviews")
    
#     # Step 7: Create visualizations
#     logger.info("\nStep 7: Creating visualizations...")
#     create_comparison_visualizations(df_with_aspects)
    
#     # Step 8: Save results
#     logger.info("\nStep 8: Saving results...")
#     df_with_aspects.to_csv('/Users/Sem End MLOPS/echo-ai/data/raw/comparison.csv', index=False)
    
#     import json
#     with open('/Users/Sem End MLOPS/echo-ai/Model-Pipeline/results/absa_comparison_report.json', 'w') as f:
#         json.dump({
#             'traditional': traditional_summary,
#             'aspect_analysis': aspect_report,
#             'comparison': comparison,
#             'insights': insights
#         }, f, indent=2, default=str)
    
#     print("\n" + "="*70)
#     print("  ANALYSIS COMPLETE")
#     print("="*70)
#     print("\nResults saved to:")
#     print("  - data/processed/reviews_with_aspects.csv")
#     print("  - Model-Pipeline/results/absa_comparison_report.json")
#     print("  - Model-Pipeline/results/aspect_*.png (visualizations)")
#     print("\n" + "="*70)
    
#     return df_with_aspects, aspect_report, comparison, insights


# if __name__ == "__main__":
#     run_complete_absa_analysis()