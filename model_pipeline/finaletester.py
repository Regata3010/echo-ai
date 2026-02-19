edge_cases = [
    # 1. Sarcasm
    {
        "text": "Oh great, another cold burger. Just wonderful.",
        "expected_overall": "negative",
        "expected_food": "negative",
        "challenge": "Sarcasm - 'great' and 'wonderful' are actually negative"
    },
    
    # 2. Negation stacking
    {
        "text": "Not bad at all, but wasn't particularly great either.",
        "expected_overall": "neutral",
        "expected_food": "neutral",
        "challenge": "Double negation - 'not bad' but 'wasn't great'"
    },
    
    # 3. Extreme mixed sentiment
    {
        "text": "Best pizza I've ever had in my life, but the waiter was incredibly rude and ruined the entire experience.",
        "expected_overall": "negative",
        "expected_food": "positive",
        "expected_service": "negative",
        "challenge": "Amazing food, terrible service - overall should lean negative"
    },
    
    # 4. Subtle negative
    {
        "text": "It was fine I guess. Nothing to complain about really.",
        "expected_overall": "neutral",
        "challenge": "Lukewarm language - 'fine', 'I guess' suggests dissatisfaction"
    },
    
    # 5. Context-dependent words
    {
        "text": "Small intimate space with cozy atmosphere, but tiny portions that left us hungry.",
        "expected_ambiance": "positive",
        "expected_food": "negative",
        "challenge": "'Small' is positive for ambiance, negative for portions"
    },
    
    # 6. Misspellings and typos
    {
        "text": "Amazzing food! Servise was grate too. Definately recomend!!!",
        "expected_overall": "positive",
        "expected_food": "positive",
        "expected_service": "positive",
        "challenge": "Typos and misspellings - 'amazzing', 'servise', 'grate'"
    },
    
    # 7. Comparison with past experience
    {
        "text": "Used to be my favorite spot. Last three visits have been disappointing. Quality has really gone downhill.",
        "expected_overall": "negative",
        "expected_food": "negative",
        "challenge": "Temporal sentiment shift - was good, now bad"
    },
    
    # 8. Backhanded compliment
    {
        "text": "For the price, it's acceptable. Don't expect fine dining.",
        "expected_overall": "neutral",
        "expected_price": "negative",
        "challenge": "Damning with faint praise - sounds neutral but implies disappointment"
    },
    
    # 9. Health/safety concern
    {
        "text": "Found a hair in my salad. Otherwise the food was decent.",
        "expected_overall": "terrible",
        "expected_food": "negative",
        "expected_cleanliness": "negative",
        "challenge": "Health violation should override 'decent' food"
    },
    
    # 10. Multiple aspects, all negative but varying intensity
    {
        "text": "Service was slow, food was cold, place was dirty, and prices are outrageous. Complete disaster.",
        "expected_overall": "terrible",
        "expected_food": "negative",
        "expected_service": "negative",
        "expected_cleanliness": "negative",
        "expected_price": "negative",
        "challenge": "Multiple problems - system should flag as urgent/high priority"
    }
]

import sys
sys.path.append('.')

from tetser_inference_pipeline import EnhancedInferencePipeline

def test_edge_cases():
    # Load pipeline
    pipeline = EnhancedInferencePipeline(use_bert=True, use_enhanced_absa=True)
    pipeline.load_models(load_llm=False)
    
    print("="*80)
    print("EDGE CASE TESTING")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n[TEST {i}] {case['challenge']}")
        print("-"*80)
        print(f"Review: {case['text']}")
        
        result = pipeline.process_review(case['text'], generate_response=True)
        
        if result['status'] == 'success':
            overall = result['sentiment_analysis']['sentiment']
            aspects = result['aspect_analysis']['aspect_sentiments']
            
            print(f"\nPredicted Overall: {overall}")
            print(f"Expected Overall: {case.get('expected_overall', 'N/A')}")
            
            print("\nAspect Breakdown:")
            for aspect, data in aspects.items():
                if data['sentiment'] != 'not_mentioned':
                    expected_key = f'expected_{aspect}'
                    expected = case.get(expected_key, 'N/A')
                    match = "PASS" if data['sentiment'] == expected or expected == 'N/A' else "FAIL"
                    print(f"  {aspect}: {data['sentiment']} (expected: {expected}) [{match}]")
            
            print(f"\nGenerated Response:")
            print(f"{result['generated_response']}")
            
            # Simple pass/fail
            overall_match = overall == case.get('expected_overall', overall)
            if overall_match:
                passed += 1
                print("\nResult: PASS")
            else:
                failed += 1
                print("\nResult: FAIL")
        else:
            print(f"ERROR: {result.get('error')}")
            failed += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(edge_cases)}")
    print(f"Failed: {failed}/{len(edge_cases)}")
    print(f"Success Rate: {passed/len(edge_cases)*100:.1f}%")

if __name__ == "__main__":
    test_edge_cases()