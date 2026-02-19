"""
Benchmark current TF-IDF sentiment model
"""
import sys
sys.path.append('../model_pipeline')

from inference_pipeline import EchoAIInference
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json

# Test cases with ground truth
test_reviews = [
    # Terrible examples
    {"text": "Worst experience ever! Disgusting food, rude staff.", "true_label": "terrible"},
    {"text": "Absolutely horrible. Never coming back.", "true_label": "terrible"},
    {"text": "Terrible service and cold food. Waste of money.", "true_label": "terrible"},
    
    # Negative examples
    {"text": "Disappointing. Food was cold and service slow.", "true_label": "negative"},
    {"text": "Not great. Below average experience.", "true_label": "negative"},
    {"text": "Food quality has declined. Not impressed.", "true_label": "negative"},
    
    # Neutral examples
    {"text": "It was okay. Nothing special.", "true_label": "neutral"},
    {"text": "Average experience. Decent food.", "true_label": "neutral"},
    {"text": "Acceptable but not memorable.", "true_label": "neutral"},
    
    # Positive examples
    {"text": "Good food and friendly service!", "true_label": "positive"},
    {"text": "Really enjoyed the meal. Would recommend.", "true_label": "positive"},
    {"text": "Great experience overall!", "true_label": "positive"},
    
    # Amazing examples
    {"text": "Absolutely phenomenal! Best restaurant ever!", "true_label": "amazing"},
    {"text": "Mind-blowing food! Cannot recommend enough!", "true_label": "amazing"},
    {"text": "Perfect in every way. 10/10!", "true_label": "amazing"},
]

def benchmark():
    # Load model
    pipeline = EchoAIInference()
    pipeline.load_models(load_llm=False)
    
    # Predict
    predictions = []
    true_labels = []
    
    for review in test_reviews:
        result = pipeline.predict_sentiment(review['text'])
        predictions.append(result['sentiment'])
        true_labels.append(review['true_label'])
        
        print(f"Review: {review['text'][:50]}...")
        print(f"  True: {review['true_label']}, Predicted: {result['sentiment']}, Confidence: {result.get('confidence', 0):.2f}")
        print()
    
    # Metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    cm = confusion_matrix(true_labels, predictions, 
                         labels=['terrible', 'negative', 'neutral', 'positive', 'amazing'])
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, predictions))
    
    print("\nCONFUSION MATRIX")
    print(cm)
    
    # Save results
    with open('model_pipeline/results/baseline_tfidf.json', 'w') as f:
        json.dump({
            'model': 'TF-IDF + LogisticRegression',
            'test_size': len(test_reviews),
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': report
        }, f, indent=2)
    
    print(f"\n✓ Results saved to model_pipeline/results/baseline_tfidf.json")

if __name__ == "__main__":
    benchmark()