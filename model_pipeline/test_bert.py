"""
Test pre-trained BERT sentiment model
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BERTSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        pred_class = torch.argmax(probs).item()
        confidence = probs[0][pred_class].item()
        
        sentiment_map = {0: 'terrible', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'amazing'}
        
        return {
            'sentiment': sentiment_map[pred_class],
            'confidence': confidence
        }

# Test on same examples
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


bert = BERTSentiment()

correct = 0
for review in test_reviews:
    result = bert.predict(review['text'])
    match = result['sentiment'] == review['true_label']
    if match:
        correct += 1
    
    symbol = '✓' if match else '✗'
    print(f"{symbol} {review['text'][:50]}...")
    print(f"   True: {review['true_label']}, Predicted: {result['sentiment']}, Confidence: {result['confidence']:.2f}\n")

print(f"\nBERT Accuracy: {correct}/{len(test_reviews)} = {correct/len(test_reviews)*100:.1f}%")