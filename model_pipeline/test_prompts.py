"""
Quick test to verify prompt has few-shot examples
"""
from response_generator import ResponseGenerator

gen = ResponseGenerator(model_name='google/flan-t5-base')

# Create a test prompt
prompt = gen.create_prompt(
    reviewText="Great food!",
    sentiment="amazing",
    placeName="Test Restaurant"
)

print("="*70)
print("PROMPT VERIFICATION")
print("="*70)
print(prompt[:500])  # First 500 chars
print("...")
print("="*70)

# Check if few-shot examples are in prompt
if "Example 1:" in prompt:
    print("✅ FEW-SHOT EXAMPLES FOUND IN PROMPT")
else:
    print("❌ FEW-SHOT EXAMPLES MISSING FROM PROMPT")