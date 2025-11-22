"""
Test the recommendations API endpoint with a real query
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from api.main import app
from api.routes import recommendations

print("=" * 60)
print("Testing Recommendations API with Real Query")
print("=" * 60)

# Initialize recommendation system
print("\n🚀 Initializing recommendation system...")
try:
    recommendations.initialize_recommendation_system()
    print("✅ Recommendation system initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    print("\n⚠️  Make sure:")
    print("   - Vectorstore exists at alexs_vectorstore/")
    print("   - All dependencies are installed")
    print("   - OPENAI_API_KEY is set in environment")
    sys.exit(1)

# Create test client
client = TestClient(app)

# Test query
test_queries = [
    "Best running shoes under $200",
    "I need a laptop for gaming",
    "Wireless headphones with noise cancellation"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"Test Query {i}: {query}")
    print(f"{'='*60}")
    
    try:
        # Make POST request to recommendations endpoint
        response = client.post(
            "/api/recommendations/",
            json={
                "query": query,
                "max_results": 5
            }
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Success!")
            print(f"\nQuery: {data.get('query', 'N/A')}")
            print(f"Total Results: {data.get('total_results', 0)}")
            print(f"\nExplanation:")
            print(f"  {data.get('explanation', 'N/A')}")
            
            print(f"\nRecommendations ({len(data.get('recommendations', []))}):")
            for j, rec in enumerate(data.get('recommendations', [])[:5], 1):
                print(f"  {j}. {rec.get('name', 'N/A')}")
                print(f"     Price: ${rec.get('price', 0)}")
                print(f"     Category: {rec.get('category', 'N/A')}")
                if rec.get('description'):
                    desc = rec.get('description', '')[:100]
                    print(f"     Description: {desc}...")
                print()
            
            if data.get('intent'):
                print(f"\nDetected Intent:")
                intent = data.get('intent', {})
                if intent.get('product'):
                    print(f"  Product: {intent.get('product')}")
                if intent.get('category'):
                    print(f"  Category: {intent.get('category')}")
                if intent.get('price_range'):
                    price_range = intent.get('price_range', {})
                    print(f"  Price Range: ${price_range.get('min', 0)} - ${price_range.get('max', '∞')}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)

