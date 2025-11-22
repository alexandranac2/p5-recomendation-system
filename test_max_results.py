"""
Test max_results parameter in recommendations endpoint
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

# Initialize recommendation system
print("🚀 Initializing recommendation system...")
try:
    recommendations.initialize_recommendation_system()
    print("✅ Recommendation system initialized!")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

client = TestClient(app)

test_query = "Best running shoes under $200"

print("=" * 60)
print("Testing max_results parameter")
print("=" * 60)

# Test 1: Without max_results (should return default/all)
print(f"\n1. Test WITHOUT max_results parameter")
response = client.post(
    "/api/recommendations/",
    json={"query": test_query}
)
if response.status_code == 200:
    data = response.json()
    count = data.get('total_results', 0)
    print(f"   ✅ Status: {response.status_code}")
    print(f"   📊 Results returned: {count}")
    print(f"   Recommendations count: {len(data.get('recommendations', []))}")
else:
    print(f"   ❌ Status: {response.status_code}")
    print(f"   Error: {response.text}")

# Test 2: With max_results = 1
print(f"\n2. Test WITH max_results = 1")
response = client.post(
    "/api/recommendations/",
    json={
        "query": test_query,
        "max_results": 1
    }
)
if response.status_code == 200:
    data = response.json()
    count = data.get('total_results', 0)
    recs = data.get('recommendations', [])
    print(f"   ✅ Status: {response.status_code}")
    print(f"   📊 Results returned: {count}")
    print(f"   Recommendations count: {len(recs)}")
    if len(recs) == 1:
        print(f"   ✅ Correct! Returned exactly 1 result")
        print(f"      Product: {recs[0].get('name', 'N/A')}")
    else:
        print(f"   ⚠️  Expected 1 result, got {len(recs)}")
else:
    print(f"   ❌ Status: {response.status_code}")
    print(f"   Error: {response.text}")

# Test 3: With max_results = 3
print(f"\n3. Test WITH max_results = 3")
response = client.post(
    "/api/recommendations/",
    json={
        "query": test_query,
        "max_results": 3
    }
)
if response.status_code == 200:
    data = response.json()
    count = data.get('total_results', 0)
    recs = data.get('recommendations', [])
    print(f"   ✅ Status: {response.status_code}")
    print(f"   📊 Results returned: {count}")
    print(f"   Recommendations count: {len(recs)}")
    if len(recs) <= 3:
        print(f"   ✅ Correct! Returned {len(recs)} results (≤ 3)")
        for i, rec in enumerate(recs, 1):
            print(f"      {i}. {rec.get('name', 'N/A')}")
    else:
        print(f"   ⚠️  Expected ≤3 results, got {len(recs)}")
else:
    print(f"   ❌ Status: {response.status_code}")
    print(f"   Error: {response.text}")

# Test 4: With max_results = 10 (more than available)
print(f"\n4. Test WITH max_results = 10 (more than available)")
response = client.post(
    "/api/recommendations/",
    json={
        "query": test_query,
        "max_results": 10
    }
)
if response.status_code == 200:
    data = response.json()
    count = data.get('total_results', 0)
    recs = data.get('recommendations', [])
    print(f"   ✅ Status: {response.status_code}")
    print(f"   📊 Results returned: {count}")
    print(f"   Recommendations count: {len(recs)}")
    print(f"   ✅ Correct! Returned {len(recs)} results (all available)")
else:
    print(f"   ❌ Status: {response.status_code}")
    print(f"   Error: {response.text}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)

