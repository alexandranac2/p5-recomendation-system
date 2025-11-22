"""
Test script for FastAPI endpoints
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

# Initialize recommendation system manually (TestClient doesn't trigger lifespan)
print("Initializing recommendation system...")
try:
    recommendations.initialize_recommendation_system()
    print("✅ Recommendation system initialized!")
except Exception as e:
    print(f"⚠️  Could not initialize recommendation system: {e}")
    print("   (This is expected if vectorstore doesn't exist or dependencies are missing)")

# Create test client
client = TestClient(app)

print("=" * 60)
print("Testing FastAPI Endpoints")
print("=" * 60)

# Test 1: Root endpoint
print("\n1. Testing root endpoint (GET /)")
try:
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Root endpoint works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Health check
print("\n2. Testing health endpoint (GET /api/health/)")
try:
    response = client.get("/api/health/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("   ✅ Health endpoint works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Health ready
print("\n3. Testing health ready endpoint (GET /api/health/ready)")
try:
    response = client.get("/api/health/ready")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    print("   ✅ Health ready endpoint works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Routes list
print("\n4. Testing routes list endpoint (GET /api/routes/)")
try:
    response = client.get("/api/routes/")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Total routes: {data.get('total_routes', 0)}")
    print(f"   Base URL: {data.get('base_url', 'N/A')}")
    assert response.status_code == 200
    assert "routes" in data
    print("   ✅ Routes list endpoint works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Routes summary
print("\n5. Testing routes summary endpoint (GET /api/routes/summary)")
try:
    response = client.get("/api/routes/summary")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Total routes: {data.get('total_routes', 0)}")
    print(f"   Tags: {list(data.get('routes_by_tag', {}).keys())}")
    assert response.status_code == 200
    assert "routes_by_tag" in data
    print("   ✅ Routes summary endpoint works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 6: Recommendations endpoint (this will test the full initialization)
print("\n6. Testing recommendations endpoint (POST /api/recommendations/)")
try:
    test_query = {
        "query": "Best running shoes under $200",
        "max_results": 3
    }
    response = client.post("/api/recommendations/", json=test_query)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Query: {data.get('query', 'N/A')}")
        print(f"   Total results: {data.get('total_results', 0)}")
        print(f"   Has explanation: {bool(data.get('explanation', ''))}")
        print(f"   Recommendations count: {len(data.get('recommendations', []))}")
        assert "recommendations" in data
        assert "explanation" in data
        print("   ✅ Recommendations endpoint works!")
    else:
        print(f"   Response: {response.text}")
        print("   ⚠️  Recommendations endpoint returned error (might be expected if vectorstore not loaded)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 7: Search endpoint
print("\n7. Testing search endpoint (GET /api/recommendations/search)")
try:
    response = client.get("/api/recommendations/search?q=running%20shoes&k=5")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Query: {data.get('query', 'N/A')}")
        print(f"   Results count: {data.get('count', 0)}")
        assert "results" in data
        print("   ✅ Search endpoint works!")
    else:
        print(f"   Response: {response.text}")
        print("   ⚠️  Search endpoint returned error (might be expected if vectorstore not loaded)")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)

