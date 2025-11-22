"""
Quick test to verify /docs endpoint works
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

print("Testing /docs endpoint...")
print("=" * 60)

# Test if docs endpoint exists
response = client.get("/docs")
print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")

if response.status_code == 200:
    print("✅ /docs endpoint is working!")
    print(f"   Response length: {len(response.text)} characters")
    if "swagger" in response.text.lower() or "openapi" in response.text.lower():
        print("   ✅ Swagger UI content detected")
else:
    print(f"❌ /docs endpoint returned: {response.status_code}")
    print(f"   Response: {response.text[:200]}")

# Test OpenAPI JSON
print("\nTesting /openapi.json endpoint...")
response = client.get("/openapi.json")
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    print("✅ /openapi.json endpoint is working!")
    data = response.json()
    print(f"   API Title: {data.get('info', {}).get('title', 'N/A')}")
    print(f"   API Version: {data.get('info', {}).get('version', 'N/A')}")
    print(f"   Number of paths: {len(data.get('paths', {}))}")
else:
    print(f"❌ /openapi.json endpoint returned: {response.status_code}")

print("\n" + "=" * 60)
print("To start the server, run: python run_api.py")
print("Then visit: http://localhost:8000/docs")

