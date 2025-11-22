import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rag.create_vector_store import create_load_vector_store
from rag.agent import RecommendationAgent


# Setup
name = "alexs_vectorstore"
products_path = Path(__file__).parent.parent / "data" / "products.json"
vectorstore = create_load_vector_store(name=name, products_path=products_path)

# Create agent
agent = RecommendationAgent(vectorstore)

# Test query
query = "Best running shoes under $200"
print(f"\n🔍 Query: {query}\n")

result = agent.recommend(query)

# Display results
print(f"\n📊 Intent: {result['intent']}")
print(f"\n🎯 Recommendations ({len(result['recommendations'])}):")
for i, rec in enumerate(result["recommendations"][:5], 1):
    print(f"   {i}. {rec.get('name')} - ${rec.get('price', 0)}")

print(f"\n💡 {result['explanation']}")

