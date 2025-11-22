import sys
from pathlib import Path

# Add project root to Python path FIRST, before any imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import after path is set
from config.settings import settings  # noqa: E402
from rag.create_vector_store import create_load_vector_store  # noqa: E402
from rag.agent import build_recommendation_graph  # noqa: E402



# Setup - using config settings
products_path = Path(__file__).parent.parent / "data" / "products.json"
vectorstore = create_load_vector_store(
    name=settings.VECTOR_STORE_NAME,
    products_path=products_path
)

# Build graph (returns a function)
recommend = build_recommendation_graph(vectorstore)

# Test query
# query = "Best running shoes under $200"
query = "food for my pet"
print(f"\n🔍 Query: {query}\n")

result = recommend(query)

# Display results
print(f"\n📊 Intent: {result['intent']}")
print(f"\n🎯 Recommendations ({len(result['recommendations'])}):")
for i, rec in enumerate(result["recommendations"][:5], 1):
    print(f"   {i}. {rec.get('name')} - ${rec.get('price', 0)}")

print(f"\n💡 {result['explanation']}")
