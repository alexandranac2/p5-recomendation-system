import sys
from pathlib import Path

# Add project root to Python path BEFORE imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import after path is set up (must be after sys.path modification)
from rag.create_vector_store import create_load_vector_store  # noqa: E402
from simple_rag import build_simple_recommendation_graph  # noqa: E402

# Setup
name = "alexs_vectorstore"
products_path = Path(__file__).parent.parent / "data" / "products.json"
vectorstore = create_load_vector_store(name=name, products_path=products_path)

# Build simple graph (returns a function)
recommend = build_simple_recommendation_graph(vectorstore)

# Test query
query = "Best running shoes under $200"
print(f"\n🔍 Query: {query}\n")

result = recommend(query)

# Display results
print(f"\n🎯 Recommendations ({len(result['recommendations'])}):")
for i, rec in enumerate(result["recommendations"][:5], 1):
    print(f"   {i}. {rec.get('name')} - ${rec.get('price', 0)}")

print(f"\n💡 {result['explanation']}")

