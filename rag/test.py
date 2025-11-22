import json

# from format_data import format_search_results

# # Ensure OpenMP only initializes once (macOS FAISS/Torch clash)
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path


from create_vector_store import create_load_vector_store
from query import query_vector_store


# Load and process
# inital_query = "coffee maker"
# inital_query = "Best running shoes under $200"
inital_query = "food for my pet"
name = "alexs_vectorstore"
products_path = Path(__file__).parent.parent / "data" / "products.json"

# Automatically loads products and creates documents if vectorstore doesn't exist
vectorstore = create_load_vector_store(name=name, products_path=products_path)

results = query_vector_store(
    inital_query, vectorstore=vectorstore, k=10, format_results=True, max_score=1.3
)

print(json.dumps(results, indent=2, ensure_ascii=False))
