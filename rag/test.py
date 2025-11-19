import os

# Ensure OpenMP only initializes once (macOS FAISS/Torch clash)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

from dotenv import load_dotenv

from load_products import load_products
from create_vector_store import create_vector_store
from query import query_vector_store
from ingest import create_documents
from format_data import format_search_results
from analazye_promt import analyse_promt
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not set")
    print("   Create .env file with: OPENAI_API_KEY=sk-your-key")
    exit(1)
    
# Load and process
# inital_query = "coffee maker"
inital_query = "Best running shoes under $200"
products_path = Path(__file__).parent.parent / "data" / "products.json"

products = load_products(products_path)
documents = create_documents(products)
vectorstore = create_vector_store(documents)
analyse_promt_data = analyse_promt(inital_query)

results = query_vector_store(analyse_promt_data, vectorstore)
results = format_search_results(results)
print(json.dumps(results, indent=2, ensure_ascii=False))