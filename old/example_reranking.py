"""
Example demonstrating how to use reranking after vector store queries.

Reranking improves search relevance by using cross-encoder models that
consider both the query and document content together, rather than just
embedding similarity.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from load_products import load_products
from create_vector_store import create_vector_store
from ingest import create_documents
from query import query_vector_store
from format_data import format_search_results
from reranker import create_reranker
from analazye_promt import analyse_promt

# Ensure OpenMP only initializes once (macOS FAISS/Torch clash)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not set")
    exit(1)


def example_basic_reranking():
    """Basic example: Query with reranking enabled (default)."""
    print("\n" + "="*60)
    print("Example 1: Basic Reranking")
    print("="*60)
    
    query = "coffee maker"
    products_path = Path(__file__).parent.parent / "data" / "products.json"
    
    products = load_products(products_path)
    documents = create_documents(products)
    vectorstore = create_vector_store(documents)
    analysed_query = analyse_promt(query)
    
    # Query with reranking (default: use_reranking=True)
    results = query_vector_store(
        analysed_query,
        vectorstore,
        k=5,                    # Return top 5 results
        use_reranking=True,     # Enable reranking
        rerank_top_k=20         # Fetch 20 candidates for reranking
    )
    
    formatted = format_search_results(results, is_reranked=True)
    
    print(f"\n📊 Top {len(formatted)} results after reranking:")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Score: {result['score']:.4f} ({result['score_type']})")
        print(f"   Category: {result['category']}")
        print(f"   Price: ${result['price']}")


def example_without_reranking():
    """Example: Query without reranking for comparison."""
    print("\n" + "="*60)
    print("Example 2: Without Reranking (for comparison)")
    print("="*60)
    
    query = "coffee maker"
    products_path = Path(__file__).parent.parent / "data" / "products.json"
    
    products = load_products(products_path)
    documents = create_documents(products)
    vectorstore = create_vector_store(documents)
    analysed_query = analyse_promt(query)
    
    # Query without reranking
    results = query_vector_store(
        analysed_query,
        vectorstore,
        k=5,
        use_reranking=False  # Disable reranking
    )
    
    formatted = format_search_results(results, is_reranked=False)
    
    print(f"\n📊 Top {len(formatted)} results (no reranking):")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Score: {result['score']:.4f} ({result['score_type']})")
        print(f"   Category: {result['category']}")


def example_reuse_reranker():
    """Example: Reuse reranker instance for better performance."""
    print("\n" + "="*60)
    print("Example 3: Reusing Reranker Instance")
    print("="*60)
    
    # Create reranker once (model loading is expensive)
    print("🔄 Creating reranker instance...")
    reranker = create_reranker()
    
    query = "coffee maker"
    products_path = Path(__file__).parent.parent / "data" / "products.json"
    
    products = load_products(products_path)
    documents = create_documents(products)
    vectorstore = create_vector_store(documents)
    analysed_query = analyse_promt(query)
    
    # Multiple queries reusing the same reranker
    queries = ["coffee maker", "gaming laptop", "wireless headphones"]
    
    for q in queries:
        analysed_q = analyse_promt(q)
        results = query_vector_store(
            analysed_q,
            vectorstore,
            k=3,
            use_reranking=True,
            reranker=reranker  # Reuse the same reranker instance
        )
        
        formatted = format_search_results(results, is_reranked=True)
        print(f"\n🔍 Query: '{q}'")
        print(f"   Top result: {formatted[0]['name']} (score: {formatted[0]['score']:.4f})")


def example_custom_reranker_model():
    """Example: Using a different reranker model."""
    print("\n" + "="*60)
    print("Example 4: Custom Reranker Model")
    print("="*60)
    
    # Use a larger, more accurate model (slower but better quality)
    reranker = create_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    
    query = "coffee maker"
    products_path = Path(__file__).parent.parent / "data" / "products.json"
    
    products = load_products(products_path)
    documents = create_documents(products)
    vectorstore = create_vector_store(documents)
    analysed_query = analyse_promt(query)
    
    results = query_vector_store(
        analysed_query,
        vectorstore,
        k=5,
        reranker=reranker
    )
    
    formatted = format_search_results(results, is_reranked=True)
    print(f"\n📊 Results with custom reranker model:")
    for i, result in enumerate(formatted, 1):
        print(f"{i}. {result['name']} (score: {result['score']:.4f})")


if __name__ == "__main__":
    # Run examples
    example_basic_reranking()
    example_without_reranking()
    example_reuse_reranker()
    example_custom_reranker_model()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)

