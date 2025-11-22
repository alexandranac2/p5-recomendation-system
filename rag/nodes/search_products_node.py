from rag.agent.state import AgentState
from rag.query import query_vector_store
from config.settings import settings


def search_products_node(state: AgentState, vectorstore) -> AgentState:
    """Node 2: Search products using vector store"""
    print("🔍 Searching products...")
    
    intent = state["analyzed_intent"]
    
    # Build search query
    if intent:
        search_query = intent.product
        if intent.use_case:
            search_query += f" {intent.use_case}"
    else:
        search_query = state["query"]
    
    print("search_query: ", search_query)
    # Search using config values
    results = query_vector_store(
        search_query,
        vectorstore=vectorstore,
        k=settings.DEFAULT_SEARCH_K,
        format_results=True,
        max_score=settings.MAX_SIMILARITY_SCORE
    )
    
    state["search_results"] = results
    print(f"   Found {len(results)} products")
    return state

