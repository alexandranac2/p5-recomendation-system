from simple_rag.agent.simple_state import SimpleAgentState
from rag.query import query_vector_store  # Reuse existing query function


def simple_search_node(state: SimpleAgentState, vectorstore) -> SimpleAgentState:
    """Search products using raw query - no intent analysis needed"""
    print("🔍 Searching products...")
    
    # Vector search is semantic, so raw query works great!
    results = query_vector_store(
        state["query"],
        vectorstore=vectorstore,
        k=15,
        format_results=True,
        max_score=1.3
    )
    
    state["search_results"] = results
    print(f"   Found {len(results)} products")
    return state

