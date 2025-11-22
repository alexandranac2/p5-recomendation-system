import re
from simple_rag.agent.simple_state import SimpleAgentState


def simple_refine_node(state: SimpleAgentState) -> SimpleAgentState:
    """Filter by price range extracted with regex (no LLM needed)"""
    print("✨ Refining results...")
    
    query = state["query"].lower()
    results = state["search_results"]
    filtered = []
    
    # Extract price constraints with regex
    max_match = re.search(r'(?:under|below|less than|max|maximum|up to)\s*\$?(\d+)', query)
    max_price = float(max_match.group(1)) if max_match else float('inf')
    
    min_match = re.search(r'(?:over|above|more than|min|minimum|from)\s*\$?(\d+)', query)
    min_price = float(min_match.group(1)) if min_match else 0
    
    # Filter by price
    for result in results:
        price = result.get("price", 0)
        if min_price <= price <= max_price:
            filtered.append(result)
    
    state["recommendations"] = filtered[:8]
    print(f"   {len(state['recommendations'])} recommendations")
    return state

