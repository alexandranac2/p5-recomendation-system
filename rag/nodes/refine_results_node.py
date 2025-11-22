from rag.agent.state import AgentState


def refine_results_node(state: AgentState) -> AgentState:
    """
    Node 3: Refine results based on explicit intent constraints.
    
    Note: Vector search already filtered by semantic similarity (max_score=1.3),
    so we only apply hard constraints like price ranges here.
    Category matching is lenient since vector search already found relevant products.
    """
    print("✨ Refining results...")
    
    intent = state["analyzed_intent"]
    results = state["search_results"]
    filtered = []
    
    for result in results:
        # Price filter - hard constraint, must apply
        if intent and intent.price_range:
            price = result.get("price", 0)
            min_price = intent.price_range.get("min", 0)
            max_price = intent.price_range.get("max", float("inf"))
            if not (min_price <= price <= max_price):
                continue
        
        # Category filter - very lenient since vector search already did semantic matching
        # Only filter out if category is completely unrelated
        if intent and intent.category:
            product_category = result.get("category", "").lower()
            product_type = result.get("type", "").lower()
            intent_category = intent.category.lower()
            
            # Very lenient: check if there's ANY semantic connection
            # Since vector search found it, it's probably relevant
            category_match = (
                intent_category in product_category or 
                product_category in intent_category or
                intent_category in product_type or
                product_type in intent_category or
                any(word in product_type for word in intent_category.split() if len(word) > 3)
            )
            
            # Only skip if completely unrelated (very rare case)
            if not category_match:
                # But still allow if price matches and vector search found it
                # (vector search is usually more reliable than category strings)
                pass  # Keep it anyway - trust the vector search
        
        filtered.append(result)
    
    # Take top 5-8 results
    state["recommendations"] = filtered[:8]
    print(f"   {len(state['recommendations'])} recommendations")
    return state

