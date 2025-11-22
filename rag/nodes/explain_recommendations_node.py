from langchain_openai import ChatOpenAI
from rag.agent.state import AgentState


def explain_recommendations_node(state: AgentState) -> AgentState:
    """Node 4: Generate explanation for recommendations"""
    print("💬 Generating explanation...")
    
    recommendations = state["recommendations"]
    query = state["query"]
    
    if not recommendations:
        state["explanation"] = "No products found matching your criteria."
        return state
    
    # Simple explanation
    products_text = "\n".join([
        f"- {r.get('name')} (${r.get('price', 0)})"
        for r in recommendations[:3]
    ])
    
    prompt = f"""User asked: "{query}"

Top products:
{products_text}

Explain in 2 sentences why these match the user's needs."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(prompt)
    
    state["explanation"] = response.content.strip()
    return state

