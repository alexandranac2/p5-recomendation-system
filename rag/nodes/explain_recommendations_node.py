from langchain_openai import ChatOpenAI
from rag.agent.state import AgentState
from config.settings import settings


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
        for r in recommendations[:settings.MAX_RECOMMENDATIONS_TO_EXPLAIN]
    ])
    
    prompt = f"""User asked: "{query}"

Top products:
{products_text}

Explain in 2 sentences why these match the user's needs."""
    
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE
    )
    response = llm.invoke(prompt)
    
    state["explanation"] = response.content.strip()
    return state

