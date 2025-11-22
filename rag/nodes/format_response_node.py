from typing import Dict, Any
from rag.agent.state import AgentState


def format_response_node(state: AgentState) -> AgentState:
    """Node 5: Format the final response"""
    print("📋 Formatting response...")
    
    formatted_response: Dict[str, Any] = {
        "query": state["query"],
        "intent": state["analyzed_intent"].dict() if state["analyzed_intent"] else None,
        "recommendations": state["recommendations"],
        "explanation": state["explanation"]
    }
    
    state["formatted_response"] = formatted_response
    return state

