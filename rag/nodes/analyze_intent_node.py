from rag.agent.state import AgentState
from rag.analazye_promt import analyse_promt


def analyze_intent_node(state: AgentState) -> AgentState:
    """Node 1: Analyze user query to understand intent"""
    print("🤖 Analyzing intent...")
    
    analyzed = analyse_promt(state["query"])
    state["analyzed_intent"] = analyzed
    
    print(f"   Product: {analyzed.product}, Intent: {analyzed.intent}")
    return state

