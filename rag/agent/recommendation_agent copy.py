from typing import Dict, Any
from functools import partial
from langgraph.graph import StateGraph, START, END

from rag.agent.state import AgentState
from rag.nodes import (
    analyze_intent_node,
    search_products_node,
    refine_results_node,
    explain_recommendations_node,
    format_response_node,
)


class RecommendationAgent:
    """Simple recommendation agent"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_intent_node)
        workflow.add_node("search", partial(search_products_node, vectorstore=self.vectorstore))
        workflow.add_node("refine", refine_results_node)
        workflow.add_node("explain", explain_recommendations_node)
        workflow.add_node("format", format_response_node)
        
        # Connect nodes
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "search")
        workflow.add_edge("search", "refine")
        workflow.add_edge("refine", "explain")
        workflow.add_edge("explain", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()
    
    def recommend(self, query: str) -> Dict[str, Any]:
        """Get recommendations for a query"""
        state: AgentState = {
            "query": query,
            "analyzed_intent": None,
            "search_results": [],
            "recommendations": [],
            "explanation": "",
            "formatted_response": None
        }
        
        final_state = self.graph.invoke(state)
        
        return final_state["formatted_response"] or {}

