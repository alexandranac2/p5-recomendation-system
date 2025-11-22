from typing import TypedDict, List, Dict, Any, Optional
from rag.analazye_promt import understand_promt


class AgentState(TypedDict):
    """Simple state passed between nodes"""
    query: str
    analyzed_intent: Optional[understand_promt]
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    explanation: str
    formatted_response: Optional[Dict[str, Any]]

