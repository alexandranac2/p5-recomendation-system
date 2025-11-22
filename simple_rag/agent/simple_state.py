from typing import TypedDict, List, Dict, Any, Optional


class SimpleAgentState(TypedDict):
    """Simplified state - no intent analysis needed"""
    query: str
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    explanation: str
    formatted_response: Optional[Dict[str, Any]]

