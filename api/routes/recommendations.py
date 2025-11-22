from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])

# Global variables to be initialized by lifespan
_recommendation_fn = None


def initialize_recommendation_system():
    """Initialize vectorstore and recommendation graph - called by lifespan"""
    global _recommendation_fn
    
    from rag.agent.recommendation_agent import build_recommendation_graph
    from rag.create_vector_store import create_load_vector_store
    from config.settings import settings
    
    print("🚀 Initializing recommendation system...")
    products_path = Path(__file__).parent.parent.parent / "data" / "products.json"
    
    vectorstore = create_load_vector_store(
        name=settings.VECTOR_STORE_NAME,
        products_path=products_path
    )
    
    # Build the graph - this already includes explain_recommendations_node!
    _recommendation_fn = build_recommendation_graph(vectorstore)
    print("✅ Recommendation system initialized")


def get_recommendation_function():
    """Get the recommendation function (must be initialized via lifespan)"""
    if _recommendation_fn is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation system not initialized. Please wait for startup to complete."
        )
    return _recommendation_fn


# Request/Response models
class RecommendationRequest(BaseModel):
    query: str
    max_results: Optional[int] = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Dict[str, Any]]
    explanation: str
    total_results: int
    intent: Optional[Dict[str, Any]] = None


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get product recommendations based on a natural language query.
    
    The graph automatically handles:
    - Intent analysis
    - Product search
    - Result refinement
    - Explanation generation (via explain_recommendations_node)
    - Response formatting
    
    - **query**: User's search query (e.g., "I need a laptop for gaming")
    - **max_results**: Optional limit on number of recommendations
    """
    try:
        recommend = get_recommendation_function()
        
        # Just call it like in test_agent.py - the explain node is already in the graph!
        result = recommend(request.query)
        
        recommendations = result.get("recommendations", [])
        if request.max_results:
            recommendations = recommendations[:request.max_results]
        
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations,
            explanation=result.get("explanation", ""),
            total_results=len(recommendations),
            intent=result.get("intent")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")


@router.get("/search")
async def search_products(
    q: str,
    k: Optional[int] = None,
    max_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Direct product search using vector similarity.
    
    - **q**: Search query
    - **k**: Number of results (defaults to config value)
    - **max_score**: Maximum similarity score threshold
    """
    try:
        from rag.query import query_vector_store
        from rag.create_vector_store import create_load_vector_store
        from config.settings import settings
        
        products_path = Path(__file__).parent.parent.parent / "data" / "products.json"
        vectorstore = create_load_vector_store(
            name=settings.VECTOR_STORE_NAME,
            products_path=products_path
        )
        
        k = k or settings.DEFAULT_SEARCH_K
        max_score = max_score or settings.MAX_SIMILARITY_SCORE
        
        results = query_vector_store(
            q,
            vectorstore=vectorstore,
            k=k,
            format_results=True,
            max_score=max_score
        )
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching products: {str(e)}")

