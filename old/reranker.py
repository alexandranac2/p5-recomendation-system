"""
Reranking module for improving search result relevance.
Uses cross-encoder models from sentence-transformers for reranking.
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:
    """Reranker using cross-encoder models for better relevance scoring."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                       Options:
                       - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good quality)
                       - "cross-encoder/ms-marco-MiniLM-L-12-v2" (slower, better quality)
                       - "cross-encoder/ms-marco-electra-base" (best quality, slower)
        """
        print(f"🔄 Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✅ Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of documents to rerank (from vector store)
            top_k: Number of top results to return after reranking
            
        Returns:
            List of tuples (document, rerank_score) sorted by relevance (higher is better)
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder: (query, document_content)
        pairs = [
            [query, doc.page_content] for doc in documents
        ]
        
        # Get rerank scores (higher is better)
        scores = self.model.predict(pairs)
        
        # Combine documents with their rerank scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score descending (higher is better)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return doc_scores[:top_k]


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """Factory function to create a reranker instance."""
    return Reranker(model_name=model_name)

