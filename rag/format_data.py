def format_search_results(results, is_reranked: bool = False):
    """
    Format search results for display.
    
    Args:
        results: List of tuples (document, score)
        is_reranked: If True, score is a rerank score (higher is better).
                    If False, score is a similarity score (lower is better).
    """
    formatted = []

    for doc, score in results:
        formatted.append(
            {
                "product_id": doc.metadata.get("product_id"),
                "name": doc.metadata.get("name"),
                "category": doc.metadata.get("category"),
                "price": doc.metadata.get("price"),
                "rating": doc.metadata.get("rating"),
               "stock": doc.metadata.get("stock"),
                "content": doc.page_content,
                "score": float(score),
                "score_type": "rerank_score" if is_reranked else "similarity_score",
                "chunk_index": doc.metadata.get("chunk_index"),
            }
        )

    return formatted
