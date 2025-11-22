def query_vector_store(query, vectorstore, k=5, format_results=True, max_score=None):
    """
    Query the vectorstore and optionally filter by similarity score.
    
    Args:
        query: Search query string
        vectorstore: FAISS vectorstore object
        k: Number of results to return
        format_results: Whether to format results
        max_score: Maximum similarity score threshold (lower is better, so this filters out bad matches)
                   If None, no filtering is applied
    """
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Filter by score if threshold is provided
    if max_score is not None:
        filtered_results = [(doc, score) for doc, score in results if score <= max_score]
        if not filtered_results:
            print(f"⚠️  No results found below score threshold of {max_score}")
        else:
            print(f"✅ Filtered {len(results)} results to {len(filtered_results)} below score {max_score}")
        results = filtered_results
    
    if format_results:
        from rag.format_data import format_search_results
        return format_search_results(results)
    return results
