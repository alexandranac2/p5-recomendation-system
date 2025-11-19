def query_vector_store(query, vectorstore):
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=5)
    return results
