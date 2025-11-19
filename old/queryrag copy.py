from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv
from load_vectorstore import load_vectorstore
import nltk
import ssl
import certifi
import os
import spacy

# Fix SSL certificate verification for NLTK downloads (only needed if downloading new data)
# Set SSL certificate path to use certifi's certificate bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Download stopwords only if not already downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    # SSL fix needed here if downloading
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords


load_dotenv()


store = load_vectorstore()


def search_products(query: str, k: int = 5, filter_dict: dict = None):
    """
    Search for products using semantic similarity

    Args:
        query: The search query text
        k: Number of results to return (default: 5)
        filter_dict: Optional metadata filter (e.g., {"category": "Electronics"})

    Returns:
        List of tuples (document, score)
    """
    vectorstore = store

    if filter_dict:
        # Search with metadata filtering
        results = vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
    else:
        # Search without filtering
        results = vectorstore.similarity_search_with_score(query, k=k)

    return results


def search_products_by_category(query: str, category: str, k: int = 5):
    """Convenience function to search within a specific category"""
    return search_products(query, k=k, filter_dict={"category": category})


def format_search_results(results):
    """Format search results for display"""
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
                "similarity_score": float(score),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
        )

    return formatted


def quick_search(query: str, top_k: int = 5):
    """Quick search function that returns formatted results"""
    results = search_products(query, k=top_k)
    return format_search_results(results)


def search_with_filters(query: str, k: int = 5, **filters):
    """
    Search with multiple metadata filters

    Example:
        search_with_filters("laptop", k=5, category="Electronics", rating=4.5)
    """
    vectorstore = load_vectorstore()

    # Build filter dict from kwargs
    filter_dict = {k: v for k, v in filters.items() if v is not None}

    if filter_dict:
        results = vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)

    return results


def get_retriever(k: int = 5, filter_dict: dict = None):
    """Get a retriever for use with LangChain chains"""
    vectorstore = load_vectorstore()

    if filter_dict:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": filter_dict}
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever


def hybrid_search(query: str, k: int = 5, keywords: list = None):
    # Semantic search
    semantic_results = search_products(query, k=k * 2)

    # Keyword filtering (coffee-related terms)
    query_lower = query.lower()

    # Boost results that contain keywords
    filtered = []
    for doc, score in semantic_results:
        content_lower = doc.page_content.lower()
        if any(keyword in content_lower for keyword in keywords):
            # Boost coffee-related results
            filtered.append((doc, score * 0.8))  # Lower score = better
        elif any(keyword in query_lower for keyword in keywords):
            # If query mentions coffee, filter out non-coffee products
            continue
        else:
            filtered.append((doc, score))

    return sorted(filtered, key=lambda x: x[1])[:k]  # Sort by score ascending


queryString = "Birthday gift for dad who loves coffee"

# stopwords
stop_words = set(stopwords.words("english"))
tokens = queryString.lower().split()
filtered = [w for w in tokens if w not in stop_words]
nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")

# lemmatize
lemmas = [nlp_sm(w)[0].lemma_ for w in filtered]

# get spaCy semantic expansion
expanded_terms = set()
for lemma in lemmas:
    token = nlp_lg(lemma)
    for vocab_word in nlp_lg.vocab:
        if vocab_word.has_vector and vocab_word.is_lower and vocab_word.is_alpha:
            sim = token.similarity(nlp_lg(vocab_word.text))
            if sim > 0.60:  # threshold
                expanded_terms.add(vocab_word.text)

print("Filtered:", filtered)
print("Lemmas:", lemmas)
print("Similar semantic words:", list(expanded_terms)[:20])

# nlp = spacy.load("en_core_web_sm")
# stop_words = set(stopwords.words("english"))
# tokens = queryString.lower().split()
# filtered = [word for word in tokens if word not in stop_words]
# print("Filtered tokens: ", filtered)


# print(f"🔍 Searching for '{queryString}'...")
# results = hybrid_search(queryString, k=3, keywords=filtered)
# formatted = format_search_results(results)
# print("results: ", results)
# print("formatted: ", formatted)


# print(f"🔍 Searching for '{queryString}'...")
# results = search_products(queryString, k=3)
# formatted = format_search_results(results)

# print(formatted)
