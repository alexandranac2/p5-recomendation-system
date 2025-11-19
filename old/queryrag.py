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
import json
import re
from langchain_openai import ChatOpenAI

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


def parse_keywords_to_list(keywords):
    """
    Convert keywords from various formats (string, list, None) to a list of strings.
    
    Args:
        keywords: Can be None, a string (JSON array, comma-separated, or plain text), or a list
        
    Returns:
        List of lowercase keyword strings
    """
    if keywords is None:
        return []
    
    if isinstance(keywords, list):
        # Already a list, just normalize
        return [str(kw).strip().lower() for kw in keywords if str(kw).strip()]
    
    if isinstance(keywords, str):
        # Try to parse as JSON array first
        try:
            parsed = json.loads(keywords)
            if isinstance(parsed, list):
                return [str(kw).strip().lower() for kw in parsed if str(kw).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If not JSON, try to extract from string format like "[word1, word2]" or "word1, word2"
        # Remove brackets, quotes, and split by commas
        cleaned = re.sub(r'[\[\]"\']', "", keywords)
        keywords_list = [kw.strip().lower() for kw in cleaned.split(",") if kw.strip()]
        
        # If still empty, try splitting by spaces
        if not keywords_list:
            keywords_list = [kw.strip().lower() for kw in keywords.split() if kw.strip()]
        
        return keywords_list
    
    # Fallback: convert to string and split
    return [str(keywords).strip().lower()]


def hybrid_search(query: str, k: int = 5, keywords=None):
    # Semantic search
    semantic_results = search_products(query, k=k * 2)

    # Convert keywords to list (handles string, list, or None)
    keywords = parse_keywords_to_list(keywords)

    # Keyword filtering
    query_lower = query.lower()
    query_has_keywords = (
        any(keyword in query_lower for keyword in keywords) if keywords else False
    )

    # Boost results that contain keywords
    filtered = []
    for doc, score in semantic_results:
        content_lower = doc.page_content.lower()
        product_has_keywords = (
            any(keyword in content_lower for keyword in keywords) if keywords else False
        )

        if product_has_keywords:
            # Boost products that contain keywords (lower score = better)
            filtered.append((doc, score * 0.8))
        elif query_has_keywords:
            # If query mentions keywords but product doesn't, filter it out
            continue
        else:
            # No keyword filtering needed, keep original score
            filtered.append((doc, score))

    # Sort by score ascending (lower = better) and return top k
    return sorted(filtered, key=lambda x: x[1])[:k]


queryString = "Birthday gift for dad who loves coffee"


nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
tokens = queryString.lower().split()
filtered = [word for word in tokens if word not in stop_words]
print("Filtered tokens: ", filtered)

llm = ChatOpenAI(model="gpt-4o-mini")
# response = llm.invoke(
#     "can you please give me synonym of the following words: " + ", ".join(filtered)
# )
response = llm.invoke(
    "can you get the intension of the following query: "
    + queryString
    + "Give me sinonyms for the product? As output plase give me just the array of synonyms. No other text or explanation. no ```json or ``` or anything else. Just the array of synonyms."
)
print(f"📝 LLM Response (type: {type(response.content).__name__}): {response.content}")

# Convert string to array
keywords_array = parse_keywords_to_list(response.content)
print(f"✅ Converted to array (type: {type(keywords_array).__name__}): {keywords_array}")

print(f"\n🔍 Searching for '{queryString}'...")
results = hybrid_search(queryString, k=3, keywords=keywords_array)
formatted = format_search_results(results)
print("📊 Search Results:")
print(formatted)


# print(f"🔍 Searching for '{queryString}'...")
# results = search_products(queryString, k=3)
# formatted = format_search_results(results)

# print(formatted)
