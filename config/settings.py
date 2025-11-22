from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration for magic numbers and settings"""
    
    # Vector Store Settings
    VECTOR_STORE_NAME: str = "alexs_vectorstore"
    EMBEDDING_MODEL: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    
    # Search Settings
    DEFAULT_SEARCH_K: int = 15  # Number of results to retrieve
    MAX_SIMILARITY_SCORE: float = 1.3  # Maximum similarity score threshold
    DEFAULT_QUERY_K: int = 5  # Default k for query_vector_store
    
    # LLM Settings
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.7
    
    # Recommendation Settings
    MAX_RECOMMENDATIONS_TO_EXPLAIN: int = 3  # Top N products to explain
    MAX_RECOMMENDATIONS_TO_RETURN: int = 8  # Maximum recommendations to return
    
    # API Settings
    API_TITLE: str = "Product Recommendation System API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for product recommendations using RAG"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables (like OPENAI_API_KEY)


# Global settings instance
settings = Settings()

