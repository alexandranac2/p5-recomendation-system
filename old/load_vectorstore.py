from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_vectorstore():
    """Load the FAISS vectorstore from disk"""
    faiss_path = Path(__file__).parent.parent / "vectorstore"
    # Initialize embeddings - CORRECT WAY
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    
    if not faiss_path.exists():
        raise FileNotFoundError(f"Vectorstore not found at {faiss_path}. Run ingestData.py first.")
    
    vectorstore = FAISS.load_local(
        str(faiss_path), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    return vectorstore
