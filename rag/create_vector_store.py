from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_load_vector_store(
    name: str = "vectorstore",
    products_path: Optional[Path] = None
) -> FAISS:
    """
    Create or load a FAISS vector store from product documents.
    If the vectorstore already exists, loads and returns it.
    If it doesn't exist, loads products from products_path and creates it.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    project_root = (Path(__file__).parent.parent)
    faiss_path = project_root / name
    faiss_path.mkdir(parents=True, exist_ok=True)

    # Check if vectorstore already exists
    index_faiss = faiss_path / "index.faiss"
    index_pkl = faiss_path / "index.pkl"
    
    if index_faiss.exists() and index_pkl.exists():
        print("📂 Loading existing vectorstore from disk...")
        vectorstore = FAISS.load_local(
            str(faiss_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded existing vectorstore from: {faiss_path}")
        return vectorstore

    # If vectorstore doesn't exist, load products and create documents
    if products_path is None or not products_path.exists():
        raise ValueError(
            f"Vectorstore '{name}' does not exist and products_path not provided or invalid. "
            f"Please provide a valid products_path to create a new vectorstore."
        )

    print(f"📂 Vectorstore doesn't exist. Loading products from {products_path}...")
    from load_products import load_products
    from ingest import create_documents
    
    products = load_products(products_path)
    documents = create_documents(products)

    # Create the vectorstore from documents
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(str(faiss_path))

    print(f"✅ Created FAISS index with {len(documents)} chunks")
    print(f"📁 Saved to: {faiss_path}")

    return vectorstore