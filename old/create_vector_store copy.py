from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def create_vector_store(documents: Sequence[Document]) -> FAISS:
    """Create (and persist) a FAISS vector store from product documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    faiss_path = Path(__file__).parent.parent / "vectorstore"
    faiss_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Document] = []

    for document in documents:
        metadata = document.metadata or {}

        attributes = metadata.get("attributes") or {}
        attributes_text = ", ".join(f"{k}={v}" for k, v in attributes.items()) or "None"

        product_text = (
            f"Name: {metadata.get('name', 'Unknown')}\n"
            f"Description: {metadata.get('description', '')}\n"
            f"Category: {metadata.get('category', 'Unknown')}\n"
            f"Price: ${metadata.get('price', 'N/A')}\n"
            f"Rating: {metadata.get('rating', 'N/A')}\n"
            f"Stock: {metadata.get('stock', 'N/A')}\n"
            f"Attributes: {attributes_text}"
        )

        base_metadata = {
            "product_id": metadata.get("id"),
            "name": metadata.get("name"),
            "category": metadata.get("category"),
            "price": metadata.get("price"),
            "rating": metadata.get("rating"),
            "stock": metadata.get("stock"),
        }

        doc = Document(page_content=product_text, metadata=base_metadata)
        chunks = text_splitter.split_documents([doc])

        for index, chunk in enumerate(chunks):
            chunk.metadata["product_id"] = base_metadata.get("product_id")
            chunk.metadata["chunk_index"] = index
            chunk.metadata["total_chunks"] = len(chunks)

        all_chunks.extend(chunks)
        print(
            f"✅ Product {base_metadata.get('product_id')} ({base_metadata.get('name')}): "
            f"{len(chunks)} chunks created"
        )

    if not all_chunks:
        raise ValueError("No chunks were created; ensure documents are not empty.")

    print(f"\n📦 Total chunks created: {len(all_chunks)}")

    vectorstore = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
    vectorstore.save_local(str(faiss_path))

    print(
        f"✅ Created FAISS index with {len(all_chunks)} chunks "
        f"from {len(documents)} documents"
    )
    print(f"📁 Saved to: {faiss_path}")

    return vectorstore