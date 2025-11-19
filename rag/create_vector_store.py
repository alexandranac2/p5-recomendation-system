from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

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

    project_root = (Path(__file__).parent.parent)
    faiss_path = project_root / "vectorstore"
    faiss_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Document] = []

    for document in documents:
        metadata = document.metadata or {}

        base_metadata = metadata.copy()
        if metadata.get("id") and "product_id" not in base_metadata:
            base_metadata["product_id"] = metadata["id"]

        product_text = "\n".join(_stringify_metadata(metadata)) or document.page_content

        print("product_text: ", product_text)
        print("base_metadata: ", base_metadata)

        doc = Document(page_content=product_text, metadata=base_metadata)
        chunks = text_splitter.split_documents([doc])

        for index, chunk in enumerate(chunks):
            chunk.metadata.update(base_metadata)
            chunk.metadata["chunk_index"] = index
            chunk.metadata["total_chunks"] = len(chunks)

        all_chunks.extend(chunks)
        identifier = base_metadata.get("product_id") or base_metadata.get("name") or "unknown"
        print(f"✅ Item {identifier}: {len(chunks)} chunks created")

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


def _stringify_metadata(data: Any, prefix: str = "") -> Iterable[str]:
    """Yield human-readable lines for nested metadata objects."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}{key}"
            yield from _stringify_metadata(value, f"{new_prefix}.")
    elif isinstance(data, list):
        joined = ", ".join(str(item) for item in data)
        field_name = prefix.rstrip(".") or "values"
        yield f"{field_name}: {joined}"
    else:
        field_name = prefix.rstrip(".") or "value"
        yield f"{field_name}: {data}"