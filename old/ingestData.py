   
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
# Load products
products_path = Path(__file__).parent.parent / "data" / "products.json"
with open(products_path, "r", encoding="utf-8") as f:
    products = json.load(f)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

# Initialize text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust based on your needs
    chunk_overlap=50,  # Overlap between chunks
    length_function=len,
)

# Initialize FAISS vector store (create empty first)
faiss_path = Path(__file__).parent.parent / "vectorstore"

# Process each product individually
all_chunks = []

for product in products:
    # Create a text representation of the product
    product_text = (
        f"Name: {product['name']}\n"
        f"Description: {product['description']}\n"
        f"Category: {product['category']}\n"
        f"Price: ${product['price']}\n"
        f"Rating: {product['rating']}\n"
        f"Stock: {product['stock']}\n"
        f"Attributes: {', '.join([f'{k}={v}' for k, v in product.get('attributes', {}).items()])}"
    )
    
    # Create metadata for this product
    metadata = {
        "product_id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "rating": product["rating"],
        "stock": product["stock"],
    }
    
    # Create a Document object
    doc = Document(page_content=product_text, metadata=metadata)
    
    # Chunk the product document
    chunks = text_splitter.split_documents([doc])
    
    # Add product_id to each chunk's metadata (so you can trace back to original product)
    for chunk in chunks:
        chunk.metadata["product_id"] = product["id"]
        chunk.metadata["chunk_index"] = chunks.index(chunk)
        chunk.metadata["total_chunks"] = len(chunks)
    
    all_chunks.extend(chunks)
    print(f"✅ Product {product['id']} ({product['name']}): {len(chunks)} chunks created")

print(f"\n📦 Total chunks created: {len(all_chunks)}")

# Create FAISS vector store from all chunks
vectorstore = FAISS.from_documents(documents=all_chunks, embedding=embeddings)

# Save the vector store
vectorstore.save_local(str(faiss_path))

print(f"✅ Created FAISS index with {len(all_chunks)} chunks from {len(products)} products")
print(f"📁 Saved to: {faiss_path}")