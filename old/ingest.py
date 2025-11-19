"""
Ingest products.json into FAISS vector store

Run once to setup: python ingest.py
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from load_products import load_products
from create_vector_store import create_vector_store




def create_documents(products):
    """Convert products to Document objects"""
    documents = []
    
    for product in products:
        # Create searchable content
        content = f"{product['description']} | Category: {product['category']}"
        
        if product['attributes'].get('brand'):
            content += f" | Brand: {product['attributes']['brand']}"
        if product['attributes'].get('use_case'):
            content += f" | Use case: {product['attributes']['use_case']}"
        
        doc = Document(
            page_content=content,
            metadata={
                "id": product['id'],
                "name": product['name'],
                "category": product['category'],
                "price": product['price'],
                "rating": product['rating'],
                "stock": product['stock'],
                "attributes": product['attributes'],
                "description": product['description']
            }
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents")
    return documents





def test_search(vectorstore):
    """Test the vector store"""
    test_queries = [
        "gift for dad who loves coffee",
        "running shoes under $200"
    ]
    
    print("\n" + "="*60)
    print("TESTING VECTOR SEARCH")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata['name']} - ${doc.metadata['price']}")


def main():
    print("\n" + "="*60)
    print("PRODUCT INGESTION")
    print("="*60 + "\n")
    
    # Check API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("   Create .env file with: OPENAI_API_KEY=sk-your-key")
        return
    
    # Load and process
    products = load_products()
    documents = create_documents(products)
    vectorstore = create_vector_store(documents)
    test_search(vectorstore)
    
    print("\n✅ Setup complete! Start API with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()