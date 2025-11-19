"""
Ingest products.json into FAISS vector store

Run once to setup: python ingest.py
"""

from langchain_core.documents import Document


def create_product_metadata(product):
    """
    Create metadata dictionary from a product dynamically.
    Copies all product fields and flattens attributes inline.
    
    Args:
        product: Dictionary containing product data
        
    Returns:
        Dictionary with product metadata (attributes flattened inline)
    """
    # Copy all product fields
    metadata = product.copy()
    
    # Extract and remove attributes to flatten them
    attributes = metadata.pop("attributes", {})
    
    # Flatten attributes inline with the rest of the metadata
    return {**metadata, **attributes}


def create_documents(products):
    """Convert products to Document objects"""
    documents = []

    for product in products:
        # Create searchable content
        content = f"{product['description']} | Category: {product['category']}"

        if product["attributes"].get("brand"):
            content += f" | Brand: {product['attributes']['brand']}"
        if product["attributes"].get("use_case"):
            content += f" | Use case: {product['attributes']['use_case']}"

        doc = Document(
            page_content=content,
            metadata=create_product_metadata(product),
        )
        documents.append(doc)

    print(f"✅ Created {len(documents)} documents")
    return documents
