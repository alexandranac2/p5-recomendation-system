import json
from pathlib import Path

# Method 1: Simple Python JSON loading
def load_products_json():
    """Load products.json using Python's built-in json module"""
    products_path = Path(__file__).parent / "data" / "products.json"
    
    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    return products

# Example usage
if __name__ == "__main__":
    products = load_products_json()
    print(f"✅ Loaded {len(products)} products")
    print(f"📦 First product: {products[0]['name']}")
    
    # Access product data
    for product in products:
        print(f"ID: {product['id']}, Name: {product['name']}, Price: ${product['price']}")

