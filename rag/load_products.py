import json
from pathlib import Path

def load_products(file_path="products.json"):
    """Load products from JSON"""
    
    with open(file_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"✅ Loaded {len(products)} products")
    return products
