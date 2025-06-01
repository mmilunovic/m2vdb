import json
import requests
import asyncio
from typing import Dict, Any

async def add_text(text: str, metadata: Dict[str, Any]):
    """Add text to the vector database using the API."""
    response = requests.post(
        'http://localhost:8000/add_text',
        json={
            'text': text,
            'metadata': metadata
        }
    )
    response.raise_for_status()
    return response.json()

async def main():
    # Read product descriptions
    with open('product-descriptions.json', 'r') as f:
        products = json.load(f)

    # Process each product
    for product in products:
        description = product.get('description', '')
        if not description:
            continue
            
        # Create metadata
        metadata = {
            'id': product.get('id'),
            'name': product.get('name'),
            'price': product.get('price')
        }
        
        # Add to database via API
        result = await add_text(description, metadata)
        print(f"Added product: {metadata['name']}")

    print(f"Successfully processed {len(products)} products")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main()) 