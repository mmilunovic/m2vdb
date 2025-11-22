"""
Example usage of m2vdb Python SDK.
"""

from m2vdb import M2VDBClient

def main():
    # Initialize client
    client = M2VDBClient(
        api_key="sk-test-user1",
        host="http://localhost:8000"
    )
    
    # Check server health
    health = client.health()
    print(f"Server status: {health['status']}")
    
    # Create an index
    print("\n1. Creating index...")
    index = client.create_index(
        name="products",
        dimension=3,
        metric="cosine"
    )
    print("Created index: products")
    
    # Upsert vectors
    print("\n2. Upserting vectors...")
    vectors = [
        {
            "id": "prod1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": {"name": "Red Phone", "price": 999}
        },
        {
            "id": "prod2", 
            "vector": [0.0, 1.0, 0.0],
            "metadata": {"name": "Green Laptop", "price": 1299}
        },
        {
            "id": "prod3",
            "vector": [0.0, 0.0, 1.0],
            "metadata": {"name": "Blue Tablet", "price": 799}
        },
        {
            "id": "prod4",
            "vector": [0.707, 0.707, 0.0],
            "metadata": {"name": "Yellow Camera", "price": 599}
        }
    ]
    
    count = index.upsert(vectors)
    print(f"Upserted {count} vectors")
    
    # Get index stats
    print("\n3. Index stats...")
    stats = index.describe()
    print(f"Index '{stats['name']}': {stats['size']} vectors, {stats['dimension']}d, {stats['metric']}")
    
    # Query for similar vectors
    print("\n4. Querying...")
    query_vector = [1.0, 0.0, 0.0]  # Similar to prod1 (Red Phone)
    results = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )
    
    print(f"Top 2 results for query {query_vector}:")
    for result in results:
        print(f"  - {result.id}: distance={result.distance:.3f}")
        print(f"    {result.metadata}")
    
    # Fetch specific vector
    print("\n5. Fetching vector...")
    vector = index.fetch("prod2")
    print(f"Fetched {vector['id']}: {vector['metadata']['name']}")
    
    # Delete some vectors
    print("\n6. Deleting vectors...")
    deleted1 = index.delete("prod3")
    deleted2 = index.delete("prod4")
    print(f"Deleted prod3: {deleted1}, prod4: {deleted2}")
    
    # Verify deletion
    stats = index.describe()
    print(f"Index now has {stats['size']} vectors")
    
    # List all indexes
    print("\n7. Listing all indexes...")
    all_indexes = client.list_indexes()
    for idx in all_indexes:
        print(f"  - {idx['name']}: {idx['size']} vectors")
    
    # Get handle to existing index (alternative to create)
    print("\n8. Getting existing index handle...")
    same_index = client.Index("products")
    stats = same_index.describe()
    print(f"Retrieved index: {stats['name']}")
    
    # Delete index
    print("\n9. Deleting index...")
    client.delete_index("products")
    print("Index deleted")
    
    # Verify
    all_indexes = client.list_indexes()
    print(f"Remaining indexes: {len(all_indexes)}")


if __name__ == "__main__":
    main()