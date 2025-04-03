# Complete end-to-end test for m2vdb Vector Database
import numpy as np
import os
import shutil
from m2vdb.database import V3cT0rDaTaBas3 as VectorDatabase

# Test parameters
dim = 64
test_dir = "test_db_storage"
num_vectors = 1000
num_queries = 5
k = 10

def run_tests():
    # Clean up any previous test data
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Generate random test data
    print("Generating test data...")
    vectors = np.random.random((num_vectors, dim)).astype('float32')
    queries = np.random.random((num_queries, dim)).astype('float32')
    metadata = [{"name": f"vector_{i}", "tag": f"tag_{i % 5}"} for i in range(num_vectors)]

    # Test 1: Create and use BruteForce index
    print("\n=== Testing BruteForce Index ===")
    bf_db = VectorDatabase(dim=dim, index_type="brute_force", storage_path=f"{test_dir}/brute_force")
    print(f"Created BruteForce index with dim={dim}")

    # Add vectors with metadata
    bf_db.add(vectors, metadata_list=metadata)
    print(f"Added {num_vectors} vectors with metadata")

    # Search
    bf_results = bf_db.search(queries, k=k)
    print(f"Searched for {num_queries} queries, got {len(bf_results)} result sets")
    print(f"First query returned {len(bf_results[0])} results")
    print(f"Sample result: {bf_results[0][0]}")

    # Save the database
    bf_db.save()
    print(f"Saved BruteForce index to {test_dir}/brute_force")

    # Test 2: Create and use ANN index
    print("\n=== Testing ANN Index ===")
    ann_db = VectorDatabase(dim=dim, index_type="ann", num_candidates=50, random_seed=42,
                         storage_path=f"{test_dir}/ann")
    print(f"Created ANN index with dim={dim}, num_candidates=50, random_seed=42")

    # Add vectors with metadata
    ann_db.add(vectors, metadata_list=metadata)
    print(f"Added {num_vectors} vectors with metadata")

    # Search
    ann_results = ann_db.search(queries, k=k)
    print(f"Searched for {num_queries} queries, got {len(ann_results)} result sets")
    print(f"First query returned {len(ann_results[0])} results")
    print(f"Sample result: {ann_results[0][0]}")

    # Save the database
    ann_db.save()
    print(f"Saved ANN index to {test_dir}/ann")

    # Test 3: Load saved databases and verify functionality
    print("\n=== Testing Loading from Storage ===")

    # Load BruteForce index
    loaded_bf_db = VectorDatabase(storage_path=f"{test_dir}/brute_force", load_existing=True)
    print(f"Loaded BruteForce index with dim={loaded_bf_db.dim}")

    # Search with loaded index
    loaded_bf_results = loaded_bf_db.search(queries, k=k)
    print(f"Searched with loaded BruteForce index, got {len(loaded_bf_results)} result sets")
    print(f"Verification: results identical to original search? {loaded_bf_results[0][0]['id'] == bf_results[0][0]['id']}")

    # Load ANN index
    loaded_ann_db = VectorDatabase(storage_path=f"{test_dir}/ann", load_existing=True)
    print(f"Loaded ANN index with dim={loaded_ann_db.dim}")

    # Search with loaded index
    loaded_ann_results = loaded_ann_db.search(queries, k=k)
    print(f"Searched with loaded ANN index, got {len(loaded_ann_results)} result sets")
    print(f"Verification: results identical to original search? {loaded_ann_results[0][0]['id'] == ann_results[0][0]['id']}")

    # Test 4: Update existing database
    print("\n=== Testing Database Updates ===")

    # Add more vectors to the loaded BruteForce database
    more_vectors = np.random.random((50, dim)).astype('float32')
    more_metadata = [{"name": f"new_vector_{i}", "group": "update_batch"} for i in range(50)]
    loaded_bf_db.add(more_vectors, metadata_list=more_metadata)
    print(f"Added 50 more vectors to BruteForce index")

    # Save updated database
    loaded_bf_db.save()
    print("Saved updated BruteForce index")

    # Load again and verify update
    updated_bf_db = VectorDatabase(storage_path=f"{test_dir}/brute_force", load_existing=True)
    updated_results = updated_bf_db.search(queries, k=k+10)  # Get more results to see new vectors
    print(f"Loaded updated BruteForce index, found {len(updated_results[0])} results for first query")

    # Check if we can find any of the newly added vectors with 'group' metadata
    update_found = False
    for result in updated_results[0]:
        if result['metadata'].get('group') == 'update_batch':
            update_found = True
            break

    print(f"Found vectors from update batch? {update_found}")
    print("\n=== End-to-End Test Complete ===")

if __name__ == "__main__":
    run_tests() 