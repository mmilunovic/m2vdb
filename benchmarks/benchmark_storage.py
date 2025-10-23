import numpy as np
import os
import json
import os
import time

import humanize
import numpy as np
import psutil

from m2vdb.database import VectorDatabase

def get_process_memory():
    """Get current process memory usage in bytes"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def generate_test_data(num_vectors=10000, dim=128, metadata_size_kb=1024):
    """Generate test vectors and metadata of specified sizes
    
    Args:
        num_vectors: Number of vectors to generate (default: 10000)
        dim: Vector dimensionality (default: 128)
        metadata_size_kb: Target size in kilobytes for each metadata item (default: 1024KB = 1MB)
                         The actual size will be approximate due to JSON overhead
    """
    # Convert KB to bytes for internal calculations
    metadata_size = metadata_size_kb * 1024
    
    # Generate random vectors
    vectors = np.random.random((num_vectors, dim)).astype('float32')
    
    # Calculate approximate sizes for variable fields to reach target size
    # Fixed fields (id, name, timestamp, path) + JSON structure ≈ 100 bytes
    # Nested dict with 8 pairs ≈ 150 bytes
    # Tags array structure ≈ 50 bytes
    
    # Distribute remaining size:
    # - 95% for description (main content)
    # - 5% for tags and nested metadata
    description_size = int(metadata_size * 9.5 // 10)
    
    import random
    from datetime import datetime, timedelta
    base_time = datetime.now()
    
    # Generate metadata with specified approximate size per item
    metadata = []
    for i in range(num_vectors):
        # Generate random nested metadata
        nested_count = random.randint(5, 10)
        nested_metadata = {
            f"key_{j}": f"value_{random.randint(1000, 9999)}"
            for j in range(nested_count)
        }
        
        item_metadata = {
            "id": i,
            "name": f"vector_{i}",
            "description": "x" * description_size,
            "timestamp": (base_time - timedelta(days=random.uniform(0, 30))).isoformat(),
            "path": f"/path/to/document_{i}.pdf",
            "tags": [f"tag_{j}_{random.randint(100, 999)}" for j in range(20)],
            "nested_metadata": nested_metadata
        }
        metadata.append(item_metadata)
    
    return vectors, metadata

def run_storage_benchmark(num_vectors=10000, dim=128, metadata_size_kb=1024, storage_path="benchmark_storage/test"):
    """Run storage benchmark with specified parameters
    
    Args:
        num_vectors: Number of vectors to generate (default: 10000)
        dim: Vector dimensionality (default: 128)
        metadata_size_kb: Target size in kilobytes for each metadata item (default: 1024KB = 1MB)
        storage_path: Where to store the test database (default: "benchmark_storage/test")
    """
    print(f"\nRunning benchmark with:")
    print(f"- Number of vectors: {num_vectors:,}")
    print(f"- Dimensions: {dim}")
    print(f"- Target metadata size per item: {metadata_size_kb} KB ({metadata_size_kb/1024:.1f} MB)")
    
    # Clean up previous test data
    if os.path.exists(storage_path):
        os.system(f"rm -rf {storage_path}")
    
    # Generate test data
    print("\nGenerating test data...")
    vectors, metadata = generate_test_data(num_vectors, dim, metadata_size_kb)
    
    # Calculate actual metadata size
    metadata_sample = json.dumps(metadata[0]).encode('utf-8')
    actual_metadata_size = len(metadata_sample)
    actual_metadata_kb = actual_metadata_size / 1024
    print(f"Actual metadata size per item: {actual_metadata_kb:.1f} KB ({actual_metadata_kb/1024:.1f} MB)")
    
    # Create and populate database
    print("\nCreating database...")
    db = VectorDatabase(dim=dim, storage_path=storage_path)    
    # Add vectors and metadata
    start_time = time.time()
    db.add(vectors, metadata_list=metadata)
    add_time = time.time() - start_time
        
    print(f"Add time: {add_time:.2f} seconds")
    
    # Save database
    print("\nSaving database...")
    start_time = time.time()
    db.save()
    save_time = time.time() - start_time
    
    # Get size of saved files
    saved_size = sum(os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(storage_path)
                    for file in files)
    
    # Calculate size ratios
    vectors_size = vectors.nbytes
    total_metadata_size = actual_metadata_size * num_vectors
    print(f"Save time: {save_time:.2f} seconds")
    print(f"Saved file size: {humanize.naturalsize(saved_size)}")
    print(f"Raw vectors size: {humanize.naturalsize(vectors_size)} ({vectors_size/1024:.1f} KB)")
    print(f"Raw metadata size: {humanize.naturalsize(total_metadata_size)} ({total_metadata_size/1024:.1f} KB)")
    print(f"Metadata/Vectors size ratio: {total_metadata_size/vectors_size:.1f}x")
    
    # Clear memory
    del db
    del vectors
    del metadata
    
    # Load database
    print("\nLoading database...")
    start_time = time.time()
    pre_load_memory = get_process_memory()
    
    db = VectorDatabase(storage_path=storage_path, load_existing=True)
    load_time = time.time() - start_time
    
    # Measure memory after loading
    post_load_memory = get_process_memory()
    load_memory_increase = post_load_memory - pre_load_memory
    
    print(f"Load time: {load_time:.2f} seconds")
    
    # Return benchmark results
    return {
        "num_vectors": num_vectors,
        "dim": dim,
        "target_metadata_size_kb": metadata_size_kb,
        "actual_metadata_size_kb": actual_metadata_kb,
        "add_time": add_time,
        "save_time": save_time,
        "saved_size": saved_size,
        "vectors_size": vectors_size,
        "total_metadata_size": total_metadata_size,
        "metadata_vectors_ratio": total_metadata_size/vectors_size,
        "load_time": load_time,
        "load_memory": load_memory_increase
    }

def main():
    # Test configurations focusing on metadata size variations
    configs = [
        # Small metadata (100KB per item)
        {"metadata_size_kb": 100},
        # Large metadata (1MB = 10240KB per item)
        {"metadata_size_kb": 1024}
    ]
    
    print("\nMetadata Storage Benchmark")
    print("=" * 80)
    
    for config in configs:
        storage_path = f"./benchmarks/data/metadata_benchmark/test_metadata_{config['metadata_size_kb']}KB"
        result = run_storage_benchmark(
            metadata_size_kb=config['metadata_size_kb'],
            storage_path=storage_path
        )
        print("\nSummary:")
        print(f"Target metadata size per item: {config['metadata_size_kb']} KB ({config['metadata_size_kb']/1024:.1f} MB)")
        print(f"Actual metadata size per item: {result['actual_metadata_size_kb']:.1f} KB ({result['actual_metadata_size_kb']/1024:.1f} MB)")
        print(f"Save time: {result['save_time']:.2f} seconds")
        print(f"Load time: {result['load_time']:.2f} seconds")
        print(f"Raw vectors size: {humanize.naturalsize(result['vectors_size'])} ({result['vectors_size']/1024:.1f} KB)")
        print(f"Raw metadata size: {humanize.naturalsize(result['total_metadata_size'])} ({result['total_metadata_size']/1024:.1f} KB)")
        print(f"Metadata/Vectors size ratio: {result['metadata_vectors_ratio']:.1f}x")
        print("=" * 80)

if __name__ == "__main__":
    main() 