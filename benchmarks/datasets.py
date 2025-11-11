"""
Dataset loaders for benchmarking vector databases.

Supports:
- SIFT1M: 128D image descriptors (1M base + 10K queries)
- FastText: 300D word embeddings
"""

import os
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from urllib.request import urlretrieve

import numpy as np
import requests
from rich.progress import Progress, DownloadColumn, BarColumn, TransferSpeedColumn, TimeRemainingColumn


@dataclass
class Dataset:
    """Vector dataset with queries and ground truth."""
    name: str
    base_vectors: np.ndarray  # (n, dim) - database vectors
    query_vectors: np.ndarray  # (nq, dim) - query vectors
    ground_truth: np.ndarray   # (nq, k) - ground truth neighbor IDs
    dimension: int
    metric: str  # 'euclidean' or 'cosine'
    
    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', "
            f"base={self.base_vectors.shape}, "
            f"queries={self.query_vectors.shape}, "
            f"dim={self.dimension}, "
            f"metric='{self.metric}')"
        )


def get_data_dir() -> Path:
    """Get the data directory, create if it doesn't exist."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_file(url: str, dest_path: Path, desc: str) -> None:
    """Download a file with a progress bar."""
    if dest_path.exists():
        print(f"✓ {desc} already exists at {dest_path}")
        return
    
    print(f"Downloading {desc} from {url}...")
    
    # Use requests for HTTP/HTTPS, urllib for FTP
    if url.startswith('ftp://'):
        # For FTP, use urllib (no progress bar unfortunately)
        print(f"  (FTP download, this may take a while...)")
        urlretrieve(url, dest_path)
        print(f"✓ Downloaded to {dest_path}")
    else:
        # For HTTP/HTTPS, use requests with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(desc, total=total_size)
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
        
        print(f"✓ Downloaded to {dest_path}")


def read_fvecs(filepath: Path) -> np.ndarray:
    """
    Read .fvecs format used by SIFT dataset.
    
    Format: [dim(4 bytes)][vector(dim*4 bytes)]...
    """
    with open(filepath, 'rb') as f:
        # Read dimension from first vector
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.array([])
        
        dim = struct.unpack('i', dim_bytes)[0]
        f.seek(0)
        
        # Calculate number of vectors
        file_size = os.path.getsize(filepath)
        vec_size = 4 + dim * 4  # 4 bytes for dim + dim floats
        n_vectors = file_size // vec_size
        
        # Read all vectors
        vectors = np.zeros((n_vectors, dim), dtype=np.float32)
        for i in range(n_vectors):
            d = struct.unpack('i', f.read(4))[0]
            assert d == dim, f"Dimension mismatch at vector {i}: expected {dim}, got {d}"
            vec = struct.unpack('f' * dim, f.read(dim * 4))
            vectors[i] = vec
        
        return vectors


def read_ivecs(filepath: Path) -> np.ndarray:
    """
    Read .ivecs format used by SIFT ground truth.
    
    Format: [dim(4 bytes)][ints(dim*4 bytes)]...
    """
    with open(filepath, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.array([])
        
        dim = struct.unpack('i', dim_bytes)[0]
        f.seek(0)
        
        file_size = os.path.getsize(filepath)
        vec_size = 4 + dim * 4
        n_vectors = file_size // vec_size
        
        vectors = np.zeros((n_vectors, dim), dtype=np.int32)
        for i in range(n_vectors):
            d = struct.unpack('i', f.read(4))[0]
            assert d == dim
            vec = struct.unpack('i' * dim, f.read(dim * 4))
            vectors[i] = vec
        
        return vectors


def load_sift1m(
    limit: Optional[int] = None,
    download: bool = True
) -> Dataset:
    """
    Load SIFT1M dataset.
    
    Dataset details:
    - 1,000,000 base vectors (128D, float32)
    - 10,000 query vectors (128D, float32) - SEPARATE from base
    - Ground truth: 100 nearest neighbors per query
    - Metric: Euclidean distance
    - Size: ~500MB compressed, ~1.2GB uncompressed
    
    Note: Query vectors are NOT in the base set - this is realistic benchmarking.
    The ground truth was pre-computed by the dataset creators.
    
    Args:
        limit: Limit number of base vectors (for quick testing)
        download: Download dataset if not present
    
    Returns:
        Dataset object with vectors and ground truth
    """
    data_dir = get_data_dir()
    sift_dir = data_dir / "sift"
    sift_dir.mkdir(exist_ok=True)
    
    # File paths
    base_file = sift_dir / "sift_base.fvecs"
    query_file = sift_dir / "sift_query.fvecs"
    groundtruth_file = sift_dir / "sift_groundtruth.ivecs"
    
    # Download if needed
    if download and not all(f.exists() for f in [base_file, query_file, groundtruth_file]):
        # SIFT1M is hosted on INRIA servers
        base_url = "ftp://ftp.irisa.fr/local/texmex/corpus"
        
        # Download and extract
        tar_path = data_dir / "sift.tar.gz"
        download_file(
            f"{base_url}/sift.tar.gz",
            tar_path,
            "SIFT1M dataset"
        )
        
        print("Extracting SIFT1M...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        print("✓ SIFT1M ready")
    
    # Read vectors
    print("Loading SIFT1M base vectors...")
    base_vectors = read_fvecs(base_file)
    
    if limit is not None and limit < len(base_vectors):
        print(f"Limiting base vectors to {limit:,}")
        base_vectors = base_vectors[:limit]
    
    print("Loading SIFT1M query vectors...")
    query_vectors = read_fvecs(query_file)
    
    print("Loading SIFT1M ground truth...")
    ground_truth = read_ivecs(groundtruth_file)
    
    print(f"✓ Loaded SIFT1M: {len(base_vectors):,} base, {len(query_vectors):,} queries")
    
    return Dataset(
        name="sift1m",
        base_vectors=base_vectors,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        dimension=128,
        metric='euclidean'
    )


def load_fasttext(
    limit: Optional[int] = None,
    download: bool = True,
    subset_size: int = 1_000_000
) -> Dataset:
    """
    Load FastText word embeddings as a benchmark dataset.
    
    Dataset details:
    - 2M words from Common Crawl (300D, float32)
    - We split into base (90%) and queries (10%) - queries NOT in index
    - We compute ground truth via brute force on the split
    - Metric: Cosine similarity
    - Size: ~5GB uncompressed
    
    Note: Unlike SIFT1M, FastText doesn't come with pre-split query/base sets,
    so we split it ourselves (last 10% as queries) to ensure realistic benchmarking.
    
    Args:
        limit: Limit number of base vectors (for quick testing)
        download: Download dataset if not present
        subset_size: Maximum vectors to load from FastText (it's huge!)
    
    Returns:
        Dataset object with vectors and synthetic ground truth
    """
    data_dir = get_data_dir()
    fasttext_dir = data_dir / "fasttext"
    fasttext_dir.mkdir(exist_ok=True)
    
    vectors_file = fasttext_dir / "crawl-300d-2M.vec"
    
    # Download if needed
    if download and not vectors_file.exists():
        # FastText Common Crawl vectors
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        zip_path = data_dir / "fasttext.zip"
        
        download_file(url, zip_path, "FastText embeddings")
        
        print("Extracting FastText...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fasttext_dir)
        
        print("✓ FastText ready")
    
    # Read FastText vectors
    # Format: word dim1 dim2 ... dim300
    print(f"Loading FastText vectors (up to {subset_size:,})...")
    vectors = []
    
    with open(vectors_file, 'r', encoding='utf-8') as f:
        # First line is: n_words dimension
        n_words, dim = map(int, f.readline().split())
        print(f"FastText metadata: {n_words:,} words, {dim}D")
        
        max_load = min(subset_size, n_words)
        for i, line in enumerate(f):
            if i >= max_load:
                break
            
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vectors.append(vec)
            
            if (i + 1) % 100_000 == 0:
                print(f"  Loaded {i + 1:,} vectors...")
    
    base_vectors = np.array(vectors, dtype=np.float32)
    
    if limit is not None and limit < len(base_vectors):
        print(f"Limiting base vectors to {limit:,}")
        base_vectors = base_vectors[:limit]
    
    # Create query vectors: sample 10,000 random vectors from base
    # Note: For word embeddings, it's CORRECT that queries are in the index!
    # When you search for "king", you expect to find "king" as the top result,
    # followed by semantically similar words like "queen", "monarch", etc.
    n_queries = min(10_000, len(base_vectors))
    query_indices = np.random.choice(len(base_vectors), size=n_queries, replace=False)
    query_vectors = base_vectors[query_indices]
    
    # Generate ground truth using brute force (only compute top 100)
    print(f"Computing ground truth for {n_queries:,} queries (this may take a minute)...")
    k = 100
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)
    
    # Normalize vectors for cosine similarity
    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_normed = base_vectors / (base_norms + 1e-10)
    
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_normed = query_vectors / (query_norms + 1e-10)
    
    # Compute in batches to avoid memory issues
    batch_size = 100
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        batch_queries = query_normed[i:end]
        
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(batch_queries, base_normed.T)
        
        # Get top k indices (argsort returns ascending, so we negate)
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        ground_truth[i:end] = top_k_indices
        
        if (end) % 1000 == 0:
            print(f"  Computed ground truth for {end:,} queries...")
    
    print(f"✓ Loaded FastText: {len(base_vectors):,} base, {len(query_vectors):,} queries")
    
    return Dataset(
        name="fasttext",
        base_vectors=base_vectors,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        dimension=300,
        metric='cosine'
    )


if __name__ == "__main__":
    """
    Run this script to download datasets:
        uv run python benchmarks/datasets.py
    
    This will download both SIFT1M and FastText datasets to benchmarks/data/
    """
    print("=" * 70)
    print("m2vdb Dataset Downloader")
    print("=" * 70)
    print("\nThis will download benchmark datasets to benchmarks/data/")
    print("Total size: ~6GB\n")
    
    # Download SIFT1M
    print("\n" + "=" * 70)
    print("Downloading SIFT1M (128D, 1M vectors, ~500MB compressed)")
    print("=" * 70)
    sift = load_sift1m(limit=None, download=True)
    print(f"\n✓ SIFT1M ready: {sift}\n")
    
    # Download FastText
    print("\n" + "=" * 70)
    print("Downloading FastText (300D, 2M vectors, ~5GB)")
    print("=" * 70)
    fasttext = load_fasttext(limit=None, download=True, subset_size=1_000_000)
    print(f"\n✓ FastText ready: {fasttext}\n")
    
    print("\n" + "=" * 70)
    print("✓ All datasets downloaded successfully!")
    print("=" * 70)
    print("\nYou can now run benchmarks with:")
    print("  uv run python benchmarks/run_benchmarks.py")
    print("\nOr with a smaller dataset for quick testing:")
    print("  uv run python benchmarks/run_benchmarks.py --limit 10000")
