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
        print("  (FTP download, this may take a while...)")
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


def load_sift1m() -> Dataset:
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
    
    The dataset will be automatically downloaded if not present and cached
    as .npy files for fast loading on subsequent calls.
    
    Returns:
        Dataset object with ALL vectors and ground truth
    """
    data_dir = get_data_dir()
    sift_dir = data_dir / "sift"
    sift_dir.mkdir(exist_ok=True)
    
    # File paths
    base_file = sift_dir / "sift_base.fvecs"
    query_file = sift_dir / "sift_query.fvecs"
    groundtruth_file = sift_dir / "sift_groundtruth.ivecs"
    
    # Cached numpy arrays (much faster to load)
    base_cache = sift_dir / "sift_base.npy"
    query_cache = sift_dir / "sift_query.npy"
    gt_cache = sift_dir / "sift_groundtruth.npy"
    
    # Check if we have cached numpy arrays
    if all(f.exists() for f in [base_cache, query_cache, gt_cache]):
        # Fast path: load from numpy cache
        base_vectors = np.load(base_cache)
        query_vectors = np.load(query_cache)
        ground_truth = np.load(gt_cache)
        return Dataset(
            name="sift1m",
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            ground_truth=ground_truth,
            dimension=128,
            metric='euclidean'
        )
    
    # Download if needed
    if not all(f.exists() for f in [base_file, query_file, groundtruth_file]):
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
    
    # Read vectors from original format and cache as numpy
    print("Converting SIFT1M to numpy format (one-time operation)...")
    base_vectors = read_fvecs(base_file)
    query_vectors = read_fvecs(query_file)
    ground_truth = read_ivecs(groundtruth_file)
    
    # Cache for next time
    np.save(base_cache, base_vectors)
    np.save(query_cache, query_vectors)
    np.save(gt_cache, ground_truth)
    print("✓ Cached numpy arrays for faster loading")
    
    return Dataset(
        name="sift1m",
        base_vectors=base_vectors,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        dimension=128,
        metric='euclidean'
    )


def load_fasttext(corpus_size: int = 1_000_000) -> Dataset:
    """
    Load FastText word embeddings as a benchmark dataset.
    
    Dataset details:
    - Configurable corpus size (default 1M words from 2M available)
    - 300D float32 embeddings
    - 10,000 query vectors sampled from corpus
    - Ground truth computed via brute force cosine similarity
    - Metric: Cosine similarity
    - Size: ~5GB uncompressed (full 2M)
    
    Note: For word embeddings, queries ARE in the index. When you search for "king",
    you expect to find "king" as top result, followed by similar words.
    
    The dataset will be automatically downloaded if not present and cached
    as .npy files for fast loading on subsequent calls.
    
    Args:
        corpus_size: Number of words to load as corpus (max 2M)
    
    Returns:
        Dataset object with ALL vectors and ground truth
    """
    data_dir = get_data_dir()
    fasttext_dir = data_dir / "fasttext"
    fasttext_dir.mkdir(exist_ok=True)
    
    vectors_file = fasttext_dir / "crawl-300d-2M.vec"
    
    # Cached numpy arrays (much faster to load)
    base_cache = fasttext_dir / f"fasttext_base_{corpus_size}.npy"
    query_cache = fasttext_dir / f"fasttext_query_{corpus_size}.npy"
    gt_cache = fasttext_dir / f"fasttext_groundtruth_{corpus_size}.npy"
    
    # Check if we have cached numpy arrays
    if all(f.exists() for f in [base_cache, query_cache, gt_cache]):
        # Fast path: load from numpy cache
        base_vectors = np.load(base_cache)
        query_vectors = np.load(query_cache)
        ground_truth = np.load(gt_cache)
        return Dataset(
            name="fasttext",
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            ground_truth=ground_truth,
            dimension=300,
            metric='cosine'
        )
    
    # Download if needed
    if not vectors_file.exists():
        # FastText Common Crawl vectors
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        zip_path = data_dir / "fasttext.zip"
        
        download_file(url, zip_path, "FastText embeddings")
        
        print("Extracting FastText...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fasttext_dir)
        
        print("✓ FastText ready")
    
    # Read FastText vectors from text file and cache
    print(f"Converting FastText to numpy format (one-time operation for {corpus_size:,} vectors)...")
    vectors = []
    
    with open(vectors_file, 'r', encoding='utf-8') as f:
        # First line is: n_words dimension
        n_words, dim = map(int, f.readline().split())
        
        max_load = min(corpus_size, n_words)
        for i, line in enumerate(f):
            if i >= max_load:
                break
            
            parts = line.rstrip().split(' ')
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vectors.append(vec)
    
    base_vectors = np.array(vectors, dtype=np.float32)
    
    # Create query vectors: sample 10K from corpus
    # Note: For word embeddings, it's CORRECT that queries are in the index!
    # When you search for "king", you expect to find "king" as the top result,
    # followed by semantically similar words like "queen", "monarch", etc.
    n_queries = min(10_000, len(base_vectors))
    
    # Generate query indices
    np.random.seed(42)  # Fixed seed for reproducibility
    query_indices = np.random.choice(len(base_vectors), size=n_queries, replace=False)
    query_vectors = base_vectors[query_indices]
    
    # Generate ground truth using brute force (only compute top 100)
    print(f"Computing ground truth for {n_queries:,} queries...")
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
    
    # Cache everything for next time
    np.save(base_cache, base_vectors)
    np.save(query_cache, query_vectors)
    np.save(gt_cache, ground_truth)
    print("✓ Cached numpy arrays for faster loading")
    
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
    sift = load_sift1m()
    print(f"\n✓ SIFT1M ready: {sift}\n")
    
    # Download FastText
    print("\n" + "=" * 70)
    print("Downloading FastText (300D, 1M vectors)")
    print("=" * 70)
    fasttext = load_fasttext(corpus_size=1_000_000)
    print(f"\n✓ FastText ready: {fasttext}\n")
    
    print("\n" + "=" * 70)
    print("✓ All datasets downloaded successfully!")
    print("=" * 70)
    print("\nYou can now run benchmarks with:")
    print("  uv run python benchmarks/run_benchmarks.py")
    print("\nOr with limits for quick testing:")
    print("  uv run python benchmarks/run_benchmarks.py --limit 10000 --n-queries 100")
