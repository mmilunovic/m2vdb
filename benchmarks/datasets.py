# benchmarks/datasets.py
from pathlib import Path
import numpy as np
import tarfile

def read_fvecs(fp):
    """Read FVECS file format"""
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

def read_ivecs(file):
    with open(file, 'rb') as f:
        return np.fromfile(f, dtype='int32').reshape(-1, 101)[:, 1:]

def download_sift1m(data_dir: Path):
    """Download and extract SIFT1M dataset"""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the tar.gz file
    tar_path = data_dir / 'sift.tar.gz'
    if not tar_path.exists():
            # print("Downloading SIFT1M dataset...")
            # try:
            #     with closing(urllib.request.urlopen('ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz')) as r:
            #         with open(tar_path, 'wb') as f:
            #             shutil.copyfileobj(r, f)
            # except Exception as e:
            #     if tar_path.exists():
            #         tar_path.unlink()
            #     raise RuntimeError(
            #         "Error downloading SIFT1M dataset. Please download manually from "
            #         "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz "
            #         f"and place the extracted files in {data_dir}"
            #     ) from e
            
        # Extract the tar.gz file
        print("Extracting dataset...")
        print(tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        
        # Clean up the tar.gz file
        tar_path.unlink()
        print("Dataset ready")

def load_sift1m():
    """Load SIFT1M dataset, downloading it first if necessary"""
    data_dir = Path("benchmarks/data/sift/")
    
    # Check if required files exist
    required_files = ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"]
    files_exist = all((data_dir / f).exists() for f in required_files)
    
    if not files_exist:
        download_sift1m(data_dir)
    
    # Load the dataset
    xb = read_fvecs(data_dir / "sift_base.fvecs")[:10000]
    xq = read_fvecs(data_dir / "sift_query.fvecs")[:1000]
    gt = read_ivecs(data_dir / "sift_groundtruth.ivecs")
    return xb, xq, gt
