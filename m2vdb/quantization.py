import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PQ:
    def __init__(self, dim, num_subspaces=4, centroids_per_subspace=256, seed=42):
        """
        Product Quantizer for vector compression.

        Args:
            dim (int): Dimensionality of input vectors.
            num_subspaces (int): Number of subspaces.
            centroids_per_subspace (int): Number of centroids per subspace.
        """
        assert dim % num_subspaces == 0, "Dimensionality must be divisible by num_subspaces."

        self.dim = dim
        self.num_subspaces = num_subspaces
        self.centroids_per_subspace = centroids_per_subspace
        self.subdim = dim // num_subspaces
        self.codebooks = []
        self.seed = seed

    def fit(self, vecs):
        """Train PQ by fitting KMeans in each subspace (parallelized)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        self.codebooks = []

        def fit_one_subspace(m):
            subvecs = vecs[:, m*self.subdim:(m+1)*self.subdim]
            kmeans = KMeans(n_clusters=self.centroids_per_subspace, random_state=self.seed + m, n_init="auto")
            kmeans.fit(subvecs)
            return kmeans.cluster_centers_

        with ThreadPoolExecutor() as pool:
            self.codebooks = list(pool.map(fit_one_subspace, range(self.num_subspaces)))


    def encode(self, vecs):
        """Convert vectors to PQ codes (parallelized by row)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        codes = np.empty((len(vecs), self.num_subspaces), dtype=np.uint8)

        def encode_one(i):
            row = vecs[i]
            code = np.empty(self.num_subspaces, dtype=np.uint8)
            for m in range(self.num_subspaces):
                subvec = row[m*self.subdim:(m+1)*self.subdim]
                centroids = self.codebooks[m]
                dists = np.linalg.norm(centroids - subvec, axis=1)
                code[m] = np.argmin(dists)
            return i, code

        with ThreadPoolExecutor() as pool:
            for i, code in pool.map(encode_one, range(len(vecs))):
                codes[i] = code

        return codes
    

    def _decode(self, codes):
        """Reconstruct approximate vectors from PQ codes."""
        n = len(codes)
        vecs = np.empty((n, self.dim), dtype=np.float32)

        for m in range(self.num_subspaces):
            centroids = self.codebooks[m]
            vecs[:, m*self.subdim:(m+1)*self.subdim] = centroids[codes[:, m]]

        return vecs




# # --- Simple test code ---
# if __name__ == "__main__":
#     np.random.seed(1337)
#     dim = 64
#     n = 10_000
#     num_subspaces = 8
#     centroids_per_subspace = 32

#     # Create toy data
#     X = np.random.randn(n, dim).astype(np.float32)

#     # Create and fit PQ
#     pq = PQ(dim, num_subspaces=num_subspaces, centroids_per_subspace=centroids_per_subspace)
#     pq.fit(X)

#     # Encode + decode
#     codes = pq.encode(X)
#     X_approx = pq._decode(codes)

#     # Reconstruction error
#     mse = np.mean((X - X_approx) ** 2)
#     print("✅ PQ test complete")
#     print(f"Reconstruction MSE: {mse:.4f}")
#     print(f"Code shape: {codes.shape}, dtype: {codes.dtype}")