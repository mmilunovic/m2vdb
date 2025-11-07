import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ProductQuantizer:
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
        """Convert vectors to PQ codes (vectorized, no useless threading)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        n = vecs.shape[0]
        codes = np.empty((n, self.num_subspaces), dtype=np.uint8)

        for m in range(self.num_subspaces):
            subvecs = vecs[:, m*self.subdim:(m+1)*self.subdim]  # (n_samples, subdim)
            centroids = self.codebooks[m]  # (n_centroids, subdim)

            # Compute distance from each subvector to each centroid
            # Result: (n_samples, n_centroids)
            dists = np.linalg.norm(subvecs[:, None, :] - centroids[None, :, :], axis=2)

            # Take the nearest centroid for each subvector
            codes[:, m] = np.argmin(dists, axis=1)

        return codes

    
    def _decode(self, codes):
        """Reconstruct approximate vectors from PQ codes."""
        n = len(codes)
        vecs = np.empty((n, self.dim), dtype=np.float32)

        for m in range(self.num_subspaces):
            centroids = self.codebooks[m]
            vecs[:, m*self.subdim:(m+1)*self.subdim] = centroids[codes[:, m]]

        return vecs
    
    def build_lookup_table(self, queries: np.ndarray) -> np.ndarray:
        """
        Build lookup tables for a batch of queries.
        
        Args:
            queries (np.ndarray): (n_queries, dim)
            
        Returns:
            lookup_tables (np.ndarray): (n_queries, num_subspaces, centroids_per_subspace)
        """
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = queries.shape[0]
        lookup = np.empty((n_queries, self.num_subspaces, self.centroids_per_subspace), dtype=np.float32)

        for m in range(self.num_subspaces):
            query_subvecs = queries[:, m*self.subdim:(m+1)*self.subdim]  # (n_queries, subdim)
            centroids = self.codebooks[m]  # (centroids_per_subspace, subdim)
            
            # Compute distance: broadcasting (n_queries, 1, subdim) - (1, centroids_per_subspace, subdim)
            diff = query_subvecs[:, None, :] - centroids[None, :, :]
            lookup[:, m, :] = np.linalg.norm(diff, axis=2)  # (n_queries, centroids_per_subspace)
        
        return lookup
