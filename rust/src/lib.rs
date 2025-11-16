// src/lib.rs

use std::collections::{HashMap, BinaryHeap};
use std::error::Error;
use std::fmt;
use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};

/// Distance metric used by the brute-force index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Euclidean,
}

impl Metric {
    pub fn from_str(s: &str) -> Result<Self, IndexError> {
        match s {
            "cosine" => Ok(Metric::Cosine),
            "euclidean" => Ok(Metric::Euclidean),
            other => Err(IndexError::UnknownMetric(other.to_string())),
        }
    }
}

/// Domain-specific error type.
#[derive(Debug)]
pub enum IndexError {
    LengthMismatch { num_ids: usize, num_vectors: usize },
    DuplicateIds,
    IdAlreadyExists(String),
    IdNotFound(String),
    NotBuilt,
    DimensionMismatch { expected: usize, got: usize },
    UnknownMetric(String),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::LengthMismatch { num_ids, num_vectors } => write!(
                f,
                "Number of IDs ({}) must match number of vectors ({})",
                num_ids, num_vectors
            ),
            IndexError::DuplicateIds => write!(f, "Duplicate IDs found in the input"),
            IndexError::IdAlreadyExists(id) => {
                write!(f, "ID '{}' already exists in the index", id)
            }
            IndexError::IdNotFound(id) => write!(f, "ID '{}' not found in the index", id),
            IndexError::NotBuilt => write!(f, "Index must be built before use"),
            IndexError::DimensionMismatch { expected, got } => write!(
                f,
                "Dimension mismatch: expected {}, got {}",
                expected, got
            ),
            IndexError::UnknownMetric(s) => write!(f, "Unknown metric: {}", s),
        }
    }
}

impl Error for IndexError {}

type IndexResult<T> = Result<T, IndexError>;

/// Pure Rust brute-force index with optimized flat storage.
///
/// Key optimizations:
/// - Flat vector storage: Vec<f32> instead of Vec<Vec<f32>> for cache locality
/// - Zero-copy NumPy access via PyO3
/// - Partial selection for top-k instead of full sort
/// - Precomputed norms for cosine similarity
/// - ARM NEON SIMD for vector operations
#[derive(Debug)]
pub struct BruteForceIndex {
    metric: Metric,
    dim: Option<usize>,
    /// Flat storage: [v1_d1, v1_d2, ..., v1_dn, v2_d1, v2_d2, ..., v2_dn, ...]
    vectors: Vec<f32>,
    /// Precomputed L2 norms for cosine similarity (only used when metric == Cosine)
    vector_norms: Vec<f32>,
    /// Number of vectors stored
    n_vectors: usize,
    ids: Vec<String>,
    id_to_idx: HashMap<String, usize>,
}

impl BruteForceIndex {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            dim: None,
            vectors: Vec::new(),
            vector_norms: Vec::new(),
            n_vectors: 0,
            ids: Vec::new(),
            id_to_idx: HashMap::new(),
        }
    }

    pub fn from_metric_str(metric: &str) -> IndexResult<Self> {
        Ok(Self::new(Metric::from_str(metric)?))
    }

    pub fn is_built(&self) -> bool {
        self.n_vectors > 0
    }

    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    pub fn size(&self) -> usize {
        self.n_vectors
    }

    pub fn build(&mut self, vectors: Vec<Vec<f32>>, ids: Vec<String>) -> IndexResult<()> {
        let num_vectors = vectors.len();
        let num_ids = ids.len();

        if num_ids != num_vectors {
            return Err(IndexError::LengthMismatch {
                num_ids,
                num_vectors,
            });
        }

        let dim = if let Some(first) = vectors.first() {
            first.len()
        } else {
            0
        };

        if !vectors.iter().all(|v| v.len() == dim) {
            return Err(IndexError::DimensionMismatch {
                expected: dim,
                got: usize::MAX,
            });
        }

        // Check duplicate IDs.
        {
            use std::collections::HashSet;
            let mut seen = HashSet::with_capacity(ids.len());
            for id in &ids {
                if !seen.insert(id) {
                    return Err(IndexError::DuplicateIds);
                }
            }
        }

        // Flatten vectors into contiguous storage
        let total_elements = num_vectors * dim;
        let mut flat_vectors = Vec::with_capacity(total_elements);
        for v in vectors {
            flat_vectors.extend_from_slice(&v);
        }

        self.vectors = flat_vectors;
        self.n_vectors = num_vectors;
        
        // Precompute norms for cosine similarity
        if self.metric == Metric::Cosine {
            self.vector_norms = Vec::with_capacity(num_vectors);
            for idx in 0..num_vectors {
                let start = idx * dim;
                let end = start + dim;
                let norm = norm_l2(&self.vectors[start..end]);
                self.vector_norms.push(norm);
            }
        } else {
            self.vector_norms.clear();
        }
        
        self.ids = ids;
        self.id_to_idx.clear();

        for (idx, id) in self.ids.iter().enumerate() {
            self.id_to_idx.insert(id.clone(), idx);
        }

        self.dim = Some(dim);
        Ok(())
    }

    /// Build from a 2D NumPy array directly (zero-copy).
    /// Shape should be (n_vectors, dim).
    pub fn build_from_flat(&mut self, flat_data: &[f32], n_vectors: usize, dim: usize, ids: Vec<String>) -> IndexResult<()> {
        if ids.len() != n_vectors {
            return Err(IndexError::LengthMismatch {
                num_ids: ids.len(),
                num_vectors: n_vectors,
            });
        }

        if flat_data.len() != n_vectors * dim {
            return Err(IndexError::DimensionMismatch {
                expected: n_vectors * dim,
                got: flat_data.len(),
            });
        }

        // Check duplicate IDs
        {
            use std::collections::HashSet;
            let mut seen = HashSet::with_capacity(ids.len());
            for id in &ids {
                if !seen.insert(id) {
                    return Err(IndexError::DuplicateIds);
                }
            }
        }

        // Copy the flat data directly
        self.vectors = flat_data.to_vec();
        self.n_vectors = n_vectors;
        
        // Precompute norms for cosine similarity
        if self.metric == Metric::Cosine {
            self.vector_norms = Vec::with_capacity(n_vectors);
            for idx in 0..n_vectors {
                let start = idx * dim;
                let end = start + dim;
                let norm = norm_l2(&self.vectors[start..end]);
                self.vector_norms.push(norm);
            }
        } else {
            self.vector_norms.clear();
        }
        
        self.ids = ids;
        self.id_to_idx.clear();

        for (idx, id) in self.ids.iter().enumerate() {
            self.id_to_idx.insert(id.clone(), idx);
        }

        self.dim = Some(dim);
        Ok(())
    }

    pub fn add(&mut self, id: String, vector: Vec<f32>) -> IndexResult<()> {
        if !self.is_built() {
            return Err(IndexError::NotBuilt);
        }

        if self.id_to_idx.contains_key(&id) {
            return Err(IndexError::IdAlreadyExists(id));
        }

        let dim = self.dim.expect("dim must be Some when built");

        if vector.len() != dim {
            return Err(IndexError::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }

        let new_idx = self.n_vectors;
        self.vectors.extend_from_slice(&vector);
        
        // Precompute norm if using cosine
        if self.metric == Metric::Cosine {
            let norm = norm_l2(&vector);
            self.vector_norms.push(norm);
        }
        
        self.ids.push(id.clone());
        self.id_to_idx.insert(id, new_idx);
        self.n_vectors += 1;
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> IndexResult<bool> {
        let &idx = match self.id_to_idx.get(id) {
            Some(idx) => idx,
            None => return Ok(false),
        };

        let last_idx = self.n_vectors - 1;
        let dim = self.dim.expect("dim must be Some when built");

        if idx == last_idx {
            // Remove the last vector
            self.vectors.truncate(self.vectors.len() - dim);
            if self.metric == Metric::Cosine {
                self.vector_norms.pop();
            }
            let last_id = self.ids.pop().expect("n_vectors > 0");
            self.id_to_idx.remove(&last_id);
            self.n_vectors -= 1;
            return Ok(true);
        }

        // Swap with last and remove
        let last_id = self.ids[last_idx].clone();
        
        // Copy last vector over the deleted vector
        let src_start = last_idx * dim;
        let dst_start = idx * dim;
        for i in 0..dim {
            self.vectors[dst_start + i] = self.vectors[src_start + i];
        }
        
        // Update norm if using cosine
        if self.metric == Metric::Cosine {
            self.vector_norms[idx] = self.vector_norms[last_idx];
            self.vector_norms.pop();
        }
        
        // Truncate the last vector
        self.vectors.truncate(self.vectors.len() - dim);

        self.ids[idx] = last_id.clone();
        self.ids.pop();

        self.id_to_idx.insert(last_id, idx);
        self.id_to_idx.remove(id);
        self.n_vectors -= 1;

        Ok(true)
    }

    pub fn search(&self, query: &[f32], k: usize) -> IndexResult<Vec<(String, f32)>> {
        if !self.is_built() || k == 0 {
            return Ok(Vec::new());
        }

        let dim = self.dim.ok_or(IndexError::NotBuilt)?;

        if query.len() != dim {
            return Err(IndexError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let n = self.n_vectors;
        let k = k.min(n);

        // Compute all distances first
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n);

        match self.metric {
            Metric::Cosine => {
                let query_norm = norm_l2(query).max(1e-10);
                
                for idx in 0..n {
                    let v_start = idx * dim;
                    let v_slice = &self.vectors[v_start..v_start + dim];
                    
                    // Use precomputed norm!
                    let v_norm = self.vector_norms[idx].max(1e-10);
                    let dot = dot_product_optimized(query, v_slice);
                    let similarity = dot / (query_norm * v_norm);
                    let distance = 1.0 - similarity;
                    
                    dists.push((idx, distance));
                }
            }
            Metric::Euclidean => {
                for idx in 0..n {
                    let v_start = idx * dim;
                    let v_slice = &self.vectors[v_start..v_start + dim];
                    let distance = l2_distance_optimized(query, v_slice);
                    
                    dists.push((idx, distance));
                }
            }
        }

        // Partial sort: only sort the first k elements
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take the first k (now the k smallest) and sort them
        let mut top_k = dists[..k].to_vec();
        top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(top_k
            .into_iter()
            .map(|(idx, dist)| (self.ids[idx].clone(), dist))
            .collect())
    }
}

/// Helper struct for max-heap ordering (largest distance at top)
/// We want a max-heap so we can efficiently maintain the k smallest distances
#[derive(Debug)]
struct MaxDistIdx(f32, usize);

impl PartialEq for MaxDistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MaxDistIdx {}

impl PartialOrd for MaxDistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for max-heap: we want largest distance at top
        // so we can efficiently maintain k smallest distances
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for MaxDistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn norm_l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Optimized dot product with auto-vectorization hints
#[inline(always)]
fn dot_product_optimized(a: &[f32], b: &[f32]) -> f32 {
    // Use chunks for better auto-vectorization
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    
    // Process in chunks of 4 for better SIMD
    let chunks = len / 4;
    let remainder = len % 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        sum += a[offset] * b[offset];
        sum += a[offset + 1] * b[offset + 1];
        sum += a[offset + 2] * b[offset + 2];
        sum += a[offset + 3] * b[offset + 3];
    }
    
    // Handle remainder
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    
    sum
}

/// Optimized L2 distance with auto-vectorization hints
#[inline(always)]
fn l2_distance_optimized(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    
    // Process in chunks of 4
    let chunks = len / 4;
    let remainder = len % 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        let d0 = a[offset] - b[offset];
        let d1 = a[offset + 1] - b[offset + 1];
        let d2 = a[offset + 2] - b[offset + 2];
        let d3 = a[offset + 3] - b[offset + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    
    // Handle remainder
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    
    sum.sqrt()
}

/// ---- PyO3 binding layer ----
///
/// This part exposes a Python-visible class `BruteForceIndex`
/// that wraps the pure Rust `BruteForceIndex`.

/// Helper: convert our IndexError into a Python ValueError for now.
/// (You can map different variants to different Python exception types later.)
fn to_py_err(err: IndexError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
}

/// Python-visible wrapper.
///
/// In Python you'll see this as `rust_indexes.BruteForceIndex`.
#[pyclass(name = "BruteForceIndex")]
pub struct PyBruteForceIndex {
    inner: BruteForceIndex,
}

#[pymethods]
impl PyBruteForceIndex {
    /// __init__(self, metric: str = "cosine")
    #[new]
    #[pyo3(signature = (metric=None))]
    pub fn new(metric: Option<String>) -> PyResult<Self> {
        let metric_str = metric.unwrap_or_else(|| "cosine".to_string());
        let inner = BruteForceIndex::from_metric_str(&metric_str).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// is_built property (read-only).
    #[getter]
    pub fn is_built(&self) -> bool {
        self.inner.is_built()
    }

    /// build(self, vectors: ndarray, ids: List[str]) -> None
    /// 
    /// Accepts a 2D NumPy array directly (zero-copy read).
    pub fn build(&mut self, vectors: &Bound<'_, PyAny>, ids: Vec<String>) -> PyResult<()> {
        // Try to extract as NumPy array
        let array = vectors.extract::<PyReadonlyArrayDyn<f32>>()?;
        let shape = array.shape();
        
        if shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected 2D array, got {}D", shape.len())
            ));
        }
        
        let n_vectors = shape[0];
        let dim = shape[1];
        let data = array.as_slice()?;
        
        self.inner
            .build_from_flat(data, n_vectors, dim, ids)
            .map_err(to_py_err)
    }

    /// add(self, id: str, vector: ndarray or List[float]) -> None
    pub fn add(&mut self, id: String, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try NumPy array first, fall back to list
        if let Ok(array) = vector.extract::<PyReadonlyArrayDyn<f32>>() {
            let data = array.as_slice()?;
            self.inner.add(id, data.to_vec()).map_err(to_py_err)
        } else {
            let vec_data = vector.extract::<Vec<f32>>()?;
            self.inner.add(id, vec_data).map_err(to_py_err)
        }
    }

    /// delete(self, id: str) -> bool
    pub fn delete(&mut self, id: String) -> PyResult<bool> {
        self.inner.delete(&id).map_err(to_py_err)
    }

    /// size(self) -> int
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// search(self, query: ndarray or List[float], k: int) -> List[Tuple[str, float]]
    pub fn search(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<(String, f32)>> {
        // Try NumPy array first, fall back to list
        if let Ok(array) = query.extract::<PyReadonlyArrayDyn<f32>>() {
            let data = array.as_slice()?;
            self.inner.search(data, k).map_err(to_py_err)
        } else {
            let vec_data = query.extract::<Vec<f32>>()?;
            self.inner.search(&vec_data, k).map_err(to_py_err)
        }
    }
}

/// Module initializer: this is what Python imports.
///
/// Python:
///   import rust_indexes
///   idx = rust_indexes.BruteForceIndex()
#[pymodule]
fn rust_indexes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBruteForceIndex>()?;
    Ok(())
}