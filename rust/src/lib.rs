// src/lib.rs

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use pyo3::prelude::*;

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

/// Pure Rust brute-force index.
///
/// Internally:
/// - `vectors`: Vec<Vec<f32>>  (like 2D NumPy array)
/// - `ids`:     Vec<String>
/// - `id_to_idx`: HashMap<String, usize>
#[derive(Debug)]
pub struct BruteForceIndex {
    metric: Metric,
    dim: Option<usize>,
    vectors: Vec<Vec<f32>>,
    ids: Vec<String>,
    id_to_idx: HashMap<String, usize>,
}

impl BruteForceIndex {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            dim: None,
            vectors: Vec::new(),
            ids: Vec::new(),
            id_to_idx: HashMap::new(),
        }
    }

    pub fn from_metric_str(metric: &str) -> IndexResult<Self> {
        Ok(Self::new(Metric::from_str(metric)?))
    }

    pub fn is_built(&self) -> bool {
        !self.vectors.is_empty()
    }

    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    pub fn size(&self) -> usize {
        self.vectors.len()
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

        self.vectors = vectors;
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

        let new_idx = self.ids.len();
        self.vectors.push(vector);
        self.ids.push(id.clone());
        self.id_to_idx.insert(id, new_idx);
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> IndexResult<bool> {
        let &idx = match self.id_to_idx.get(id) {
            Some(idx) => idx,
            None => return Ok(false),
        };

        let last_idx = self.ids.len() - 1;

        if idx == last_idx {
            self.vectors.pop();
            let last_id = self.ids.pop().expect("len > 0");
            self.id_to_idx.remove(&last_id);
            return Ok(true);
        }

        let last_id = self.ids[last_idx].clone();
        self.vectors.swap(idx, last_idx);
        self.vectors.pop();

        self.ids[idx] = last_id.clone();
        self.ids.pop();

        self.id_to_idx.insert(last_id, idx);
        self.id_to_idx.remove(id);

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

        let n = self.vectors.len();
        let k = k.min(n);

        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n);

        match self.metric {
            Metric::Cosine => {
                let query_norm = norm_l2(query).max(1e-10);
                for (idx, v) in self.vectors.iter().enumerate() {
                    let v_norm = norm_l2(v).max(1e-10);
                    let dot = dot_product(query, v);
                    let similarity = dot / (query_norm * v_norm);
                    let distance = 1.0 - similarity;
                    dists.push((idx, distance));
                }
            }
            Metric::Euclidean => {
                for (idx, v) in self.vectors.iter().enumerate() {
                    let distance = l2_distance(query, v);
                    dists.push((idx, distance));
                }
            }
        }

        // Full sort for simplicity; you can optimize with partial selection later.
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut results = Vec::with_capacity(k);
        for (idx, dist) in dists.into_iter().take(k) {
            results.push((self.ids[idx].clone(), dist));
        }

        Ok(results)
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

    /// build(self, vectors: List[List[float]], ids: List[str]) -> None
    ///
    /// For simplicity, we accept Python lists of lists; you'll pass `vectors.tolist()` from Python.
    pub fn build(&mut self, vectors: Vec<Vec<f32>>, ids: Vec<String>) -> PyResult<()> {
        self.inner.build(vectors, ids).map_err(to_py_err)
    }

    /// add(self, id: str, vector: List[float]) -> None
    pub fn add(&mut self, id: String, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, vector).map_err(to_py_err)
    }

    /// delete(self, id: str) -> bool
    pub fn delete(&mut self, id: String) -> PyResult<bool> {
        self.inner.delete(&id).map_err(to_py_err)
    }

    /// size(self) -> int
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// search(self, query: List[float], k: int) -> List[Tuple[str, float]]
    ///
    /// Note: we accept `Vec<f32>` and borrow it as `&[f32]`.
    pub fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        self.inner.search(&query, k).map_err(to_py_err)
    }
}

/// Module initializer: this is what Python imports.
///
/// Python:
///   import rust_indexes
///   idx = rust_indexes.BruteForceIndex()
use pyo3::prelude::*; // make sure this is at the top

#[pymodule]
fn rust_indexes(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyBruteForceIndex>()?;
    Ok(())
}