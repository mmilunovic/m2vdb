"""Definitions of shared index abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol

import numpy as np


class MetricFn(Protocol):
    """Protocol describing the signature for distance/similarity functions."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class BaseIndex(ABC):
    """Abstract base class implemented by all index strategies."""

    def __init__(self, dim: int, metric: str = "euclidean", metric_fn: Optional[MetricFn] = None) -> None:
        self.dim = dim
        self.metric = metric
        self._metric_fn: MetricFn = metric_fn if metric_fn is not None else self._resolve_metric(metric)
        self.metadata: Dict[int, Dict] = {}

    @staticmethod
    def _resolve_metric(metric: str) -> MetricFn:
        from m2vdb.metrics import get_metric

        return get_metric(metric)

    @abstractmethod
    def add(
        self,
        vecs: np.ndarray,
        ids: Optional[List[int]] = None,
        metadata: Optional[Dict[int, Dict]] = None,
    ) -> None:
        """Add vectors to the index."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for the *k* nearest neighbours for every query."""
