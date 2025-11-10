"""Distance and similarity metric registry."""

from __future__ import annotations

from typing import Dict

from .distances import cosine_similarity, euclidean_distance, ip_distance

_METRICS: Dict[str, callable] = {
    "euclidean": euclidean_distance,
    "cosine": cosine_similarity,
    "inner_product": ip_distance,
    "ip": ip_distance,
}


def register_metric(name: str, fn) -> None:
    _METRICS[name] = fn


def get_metric(name: str):
    try:
        return _METRICS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown metric '{name}'. Available: {sorted(_METRICS)}") from exc


__all__ = [
    "get_metric",
    "register_metric",
    "cosine_similarity",
    "euclidean_distance",
    "ip_distance",
]
