"""
Shared test fixtures for m2vdb unit tests.
"""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def vectors_3d():
    """Simple 3D unit vectors for basic tests."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7071, 0.7071, 0.0],
    ], dtype=np.float32)


@pytest.fixture
def ids_3d():
    """IDs for 3D vectors."""
    return ["v0", "v1", "v2", "v3"]


@pytest.fixture
def vectors_128d(rng):
    """200 random 128D vectors for testing (deterministic)."""
    return rng.random((200, 128)).astype(np.float32)


@pytest.fixture
def ids_128d():
    """IDs for 128D vectors."""
    return [f"vec-{i}" for i in range(200)]


@pytest.fixture
def vectors_128d_small(rng):
    """50 random 128D vectors for lighter tests."""
    return rng.random((50, 128)).astype(np.float32)


@pytest.fixture
def ids_128d_small():
    """IDs for small 128D vectors."""
    return [f"vec-{i}" for i in range(50)]


@pytest.fixture
def tmp_storage(tmp_path):
    """Temporary storage directory for CollectionManager tests."""
    from m2vdb.storage import CollectionManager
    return CollectionManager(tmp_path / "data")
