from m2vdb.indexes import create_index, registered_indexes


def test_registry_contains_default_indexes():
    names = set(registered_indexes())
    assert {"brute_force", "ivf", "pq"}.issubset(names)


def test_create_index_round_trip():
    index = create_index("brute_force", dim=8)
    assert index.dim == 8
    assert index.metric == "euclidean"
