# Known Issues and Bugs

This document tracks known bugs and issues in the m2vdb codebase, prioritized by severity.

---

## Critical

### 1. Upsert Method Rejects Updates Instead of Performing Them

**Location:** `m2vdb/collection.py:94-95`, `m2vdb/collection.py:138-139`

**Severity:** Critical

The `upsert()` and `batch_upsert()` methods are named to imply "insert or update" functionality, but they reject updates by raising a `ValueError` when an ID already exists. This violates the expected upsert semantic where existing records should be updated, not rejected. Users expecting standard upsert behavior will encounter unexpected errors when trying to update vectors.

---

### 2. PQ Index add() Causes Dimension Mismatch on vstack

**Location:** `m2vdb/indexes/pq.py:286`

**Severity:** Critical

When adding a vector to a PQ index after build, the `_encode_vector()` returns a 1D array for single vectors. While NumPy's `vstack` can handle 1D+2D arrays in some cases, the inconsistent dimensionality between the first add (explicitly reshaped at line 284) and subsequent adds (raw 1D codes) creates fragile code that depends on implicit NumPy behavior. The reshape should be applied consistently for clarity and reliability.

---

## High

### 3. Client SDK Contains Unreachable Dead Code

**Location:** `m2vdb/client.py:109-110`, and similar patterns at lines 47, 85, 130, 147, 240, 261, 278, 297, 314

**Severity:** High

The `_handle_error()` method always raises an exception (all code paths end in `raise`), making any `raise` statements after calling it unreachable. Lines like `raise  # Ensure we never fall through` are dead code that indicates incomplete refactoring or a misunderstanding of control flow. This clutters the codebase and may mislead maintainers about exception handling behavior.

---

### 4. rebuild_strategy Parameter Is Accepted But Ignored

**Location:** `m2vdb/collection.py:30`, `m2vdb/collection.py:257-258`

**Severity:** High

The `Collection.__init__()` accepts a `rebuild_strategy` parameter documented to support 'eager' and 'threshold' modes, but the parameter is never used in `_should_rebuild()`. The method always returns `False` for non-PQ indexes regardless of the strategy setting. This misleads users into thinking they can control rebuild behavior when they cannot.

---

### 5. CORS Configuration Allows Credentials with Wildcard Origin

**Location:** `m2vdb/server.py:80-86`

**Severity:** High

The CORS middleware is configured with `allow_origins=["*"]` and `allow_credentials=True` simultaneously. Per the CORS specification, browsers will reject credentialed requests when the origin is a wildcard. This configuration is invalid and will cause authentication failures for browser-based clients attempting to use credentials. Either specific origins should be listed, or credentials should be disabled with wildcard origins.

---

## Medium

### 6. Redundant is_built Check in PQ Rebuild Logic

**Location:** `m2vdb/collection.py:250`

**Severity:** Medium

The condition `not self.index.is_built` at line 250 is redundant because lines 242-243 already return `True` if the index is not built. By the time execution reaches line 250, the index is guaranteed to be built, making this check always evaluate to `False`. This dead code path means PQ indexes will never trigger a rebuild after initial build, even when the condition seems to suggest otherwise.

---

### 7. Collection Length Uses Index Size Instead of Stored Vectors

**Location:** `m2vdb/collection.py:224`

**Severity:** Medium

The `__len__()` method returns `self.index.size()` rather than `len(self._vectors)`. For PQ indexes that require minimum samples for training, vectors may be stored in `_vectors` but not yet indexed. This causes `len(collection)` to report 0 even when vectors have been upserted, leading to confusing behavior where stored data appears missing.

---

### 8. IVF Index save_artifacts Fails on Unbuilt Index

**Location:** `m2vdb/indexes/ivf.py:489-490`

**Severity:** Medium

When `save_artifacts()` is called on an IVF index that hasn't been built, it raises a `RuntimeError`. However, the PQ index handles this gracefully by saving an "unbuilt" marker. This inconsistency means storage operations will fail for IVF collections that haven't accumulated enough vectors to build, while PQ collections handle this case. The storage layer calls `save_artifacts()` unconditionally, so this can cause crashes.

---

## Low

### 9. Misleading Comments About Control Flow

**Location:** `m2vdb/client.py` (multiple locations)

**Severity:** Low

Comments stating "Ensure we never fall through" appear after code that always raises exceptions, making the comments meaningless and potentially confusing to maintainers trying to understand the exception handling logic.

---

### 10. Missing Vector Dimension Validation at API Level

**Location:** `m2vdb/models.py:55-59`

**Severity:** Low

The `SearchRequest` model accepts any `vector: list[float]` without validating that its length matches the index's dimension. Validation only occurs deep in the index layer via assertions. Moving this validation to the Pydantic model would provide better error messages and fail faster.

---

### 11. IVF Cluster Assignment Loop Not Vectorized

**Location:** `m2vdb/indexes/ivf.py:163-166`

**Severity:** Low

The `_assign_to_clusters()` method uses a Python loop to compute cluster assignments one vector at a time. This could be vectorized using matrix operations for better performance on large batches, though the impact is limited since this is only used during index building.

---

### 12. Hardcoded Test API Keys in Production Code

**Location:** `m2vdb/server.py:34-37`

**Severity:** Low

API keys are hardcoded in the source code (`sk-test-user1`, `sk-test-user2`). While documented as test keys, having authentication credentials in source code is a security anti-pattern. A production deployment would need to replace this with environment-variable-based or database-backed authentication.

---

## Summary

| ID | Name | Severity | Location |
|----|------|----------|----------|
| 1 | Upsert rejects updates | Critical | collection.py:94-95 |
| 2 | PQ add() dimension handling | Critical | pq.py:286 |
| 3 | Unreachable dead code in client | High | client.py:109-110 |
| 4 | rebuild_strategy ignored | High | collection.py:257-258 |
| 5 | Invalid CORS configuration | High | server.py:80-86 |
| 6 | Redundant is_built check | Medium | collection.py:250 |
| 7 | __len__ vs _vectors mismatch | Medium | collection.py:224 |
| 8 | IVF save_artifacts fails unbuilt | Medium | ivf.py:489-490 |
| 9 | Misleading comments | Low | client.py |
| 10 | Missing API dimension validation | Low | models.py:55-59 |
| 11 | IVF loop not vectorized | Low | ivf.py:163-166 |
| 12 | Hardcoded API keys | Low | server.py:34-37 |
