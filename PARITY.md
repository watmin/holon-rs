# Python → Rust Parity Gaps

**Audit date:** February 2026
**Last updated:** February 2026

The core algebra, encoding, accumulator, distance metrics, and vector manager
utilities are at full parity. These are the remaining gaps from the Python
implementation that need porting.

## Distance Metrics (similarity.rs) — ✅ COMPLETE

Python's `distance.py` has 11 metrics. Rust now has all 11.

| Metric | Rust API | Status |
|---|---|---|
| Cosine | `Metric::Cosine` | ✅ |
| Dot Product | `Metric::Dot` | ✅ |
| Euclidean | `Metric::Euclidean` | ✅ |
| Manhattan | `Metric::Manhattan` | ✅ |
| Hamming | `Metric::Hamming` | ✅ |
| Overlap | `Metric::Overlap` | ✅ |
| Agreement | `Metric::Agreement` | ✅ Added |
| Chebyshev | `Metric::Chebyshev` | ✅ Added |
| Minkowski | `Similarity::minkowski(a, b, p)` | ✅ Added (standalone, needs `p`) |
| Weighted Cosine | `Similarity::weighted_cosine(a, b, weights)` | ✅ Added (standalone, needs `weights`) |
| Weighted Euclidean | `Similarity::weighted_euclidean(a, b, weights)` | ✅ Added (standalone, needs `weights`) |

**Design note:** Agreement and Chebyshev are on the `Metric` enum (dispatched
via `Similarity::compute()`). Minkowski, Weighted Cosine, and Weighted Euclidean
are standalone methods on `Similarity` because they require extra parameters
(`p: f64` or `weights: &[f64]`) that don't fit in a `Copy + Eq` enum. All three
are also exposed on `Holon` as convenience methods.

## VectorManager Utilities (vector_manager.rs) — ✅ COMPLETE

| Function | Rust API | Status |
|---|---|---|
| `get_position_vector(pos)` | `VectorManager::get_position_vector(pos)` | ✅ Added |
| `export_codebook()` | `VectorManager::export_codebook()` | ✅ Added |
| `import_codebook(data)` | `VectorManager::import_codebook(codebook)` | ✅ Added |
| `verify_determinism(atoms, n)` | `#[test] test_verify_determinism` | ✅ Added as test |

**`get_position_vector`** uses `__pos__{position}` as the atom key, matching
the Python `_position_to_seed()` approach. Position vectors are cached
separately from atom vectors.

**Codebook export/import** serializes the atom cache as `HashMap<String, Vec<i8>>`.
Import overwrites on conflict, preserving existing entries otherwise.

**`verify_determinism`** is a `#[test]` rather than a public API method, since
it's a testing concern — same as the Python version, but more idiomatic for Rust.

## Mathematical Primitives (encoder.py)

Python has 8 specialized encoding primitives:

```python
class MathematicalPrimitive(Enum):
    CONVERGENCE_RATE
    ITERATION_COMPLEXITY
    FREQUENCY_DOMAIN
    AMPLITUDE_SCALE
    POWER_LAW_EXPONENT
    CLUSTERING_COEFFICIENT
    TOPOLOGICAL_DISTANCE
    SELF_SIMILARITY
```

Each maps a mathematical concept to a vector encoding. These were designed
for encoding mathematical/scientific data, not network traffic. **Low
priority** for the DDoS use case but may matter for other Holon applications.

**Assessment:** These are domain-specific encodings built on top of the
scalar encoder. They could live in a separate `math_encodings` module or
even in userland rather than the core library. Review whether they belong
in core or should be an extension crate.

## AdvancedSimilarityEngine (similarity.py)

Python has composite similarity strategies:

| Method | Description | Priority |
|---|---|---|
| `multi_metric_similarity()` | Weighted blend of multiple metrics | Low |
| `contextual_similarity()` | Metadata-modulated similarity | Low |
| `hierarchical_similarity()` | Multi-resolution comparison | Low |
| `ensemble_similarity()` | Ensemble of similarity strategies | Low |

**Assessment:** These are convenience wrappers over the core similarity
primitives. They compose existing operations rather than introducing new
algebra. Could be implemented in userland. Low priority for core.

## Walkable Extensibility

Python has runtime Walkable registration:

```python
register_walkable_adapter(target_type, adapter_factory)

@register_walkable
class MyType: ...
```

Rust uses compile-time trait implementation (`impl Walkable for T`), which
is the idiomatic approach. No gap here — different languages, different
patterns. The Rust `Walkable` trait is more type-safe.

## What's NOT a Gap

Everything else is at full parity:

- All 13 core VSA primitives (bind, unbind, bundle, negate, amplify, etc.)
- All 8 extended algebra operations (attend, analogy, project, etc.)
- All 8 accumulator operations (add, decay, merge, threshold, etc.)
- All 4 sequence encoding modes (bundle, positional, chained, ngram)
- All 3 scalar encoding modes (linear, log, circular)
- All 11 distance/similarity metrics
- Vector manager with position vectors, codebook export/import
- JSON and Walkable encoding
- SIMD-accelerated similarity (Rust-only advantage)

## Remaining (Low Priority)

1. **Mathematical primitives** — domain-specific encodings, better as extension crate
2. **AdvancedSimilarityEngine** — convenience composites, easy to build in userland
