# Python → Rust Parity Gaps

**Audit date:** February 2026
**Last updated:** February 2026 (memory layer + TimeScale added)

The core algebra, encoding, accumulator, distance metrics, vector manager
utilities, memory layer, and temporal encoding are all at full parity.
The only remaining gaps are low-priority domain-specific convenience wrappers.

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

## Memory Layer (memory/) — ✅ COMPLETE

Python's `holon/memory/` contains two modules ported to `src/memory/`.

| Component | Rust API | Status |
|---|---|---|
| `OnlineSubspace` | `memory::OnlineSubspace` | ✅ |
| `OnlineSubspace.update()` | `.update(&[f64]) -> f64` | ✅ |
| `OnlineSubspace.residual()` | `.residual(&[f64]) -> f64` | ✅ |
| `OnlineSubspace.threshold()` | `.threshold() -> f64` | ✅ |
| `OnlineSubspace.eigenvalues()` | `.eigenvalues() -> Vec<f64>` | ✅ |
| `OnlineSubspace.explained_ratio()` | `.explained_ratio() -> f64` | ✅ |
| `OnlineSubspace.project()` | `.project(&[f64]) -> Vec<f64>` | ✅ |
| `OnlineSubspace.reconstruct()` | `.reconstruct(&[f64]) -> Vec<f64>` | ✅ |
| `OnlineSubspace.anomalous_component()` | `.anomalous_component(&[f64]) -> Vec<f64>` | ✅ |
| `OnlineSubspace.update_batch()` | `.update_batch(&[Vec<f64>]) -> Vec<f64>` | ✅ |
| `OnlineSubspace.snapshot()` | `.snapshot() -> SubspaceSnapshot` | ✅ |
| `OnlineSubspace.from_snapshot()` | `::from_snapshot(snap)` | ✅ |
| `Engram` | `memory::Engram` | ✅ |
| `Engram.residual()` | `.residual(&[f64]) -> f64` | ✅ |
| `Engram.eigenvalue_signature()` | `.eigenvalue_signature() -> &[f64]` | ✅ |
| `Engram.surprise_profile()` | `.surprise_profile() -> &HashMap` | ✅ |
| `Engram.metadata()` | `.metadata() -> &HashMap` | ✅ |
| `EngramLibrary` | `memory::EngramLibrary` | ✅ |
| `EngramLibrary.add()` | `.add(name, subspace, ...)` | ✅ |
| `EngramLibrary.match_vec()` | `.match_vec(x, top_k, prefilter_k)` | ✅ |
| `EngramLibrary.match_spectrum()` | `.match_spectrum(eigs, top_k)` | ✅ |
| `EngramLibrary.save/load()` | `.save(path)` / `::load(path)` | ✅ |

**Input convention:** `OnlineSubspace` accepts `&[f64]`, not `&Vector`. Convert
at the call site with `my_vector.to_f64()`. This decouples the memory layer
from the bipolar vector type and allows any float source as input.

**Algorithm:** CCIPCA (Weng et al., 2003) — exact port from Python including
the EMA-based adaptive threshold, warmup averaging, and Gram-Schmidt
re-orthogonalization. Formula constants are identical so Python and Rust
produce the same residuals given the same inputs.

**Skipped:** Persistence backends (CPUStore, QdrantStore). The Engram JSON
serialisation (`save/load`) is sufficient for the use cases we care about.

## TimeScale Encoding (walkable.rs + encoder.rs) — ✅ COMPLETE

Python's `TimeScale` walkable wrapper is ported as `TimeFloat`/`TimeResolution`.

| Component | Rust API | Status |
|---|---|---|
| `TimeScale` wrapper class | `ScalarValue::TimeFloat { value, resolution }` | ✅ |
| `TimeResolution` enum | `TimeResolution::{Second,Minute,Hour,Day}` | ✅ |
| `ScalarValue::time(ts)` | shorthand, default Hour resolution | ✅ |
| `ScalarRef::time(ts)` | zero-allocation path | ✅ |
| `WalkableRef::time(ts)` | convenience constructor | ✅ |
| Circular hour-of-day component | 0–24 circular, period=24 | ✅ |
| Circular day-of-week component | 0–7 circular, period=7 | ✅ |
| Circular month component | 0–12 circular, period=12 | ✅ |
| Positional component | transformer sin/cos, resolution-dependent | ✅ |
| Role vector names | `__time_role_hour__`, `__time_role_dow__`, etc. | ✅ match Python |

**Cross-language determinism:** Role vector names match Python's exactly so that
the same seed produces identical encodings in both languages.

**No new dependencies:** Unix timestamp decomposition uses integer arithmetic
only. No `chrono` required.

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

Everything is at full parity:

- All 13 core VSA primitives (bind, unbind, bundle, negate, amplify, etc.)
- All 8 extended algebra operations (attend, analogy, project, etc.)
- All 8 accumulator operations (add, decay, merge, threshold, etc.)
- All 4 sequence encoding modes (bundle, positional, chained, ngram)
- All 4 scalar encoding modes (linear, log, circular, time/temporal)
- All 11 distance/similarity metrics
- Vector manager with position vectors, codebook export/import
- JSON and Walkable encoding
- Memory layer: OnlineSubspace (CCIPCA), Engram, EngramLibrary
- SIMD-accelerated similarity (Rust-only advantage)

## Remaining (Low Priority)

1. **Mathematical primitives** — domain-specific encodings, better as extension crate
2. **AdvancedSimilarityEngine** — convenience composites, easy to build in userland
3. **Persistence backends (CPUStore, QdrantStore)** — deferred by design
