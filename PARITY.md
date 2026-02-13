# Python → Rust Parity Gaps

**Audit date:** February 2026

The core algebra, encoding, and accumulator are at full parity. These are
the remaining gaps from the Python implementation that need porting.

## Distance Metrics (similarity.rs)

Python's `distance.py` has 11 metrics. Rust has 6. Missing:

| Metric | Python Function | Priority | Notes |
|---|---|---|---|
| Agreement | `agreement_similarity()` | Medium | Non-zero dimension agreement rate |
| Chebyshev | `chebyshev_distance/similarity()` | Low | L∞ norm, max single-dimension diff |
| Minkowski | `minkowski_distance()` | Low | Generalized Lp norm (p parameter) |
| Weighted Cosine | `weighted_cosine_similarity()` | Medium | Per-dimension weight vector |
| Weighted Euclidean | `weighted_euclidean_distance()` | Low | Per-dimension weight vector |

**Implementation:** Add variants to `Metric` enum in `similarity.rs`. Each
is 5-15 lines. Weighted metrics need a `weights: &[f64]` parameter — either
add to `Metric` enum as `WeightedCosine { weights }` or add separate methods
on `Similarity`.

## VectorManager Utilities (vector_manager.rs)

| Function | Python | Priority | Notes |
|---|---|---|---|
| `get_position_vector(pos)` | `vector_manager.py:152` | Medium | Positional encoding for sequences |
| `export_codebook()` | `vector_manager.py:198` | Low | Serialize cached vectors to bytes |
| `import_codebook(data)` | `vector_manager.py:204` | Low | Restore cached vectors from bytes |
| `verify_determinism(atoms, n)` | `vector_manager.py:210` | Low | Test that same atom → same vector across calls |

**`get_position_vector`** is the most useful — it's used by positional
sequence encoding. Check if `encode_sequence(Positional)` in Rust generates
position vectors internally or needs this.

**Codebook export/import** enables saving and restoring the vector cache
across process restarts. Useful for distributed deployments where multiple
nodes need identical vector mappings. Low priority since deterministic
seeding already guarantees identical vectors for identical atoms.

**`verify_determinism`** is a test utility. Could be a `#[test]` instead
of a public API method.

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
- JSON and Walkable encoding
- SIMD-accelerated similarity (Rust-only advantage)

## Suggested Porting Order

1. **Agreement similarity** — small, useful for the DDoS detection loop
2. **Weighted cosine** — useful for importance-weighted comparisons
3. **get_position_vector** — check if sequence encoding needs it
4. **Chebyshev / Minkowski** — completeness, low urgency
5. **Codebook export/import** — when distributed deployment is needed
6. **Mathematical primitives** — when non-network use cases arise
