# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-20

### Added

#### Core VSA Primitives (`holon::primitives`)
- `bind` / `unbind` — associative binding (AND-like)
- `bundle` / `prototype` — superposition and majority voting (OR-like)
- `negate` / `negate_with_method` — component removal (NOT-like)
- `amplify` — strengthen a component's presence
- `difference` — delta between two states
- `blend` — weighted interpolation
- `resonance` — dimension-wise agreement filter
- `permute` — circular dimension shift for sequence encoding
- `cleanup` — nearest-neighbor retrieval from a codebook
- `analogy` — relational transfer (A:B::C:?)
- `attend` — soft attention via weighted resonance
- `project` / `reject` — subspace projection and orthogonal complement
- `conditional_bind` — gated binding with threshold or sign modes
- `segment` — structural breakpoint detection in vector streams
- `invert` — codebook decomposition
- `sparsify` / `centroid` / `flip` — vector utilities
- `topk_similar` / `similarity_matrix` — batch similarity
- `entropy` / `complexity` / `coherence` — information measures
- `random_project` — Johnson-Lindenstrauss dimensionality reduction
- `power` — fractional binding (real-valued exponent)
- `autocorrelate` / `cross_correlate` — temporal correlation
- `bundle_with_confidence` — per-dimension agreement margins
- `reflect_about_mean` / `grover_amplify` — quantum-inspired operators
- `drift_rate` — temporal similarity derivative

#### Encoding (`holon::encoder`)
- JSON encoding — convert `serde_json::Value` or JSON strings to vectors
- Sequence encoding — `Bundle`, `Positional`, `Chained`, `Ngram` modes
- Walkable encoding — zero-serialization path for typed Rust structs

#### Scalar Encoding (`holon::scalar`)
- `LogFloat` — log-scale encoding for rates, frequencies, multiplicative quantities
- `LinearFloat` — linear-scale encoding
- `TimeFloat` + `TimeResolution` — temporal encoding with circular hour/day/week components

#### Walkable Trait (`holon::walkable`)
- `Walkable` trait for zero-copy struct encoding
- `ScalarValue` / `ScalarRef` — inline scalar markers (`$log`, `$linear`, `$time`)
- `WalkableValue` / `WalkableRef` — typed walk nodes
- Built-in `Walkable` implementations for `serde_json::Value`, primitives, and collections

#### Memory Layer (`holon::memory`)
- `OnlineSubspace` — CCIPCA incremental PCA for anomaly detection
  - `update` / `update_batch` — online learning
  - `residual` — anomaly score (reconstruction error)
  - `threshold` — adaptive EMA-based anomaly threshold
  - `explained_ratio` — variance explained by top-k components
  - `project` / `reconstruct` — subspace projection
  - `snapshot` / `from_snapshot` — JSON persistence
- `Engram` — named, serializable subspace snapshot with eigenvalue signature
- `EngramLibrary` — pattern library with two-tier matching
  - `add` / `remove` / `get` — library management
  - `match_vec` — cosine-similarity retrieval
  - `match_spectrum` — eigenvalue-spectrum matching
  - `save` / `load` — JSON persistence

#### Accumulator (`holon::accumulator`)
- Streaming frequency-weighted accumulation
- `add` / `normalize` / `threshold` — core operations
- `capacity` / `purity` / `participation_ratio` — health metrics

#### Similarity (`holon::similarity`)
- `Cosine`, `Hamming`, `Euclidean`, `Jaccard` metrics
- `weighted_cosine` / `weighted_euclidean` — per-dimension weights
- `minkowski` — generalized Lp distance
- `significance` — cosine similarity to z-score

#### Facade (`holon::Holon`)
- Unified convenience wrapper exposing all primitives, encoding, accumulator,
  similarity, and memory operations through a single `Holon` struct

#### Optional Features
- `simd` feature — SIMD-accelerated similarity via `simsimd` (up to 200x faster)

#### Examples (17 runnable demos)
- Showcase: `compositional_recall`, `streaming_changepoint`, `config_drift_remediation`
- Memory: `online_anomaly_memory`, `temporal_encoding`
- Detection: `walkable_detection`, `zero_hardcode_detection`, `payload_anomaly_detection`,
  `attack_variant_detection`, `improved_detection`
- Rates: `pure_vector_rate`, `walkable_rate`, `magnitude_aware_encoding`
- Advanced: `explainable_forensics`, `targeted_rate_limiting`, `rate_limit_mitigation`,
  `byte_match_derivation`

[0.1.0]: https://github.com/watmin/holon-rs/releases/tag/v0.1.0
