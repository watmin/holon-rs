# Missing Primitives: What Holon-rs Should Provide

**Brainstorm of vector operations that belong in the library, not userland.**

The test: "Would a user reasonably implement this themselves, or would they
expect the library to provide it?" If two different applications would write
the same 10 lines of code, it belongs in the library.

---

## Tier 1: Should Definitely Have

### 1. `sparsify(vec, k) → Vector`

Keep only the `k` dimensions with the largest absolute values. Zero the rest.

```rust
// Keep the 500 strongest dimensions out of 4096
let focused = Primitives::sparsify(&noisy_vec, 500);
```

**Why it belongs in core:** This is the vector equivalent of attention —
"what are the most important dimensions?" It improves noise resistance
(weak dimensions are noise), reduces interference in bundling (fewer active
dimensions = less cross-talk), and is the foundation for sparse recovery.

Every VSA paper that discusses capacity mentions that thinning improves
retrieval accuracy. It should be a one-liner.

**Implementation:** Sort dimensions by absolute value, keep top k, zero rest.
~10 lines.

### 2. `capacity(accumulator) → f64`

Estimate how many more items can be bundled before retrieval accuracy
degrades below usable thresholds.

```rust
let remaining = Primitives::capacity(&accumulator, codebook_size);
// "You can add ~340 more items before accuracy drops below 90%"
```

**Why it belongs in core:** The theoretical capacity of a VSA accumulator is
well-studied (Kanerva 2009): approximately `d / (2 * ln(N))` items for
dimensionality `d` and codebook size `N`. But nobody wants to re-derive
this formula. The library should tell you "your accumulator is 73% full."

For the DDoS use case: "how many more packet patterns can this baseline
absorb before it becomes useless?"

**Implementation:** Estimate from dimensionality, current count, and noise
level (ratio of coherent vs random energy). ~20 lines.

### 3. `centroid(vectors) → Vector`

The true geometric average — the mean direction of a set of vectors.
NOT the same as `bundle` (majority vote) or `prototype` (thresholded majority).

```rust
let mean = Primitives::centroid(&[&v1, &v2, &v3]);
```

**Why it's different from bundle:**
- `bundle([+1,+1,-1], [+1,-1,-1])` → `[+1, 0, -1]` (majority vote, integer)
- `centroid([+1,+1,-1], [+1,-1,-1])` → `[+1, 0, -1]` normalized to unit length

For 2 vectors they look similar. For 100 vectors with varying agreement,
centroid preserves the continuous weight of each dimension's agreement,
while bundle collapses to ±1. The centroid is better for interpolation
and gradient-like operations.

**Implementation:** Sum all vectors element-wise (as f64), normalize to
unit length, threshold to bipolar. ~10 lines. (This is essentially what
`Accumulator::normalize()` does, but as a standalone operation on a slice
of vectors without an accumulator.)

### 4. `flip(vec) → Vector`

Negate every element: +1 → -1, -1 → +1, 0 → 0.

```rust
let opposite = Primitives::flip(&vec);
// similarity(vec, opposite) ≈ -1.0
```

**Why it belongs in core:** This is the logical NOT of a vector. The
"opposite" of a concept. `flip(baseline)` is "everything that ISN'T
baseline." It's `bind(vec, all_negative_ones)` but that's non-obvious and
allocates a temporary vector.

Currently if you want the complement of a vector you'd either bind with a
negation vector or manually negate each element. Both are awkward.

**Implementation:** Map each element: `x → -x`. 3 lines.

### 5. `topk_similar(query, candidates, k) → Vec<(usize, f64)>`

Find the k most similar vectors to a query from a candidate set.

```rust
let matches = Primitives::topk_similar(&probe, &codebook, 5);
// [(idx=7, sim=0.89), (idx=3, sim=0.76), ...]
```

**Why it belongs in core:** This is THE fundamental retrieval operation.
`cleanup` returns only the best match. `invert` is specialized for
decomposition with a threshold. Generic top-k search is what every
application needs for nearest-neighbor, classification, and recommendation.

**Implementation:** Compute all similarities, partial sort for top-k.
With SIMD acceleration for the similarity computations. ~15 lines.

### 6. `similarity_matrix(vectors) → Vec<Vec<f64>>`

Compute all pairwise similarities for a set of vectors.

```rust
let matrix = Primitives::similarity_matrix(&window_vectors);
// matrix[i][j] = similarity(vectors[i], vectors[j])
```

**Why it belongs in core:** Clustering, visualization, and batch analysis
all need pairwise similarities. Computing them one at a time is O(N²) calls
with per-call overhead. A batch operation can use SIMD across all pairs.

For the DDoS use case: "which detection windows are similar to each other?"
reveals attack phases without explicit segmentation.

**Implementation:** O(N²/2) similarity computations (symmetric matrix).
SIMD-friendly inner loop. ~20 lines.

---

## Tier 2: Strong Case for Inclusion

### 7. `entropy(vec) → f64`

Information-theoretic entropy of the vector's element distribution.

```rust
let h = Primitives::entropy(&vec);
// 0.0 = all same sign (fully determined)
// 1.0 = equal +1 and -1 (maximum uncertainty)
```

**Why it matters:** Entropy measures how much "information" a vector carries.
A vector that's all +1s is useless (carries no information). A vector with
a rich mix of +1 and -1 is maximally informative. For three-valued vectors
(+1, -1, 0), entropy also captures sparsity.

Different from `complexity` (which measures density × balance). Entropy is
the information-theoretic measure — it tells you the compression limit.

**Use case:** After bundling many items, is the accumulator "saturated"
(high entropy, lost all structure) or still informative (lower entropy,
dominant patterns visible)?

**Implementation:** Count +1, -1, 0 proportions, compute
`-Σ p_i * log2(p_i)`. ~10 lines.

### 8. `random_project(vec, target_dims, seed) → Vector`

Reduce dimensionality via random projection (Johnson-Lindenstrauss).

```rust
// Compress 4096-dim to 256-dim while preserving similarities
let compact = Primitives::random_project(&vec, 256, 42);
```

**Why it matters:** For storage, transmission, or approximate search,
4096 dimensions is overkill. JL lemma guarantees that random projection
preserves pairwise distances within ε with high probability if the target
dimension is O(log(N)/ε²).

256 dimensions (256 bytes at i8) preserves most similarity structure while
being 16× smaller. Useful for logging, network transmission, or building
compact indices.

**Implementation:** Generate a random projection matrix (seeded for
determinism), multiply, threshold. ~20 lines.

### 9. `power(vec, exponent) → Vector`

Fractional binding: raise a vector to a real-valued power.

```rust
let half_bound = Primitives::power(&role_vec, 0.5);
// "Half" of the binding — a soft association
```

**Why it matters:** In frequency-domain VSA (HRR), `power(v, k)` is
`exp(i * k * phase)` per dimension. For bipolar VSA, fractional power is
an interpolation between the identity (power=0) and the vector itself
(power=1). Powers > 1 sharpen the vector (increase contrast).

This enables continuous binding strength: instead of "bound" vs "not bound,"
you get "weakly associated" to "strongly associated." Useful for graded
similarity, temporal decay of associations, and fuzzy matching.

**Implementation:** For bipolar, `power(v, k)` where k is integer is
repeated binding. For fractional k, interpolate between power-floor and
power-ceil. ~15 lines.

### 10. `autocorrelate(stream, max_lag) → Vec<f64>`

Compute similarity of a vector stream with itself at different time lags.

```rust
let acf = Primitives::autocorrelate(&window_history, 20);
// acf[0] = 1.0 (identical), acf[1] = sim(t, t-1), ...
// Peaks at lag k indicate period-k patterns
```

**Why it matters:** Detects periodicity in vector streams. If traffic has
a 10-window cycle (attack ramps, subsides, ramps again), the
autocorrelation shows a peak at lag 10. This complements `segment` (which
finds breakpoints) by finding repetitions.

**Use case:** Detect pulsing attacks that toggle on and off at regular
intervals — a pattern that drift detection might normalize to.

**Implementation:** For each lag k, compute mean similarity between
`stream[i]` and `stream[i-k]`. ~15 lines.

### 11. `cross_correlate(stream_a, stream_b, max_lag) → Vec<f64>`

Compute similarity between two vector streams at different time offsets.

```rust
let xcf = Primitives::cross_correlate(&src_ip_stream, &dst_port_stream, 10);
// Peak at lag 3 means dst_port patterns follow src_ip patterns by 3 windows
```

**Why it matters:** Detects causal relationships between different aspects
of traffic. "Source IP diversity changed, and 3 windows later destination
port concentration spiked" — that's a coordinated attack with a command
lag.

**Implementation:** Same as autocorrelation but between two different
streams. ~15 lines.

---

## Tier 3: Niche but Potentially Powerful

### 12. `decompose(vec, max_components) → Vec<(Vector, f64)>`

Blind source separation: decompose a superposition into its components
WITHOUT a known codebook.

Unlike `invert` (which needs a codebook to match against), `decompose`
extracts the dominant independent signals from a superposition.

**Why it's hard:** This is essentially ICA (Independent Component Analysis)
or NMF (Non-negative Matrix Factorization) in VSA space. For bipolar
vectors, it might use iterative projection: find the strongest signal,
subtract it (negate), find the next strongest, repeat.

**Why it's interesting:** "What are the top 3 independent traffic patterns
in this accumulator?" Without knowing what patterns to look for.

### 13. `mutual_information(vec_a, vec_b) → f64`

How much knowing one vector tells you about the other. Goes beyond
cosine similarity (which measures linear agreement) to capture non-linear
dependencies.

**Implementation:** Discretize dimensions into bins, compute joint and
marginal distributions, apply MI formula. ~30 lines.

### 14. `context_gate(vec, context, temperature) → Vector`

Soft context-dependent thinning. Given a context vector, modulate the
target vector so that context-relevant dimensions are amplified and
irrelevant ones are suppressed.

Different from `attend` (which uses query-memory similarity as a gate).
`context_gate` uses the context vector directly as a relevance mask with
a temperature parameter controlling sharpness.

```rust
// Only keep the "TCP-relevant" dimensions of the packet vector
let tcp_focused = Primitives::context_gate(&packet_vec, &tcp_context, 2.0);
```

### 15. `kronecker(vec_a, vec_b) → Vector`

Tensor product that expands dimensionality: two N-dim vectors → one N²-dim
vector (or compressed to N-dim via folding). Captures ALL pairwise
interactions between dimensions.

Standard binding (element-wise multiply) only captures same-index
interactions. Kronecker captures cross-index interactions. This is richer
but more expensive.

**When you'd use it:** When binding loses too much structure. Two vectors
that are both "moderately similar" to a third might have identical bindings
but different Kronecker products.

---

## Summary: Recommended Additions

**Must-add (Tier 1):**

| Primitive | Lines | Value |
|---|---|---|
| `sparsify(vec, k)` | ~10 | Noise filtering, capacity improvement |
| `capacity(acc, codebook_size)` | ~20 | "How full is this accumulator?" |
| `centroid(vectors)` | ~10 | True geometric average |
| `flip(vec)` | ~3 | Logical NOT of a vector |
| `topk_similar(query, candidates, k)` | ~15 | Fundamental retrieval operation |
| `similarity_matrix(vectors)` | ~20 | Batch pairwise comparison |

**Should-add (Tier 2):**

| Primitive | Lines | Value |
|---|---|---|
| `entropy(vec)` | ~10 | Information content measurement |
| `random_project(vec, dims, seed)` | ~20 | Dimensionality reduction |
| `power(vec, exponent)` | ~15 | Continuous binding strength |
| `autocorrelate(stream, lag)` | ~15 | Periodicity detection |
| `cross_correlate(a, b, lag)` | ~15 | Causal relationship detection |

**Consider (Tier 3):**

| Primitive | Lines | Value |
|---|---|---|
| `decompose(vec, k)` | ~40 | Blind source separation |
| `mutual_information(a, b)` | ~30 | Non-linear dependency |
| `context_gate(vec, ctx, temp)` | ~15 | Soft context filtering |
| `kronecker(a, b)` | ~20 | Rich structural binding |

Total: ~80 lines for Tier 1, ~75 lines for Tier 2. The whole batch is
smaller than a single test file.
