//! Convenience wrapper that owns an [`Encoder`], [`VectorManager`], and
//! [`ScalarEncoder`] and delegates to the kernel and memory layers.
//!
//! For full control, import from [`kernel`](crate::kernel) and
//! [`memory`](crate::memory) directly.

use crate::error::Result;
use crate::kernel::{
    Accumulator, AttendMode, Encoder, GateMode, Metric, NegateMethod, Primitives, ScalarEncoder,
    ScalarMode, SegmentMethod, SequenceMode, Similarity, Vector, VectorManager, Walkable,
};
use crate::memory::{EngramLibrary, OnlineSubspace};

/// Convenience wrapper over the kernel and memory layers.
///
/// `Holon` owns an [`Encoder`], [`VectorManager`], and [`ScalarEncoder`],
/// providing an ergonomic API for common workflows. Every method delegates
/// to the same public types you can use directly from [`crate::kernel`] and
/// [`crate::memory`].
///
/// # When to use `Holon` vs direct imports
///
/// | Use case | Recommendation |
/// |----------|---------------|
/// | Quick scripts, examples | `Holon` (less boilerplate) |
/// | Library code, production | Direct [`kernel`](crate::kernel) / [`memory`](crate::memory) imports |
///
/// # Example
///
/// ```rust
/// use holon::highlevel::Holon;
///
/// let holon = Holon::new(4096);
///
/// let vec = holon.encode_json(r#"{"type": "billing"}"#).unwrap();
///
/// let mut accum = holon.create_accumulator();
/// holon.accumulate(&mut accum, &vec);
/// let baseline = holon.normalize_accumulator(&accum);
/// ```
pub struct Holon {
    dimensions: usize,
    vector_manager: VectorManager,
    encoder: Encoder,
    scalar_encoder: ScalarEncoder,
}

impl Holon {
    /// Create a new Holon instance with the specified dimensionality.
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality (recommended: 4096 or 8192)
    pub fn new(dimensions: usize) -> Self {
        let vector_manager = VectorManager::new(dimensions);
        let encoder = Encoder::new(vector_manager.clone());
        let scalar_encoder = ScalarEncoder::new(dimensions);

        Self {
            dimensions,
            vector_manager,
            encoder,
            scalar_encoder,
        }
    }

    /// Create a new Holon instance with a specific global seed.
    ///
    /// Using the same seed guarantees deterministic, reproducible vectors
    /// across different runs and machines.
    pub fn with_seed(dimensions: usize, global_seed: u64) -> Self {
        let vector_manager = VectorManager::with_seed(dimensions, global_seed);
        let encoder = Encoder::new(vector_manager.clone());
        let scalar_encoder = ScalarEncoder::new(dimensions);

        Self {
            dimensions,
            vector_manager,
            encoder,
            scalar_encoder,
        }
    }

    /// Get the vector dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    // =========================================================================
    // Encoding
    // =========================================================================

    /// Encode a JSON string into a vector.
    pub fn encode_json(&self, json: &str) -> Result<Vector> {
        self.encoder.encode_json(json)
    }

    /// Encode a serde_json Value directly.
    pub fn encode_value(&self, value: &serde_json::Value) -> Vector {
        self.encoder.encode_value(value, None)
    }

    /// Get the base vector for an atomic value.
    ///
    /// Same atom always produces the same vector (deterministic).
    pub fn get_vector(&self, atom: &str) -> Vector {
        self.vector_manager.get_vector(atom)
    }

    /// Encode a continuous scalar value.
    pub fn encode_scalar(&self, value: f64, mode: ScalarMode) -> Vector {
        self.scalar_encoder.encode(value, mode)
    }

    /// Encode a scalar on log scale.
    ///
    /// Equal ratios have equal similarity:
    /// 100 -> 1000 has the same similarity drop as 1000 -> 10000.
    pub fn encode_scalar_log(&self, value: f64) -> Vector {
        self.scalar_encoder.encode_log(value)
    }

    /// Encode a sequence of items.
    pub fn encode_sequence(&self, items: &[&str], mode: SequenceMode) -> Vector {
        self.encoder.encode_sequence(items, mode)
    }

    /// Encode any type implementing the [`Walkable`] trait.
    pub fn encode_walkable<W: Walkable>(&self, walkable: &W) -> Vector {
        self.encoder.encode_walkable(walkable)
    }

    /// Encode a [`WalkableValue`](crate::kernel::WalkableValue) directly.
    pub fn encode_walkable_value(&self, value: &crate::kernel::WalkableValue) -> Vector {
        self.encoder.encode_walkable_value(value)
    }

    // =========================================================================
    // VSA Primitives
    // =========================================================================

    /// Bind two vectors (AND-like association).
    pub fn bind(&self, a: &Vector, b: &Vector) -> Vector {
        Primitives::bind(a, b)
    }

    /// Unbind to retrieve associated value (inverse of bind).
    pub fn unbind(&self, bound: &Vector, key: &Vector) -> Vector {
        Primitives::bind(bound, key)
    }

    /// Bundle multiple vectors (OR-like superposition).
    pub fn bundle(&self, vectors: &[&Vector]) -> Vector {
        Primitives::bundle(vectors)
    }

    /// Remove a component's influence from a superposition.
    pub fn negate(&self, superposition: &Vector, component: &Vector) -> Vector {
        Primitives::negate(superposition, component)
    }

    /// Remove a component with a specific negation method.
    pub fn negate_with_method(
        &self,
        superposition: &Vector,
        component: &Vector,
        method: NegateMethod,
    ) -> Vector {
        Primitives::negate_with_method(superposition, component, method)
    }

    /// Strengthen a component's presence in a superposition.
    pub fn amplify(&self, superposition: &Vector, component: &Vector, strength: f64) -> Vector {
        Primitives::amplify(superposition, component, strength)
    }

    /// Extract the common pattern from a set of vectors.
    pub fn prototype(&self, vectors: &[&Vector], threshold: f64) -> Vector {
        Primitives::prototype(vectors, threshold)
    }

    /// Compute what changed between two states.
    pub fn difference(&self, before: &Vector, after: &Vector) -> Vector {
        Primitives::difference(before, after)
    }

    /// Weighted interpolation between two vectors.
    pub fn blend(&self, vec1: &Vector, vec2: &Vector, alpha: f64) -> Vector {
        Primitives::blend(vec1, vec2, alpha)
    }

    /// Extract the part of vec that resonates with reference.
    pub fn resonance(&self, vec: &Vector, reference: &Vector) -> Vector {
        Primitives::resonance(vec, reference)
    }

    /// Circular shift (permutation) of vector dimensions.
    pub fn permute(&self, vec: &Vector, k: i32) -> Vector {
        Primitives::permute(vec, k)
    }

    /// Cleanup: find the closest vector in a codebook.
    pub fn cleanup(&self, noisy: &Vector, codebook: &[Vector]) -> Option<(usize, f64)> {
        Primitives::cleanup(noisy, codebook)
    }

    /// Incremental prototype update.
    pub fn prototype_add(&self, prototype: &Vector, example: &Vector, count: usize) -> Vector {
        Primitives::prototype_add(prototype, example, count)
    }

    // =========================================================================
    // Accumulator Operations (Streaming)
    // =========================================================================

    /// Create a new empty accumulator for streaming operations.
    pub fn create_accumulator(&self) -> Accumulator {
        Accumulator::new(self.dimensions)
    }

    /// Add an example to a running accumulator WITHOUT thresholding.
    pub fn accumulate(&self, accumulator: &mut Accumulator, example: &Vector) {
        accumulator.add(example);
    }

    /// Normalize an accumulator for similarity queries.
    pub fn normalize_accumulator(&self, accumulator: &Accumulator) -> Vector {
        accumulator.normalize()
    }

    /// Threshold an accumulator to bipolar {-1, 0, 1}.
    pub fn threshold_accumulator(&self, accumulator: &Accumulator) -> Vector {
        accumulator.threshold()
    }

    // =========================================================================
    // Similarity
    // =========================================================================

    /// Compute similarity between two vectors using cosine similarity.
    pub fn similarity(&self, a: &Vector, b: &Vector) -> f64 {
        Similarity::cosine(a, b)
    }

    /// Compute similarity with a specific metric.
    pub fn similarity_with_metric(&self, a: &Vector, b: &Vector, metric: Metric) -> f64 {
        Similarity::compute(a, b, metric)
    }

    /// Compute Minkowski (Lp) similarity.
    pub fn minkowski_similarity(&self, a: &Vector, b: &Vector, p: f64) -> f64 {
        Similarity::minkowski(a, b, p)
    }

    /// Compute weighted cosine similarity with per-dimension weights.
    pub fn weighted_cosine_similarity(&self, a: &Vector, b: &Vector, weights: &[f64]) -> f64 {
        Similarity::weighted_cosine(a, b, weights)
    }

    /// Compute weighted Euclidean similarity with per-dimension weights.
    pub fn weighted_euclidean_similarity(&self, a: &Vector, b: &Vector, weights: &[f64]) -> f64 {
        Similarity::weighted_euclidean(a, b, weights)
    }

    // =========================================================================
    // Vector Manager Utilities
    // =========================================================================

    /// Get a deterministic position vector for sequence encoding.
    pub fn get_position_vector(&self, position: i64) -> Vector {
        self.vector_manager.get_position_vector(position)
    }

    /// Export the atom codebook for persistence or distribution.
    pub fn export_codebook(&self) -> std::collections::HashMap<String, Vec<i8>> {
        self.vector_manager.export_codebook()
    }

    /// Import a previously exported codebook.
    pub fn import_codebook(&self, codebook: std::collections::HashMap<String, Vec<i8>>) {
        self.vector_manager.import_codebook(codebook)
    }

    // =========================================================================
    // Vector Operations
    // =========================================================================

    /// Keep only the k dimensions with the largest absolute values.
    pub fn sparsify(&self, vec: &Vector, k: usize) -> Vector {
        Primitives::sparsify(vec, k)
    }

    /// Compute the true geometric average (centroid) of vectors.
    pub fn centroid(&self, vectors: &[&Vector]) -> Vector {
        Primitives::centroid(vectors)
    }

    /// Negate every element: +1 -> -1, -1 -> +1, 0 -> 0.
    pub fn flip(&self, vec: &Vector) -> Vector {
        Primitives::flip(vec)
    }

    /// Find the k most similar vectors to a query from candidates.
    pub fn topk_similar(
        &self,
        query: &Vector,
        candidates: &[Vector],
        k: usize,
    ) -> Vec<(usize, f64)> {
        Primitives::topk_similar(query, candidates, k)
    }

    /// Compute all pairwise similarities for a set of vectors.
    pub fn similarity_matrix(&self, vectors: &[Vector]) -> Vec<f64> {
        Primitives::similarity_matrix(vectors)
    }

    /// Information-theoretic entropy of the vector's element distribution.
    pub fn entropy(&self, vec: &Vector) -> f64 {
        Primitives::entropy(vec)
    }

    /// Reduce dimensionality via random projection (Johnson-Lindenstrauss).
    pub fn random_project(&self, vec: &Vector, target_dims: usize, seed: u64) -> Vector {
        Primitives::random_project(vec, target_dims, seed)
    }

    /// Fractional binding: raise a vector to a real-valued power.
    pub fn power(&self, vec: &Vector, exponent: f64) -> Vector {
        Primitives::power(vec, exponent)
    }

    /// Compute similarity of a vector stream with itself at different lags.
    pub fn autocorrelate(&self, stream: &[Vector], max_lag: usize) -> Vec<f64> {
        Primitives::autocorrelate(stream, max_lag)
    }

    /// Compute similarity between two vector streams at different offsets.
    pub fn cross_correlate(
        &self,
        stream_a: &[Vector],
        stream_b: &[Vector],
        max_lag: usize,
    ) -> Vec<f64> {
        Primitives::cross_correlate(stream_a, stream_b, max_lag)
    }

    // =========================================================================
    // Advanced Operations
    // =========================================================================

    /// Orthogonal complement: everything NOT explained by the subspace.
    pub fn reject(&self, vec: &Vector, subspace: &[&Vector], orthogonalize: bool) -> Vector {
        Primitives::reject(vec, subspace, orthogonalize)
    }

    /// Bundle vectors and return per-dimension agreement margins.
    pub fn bundle_with_confidence(&self, vectors: &[&Vector]) -> (Vector, Vec<f64>) {
        Primitives::bundle_with_confidence(vectors)
    }

    /// Mean pairwise cosine similarity (cluster tightness).
    pub fn coherence(&self, vectors: &[Vector]) -> f64 {
        Primitives::coherence(vectors)
    }

    /// Grover's diffusion operator: reflect vector about its mean value.
    pub fn reflect_about_mean(&self, vec: &Vector) -> Vector {
        Primitives::reflect_about_mean(vec)
    }

    /// Quantum-inspired iterative amplitude amplification.
    pub fn grover_amplify(
        &self,
        signal: &Vector,
        background: &Vector,
        iterations: usize,
    ) -> Vector {
        Primitives::grover_amplify(signal, background, iterations)
    }

    /// Temporal derivative of similarity in a vector stream.
    pub fn drift_rate(&self, stream: &[Vector], window: usize) -> Vec<f64> {
        Primitives::drift_rate(stream, window)
    }

    /// Convert cosine similarity to statistical significance (z-score).
    pub fn significance(&self, similarity: f64) -> f64 {
        Similarity::significance(similarity, self.dimensions)
    }

    /// Decode a scalar value from a log-scale encoded vector.
    pub fn decode_scalar_log(&self, vec: &Vector) -> f64 {
        self.scalar_encoder.decode_log(vec, 1e-2, 1e10, 500, 200)
    }

    /// Decode a scalar value from a log-scale encoded vector with custom range.
    pub fn decode_scalar_log_range(&self, vec: &Vector, lo: f64, hi: f64) -> f64 {
        self.scalar_encoder.decode_log(vec, lo, hi, 500, 200)
    }

    /// Estimate how close an accumulator is to saturation.
    pub fn accumulator_capacity(&self, accumulator: &Accumulator, codebook_size: usize) -> f64 {
        accumulator.capacity(codebook_size)
    }

    /// Quantum-inspired purity measure for an accumulator.
    pub fn accumulator_purity(&self, accumulator: &Accumulator) -> f64 {
        accumulator.purity()
    }

    /// Participation ratio: effective number of active dimensions.
    pub fn accumulator_participation_ratio(&self, accumulator: &Accumulator) -> f64 {
        accumulator.participation_ratio()
    }

    // =========================================================================
    // Memory Layer
    // =========================================================================

    /// Create an [`OnlineSubspace`] with this Holon's dimensionality.
    pub fn create_subspace(&self, k: usize) -> OnlineSubspace {
        OnlineSubspace::new(self.dimensions, k)
    }

    /// Create an [`EngramLibrary`] keyed to this Holon's dimensionality.
    pub fn create_engram_library(&self) -> EngramLibrary {
        EngramLibrary::new(self.dimensions)
    }

    // =========================================================================
    // Extended Algebra
    // =========================================================================

    /// Return similarity as a VECTOR, not a scalar.
    pub fn similarity_profile(&self, a: &Vector, b: &Vector) -> Vector {
        Primitives::similarity_profile(a, b)
    }

    /// Weighted resonance — soft attention in VSA algebra.
    pub fn attend(
        &self,
        query: &Vector,
        memory: &Vector,
        strength: f64,
        mode: AttendMode,
    ) -> Vector {
        Primitives::attend(query, memory, strength, mode)
    }

    /// Relational transfer: A is to B as C is to ?
    pub fn analogy(&self, a: &Vector, b: &Vector, c: &Vector) -> Vector {
        Primitives::analogy(a, b, c)
    }

    /// Project vector onto subspace defined by exemplars.
    pub fn project(&self, vec: &Vector, subspace: &[&Vector], orthogonalize: bool) -> Vector {
        Primitives::project(vec, subspace, orthogonalize)
    }

    /// Bind only where condition is met (gated binding).
    pub fn conditional_bind(
        &self,
        a: &Vector,
        b: &Vector,
        gate: &Vector,
        mode: GateMode,
    ) -> Vector {
        Primitives::conditional_bind(a, b, gate, mode)
    }

    /// Measure the "complexity" or "mixedness" of a vector.
    pub fn complexity(&self, vec: &Vector) -> f64 {
        Primitives::complexity(vec)
    }

    /// Reconstruct components from a vector using a codebook.
    pub fn invert(
        &self,
        vec: &Vector,
        codebook: &[Vector],
        top_k: usize,
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        Primitives::invert(vec, codebook, top_k, threshold)
    }

    /// Find structural breakpoints in a vector stream.
    pub fn segment(
        &self,
        stream: &[Vector],
        window: usize,
        threshold: f64,
        method: SegmentMethod,
    ) -> Vec<usize> {
        Primitives::segment(stream, window, threshold, method)
    }
}

impl Default for Holon {
    fn default() -> Self {
        Self::new(4096)
    }
}
