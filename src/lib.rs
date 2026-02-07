//! # Holon: Programmatic Neural Memory
//!
//! Holon is a vector symbolic architecture (VSA) / hyperdimensional computing (HDC)
//! library for building deterministic, explainable AI systems.
//!
//! ## Quick Start
//!
//! ```rust
//! use holon::Holon;
//!
//! // Create a Holon instance
//! let holon = Holon::new(4096);
//!
//! // Encode structured data
//! let billing = holon.encode_json(r#"{"type": "billing", "amount": 100}"#)?;
//! let technical = holon.encode_json(r#"{"type": "technical"}"#)?;
//!
//! // Compute similarity
//! let sim = holon.similarity(&billing, &technical);
//! println!("Similarity: {:.3}", sim);
//!
//! // VSA primitives
//! let combined = holon.bundle(&[&billing, &technical]);
//! let without_billing = holon.negate(&combined, &billing);
//! ```
//!
//! ## Core Concepts
//!
//! - **Vectors**: High-dimensional bipolar vectors ({-1, 0, 1})
//! - **Bind**: Create associations (AND-like)
//! - **Bundle**: Create superpositions (OR-like)
//! - **Negate**: Remove component from superposition (NOT)
//! - **Accumulator**: Streaming operations with frequency preservation

pub mod accumulator;
pub mod encoder;
pub mod error;
pub mod primitives;
pub mod scalar;
pub mod similarity;
pub mod vector;
pub mod vector_manager;

// Re-exports for convenience
pub use accumulator::Accumulator;
pub use encoder::Encoder;
pub use error::{HolonError, Result};
pub use primitives::Primitives;
pub use scalar::{ScalarEncoder, ScalarMode};
pub use similarity::{Metric, Similarity};
pub use vector::Vector;
pub use vector_manager::VectorManager;

/// The main Holon client - primary interface for all operations.
///
/// This struct provides a clean, unified API for:
/// - Data encoding
/// - VSA primitives (bind, bundle, negate, etc.)
/// - Streaming operations (accumulators)
/// - Similarity computation
///
/// # Example
///
/// ```rust
/// use holon::Holon;
///
/// let holon = Holon::new(4096);
///
/// // Encode data
/// let vec = holon.encode_json(r#"{"type": "billing"}"#)?;
///
/// // Create accumulator for streaming
/// let mut accum = holon.create_accumulator();
/// holon.accumulate(&mut accum, &vec);
/// let baseline = holon.normalize_accumulator(&accum);
/// ```
pub struct Holon {
    /// Vector dimensionality
    dimensions: usize,
    /// Vector manager for atom -> vector mapping
    vector_manager: VectorManager,
    /// Encoder for structured data
    encoder: Encoder,
    /// Scalar encoder for continuous values
    scalar_encoder: ScalarEncoder,
}

impl Holon {
    /// Create a new Holon instance with the specified dimensionality.
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality (recommended: 4096 or 8192)
    ///
    /// # Example
    /// ```rust
    /// let holon = Holon::new(4096);
    /// ```
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
    ///
    /// # Example
    /// ```rust
    /// let vec = holon.encode_json(r#"{"type": "billing", "amount": 100}"#)?;
    /// ```
    pub fn encode_json(&self, json: &str) -> Result<Vector> {
        self.encoder.encode_json(json)
    }

    /// Get the base vector for an atomic value.
    ///
    /// Same atom always produces the same vector (deterministic).
    pub fn get_vector(&self, atom: &str) -> Vector {
        self.vector_manager.get_vector(atom)
    }

    /// Encode a continuous scalar value.
    ///
    /// # Arguments
    /// * `value` - The scalar value to encode
    /// * `mode` - Encoding mode (Linear or Circular)
    ///
    /// # Example
    /// ```rust
    /// // Linear encoding - nearby values are similar
    /// let v100 = holon.encode_scalar(100.0, ScalarMode::Linear { scale: 10000.0 });
    /// let v110 = holon.encode_scalar(110.0, ScalarMode::Linear { scale: 10000.0 });
    /// // v100 and v110 have high similarity
    ///
    /// // Circular encoding - values wrap (hour 23 similar to hour 0)
    /// let h23 = holon.encode_scalar(23.0, ScalarMode::Circular { period: 24.0 });
    /// let h0 = holon.encode_scalar(0.0, ScalarMode::Circular { period: 24.0 });
    /// ```
    pub fn encode_scalar(&self, value: f64, mode: ScalarMode) -> Vector {
        self.scalar_encoder.encode(value, mode)
    }

    /// Encode a scalar on log scale.
    ///
    /// Equal ratios have equal similarity:
    /// - 100 → 1000 has same similarity drop as 1000 → 10000
    ///
    /// Perfect for rates, frequencies, and other multiplicative quantities.
    pub fn encode_scalar_log(&self, value: f64) -> Vector {
        self.scalar_encoder.encode_log(value)
    }

    // =========================================================================
    // VSA Primitives
    // =========================================================================

    /// Bind two vectors (AND-like association).
    ///
    /// Creates a vector representing the association between two concepts.
    /// `bind(A, B)` is similar to neither A nor B individually.
    /// `unbind(bind(A, B), A) ≈ B`
    pub fn bind(&self, a: &Vector, b: &Vector) -> Vector {
        Primitives::bind(a, b)
    }

    /// Unbind to retrieve associated value (inverse of bind).
    ///
    /// If `bound = bind(key, value)`, then `unbind(bound, key) ≈ value`.
    pub fn unbind(&self, bound: &Vector, key: &Vector) -> Vector {
        // For bipolar vectors, unbinding is the same as binding (self-inverse)
        Primitives::bind(bound, key)
    }

    /// Bundle multiple vectors (OR-like superposition).
    ///
    /// Creates a vector similar to ALL input vectors.
    pub fn bundle(&self, vectors: &[&Vector]) -> Vector {
        Primitives::bundle(vectors)
    }

    /// Remove a component's influence from a superposition (NOT operation).
    ///
    /// # Example
    /// ```rust
    /// let abc = holon.bundle(&[&a, &b, &c]);
    /// let ac = holon.negate(&abc, &b);
    /// // similarity(ac, b) < 0 (negative similarity)
    /// ```
    pub fn negate(&self, superposition: &Vector, component: &Vector) -> Vector {
        Primitives::negate(superposition, component)
    }

    /// Strengthen a component's presence in a superposition.
    pub fn amplify(&self, superposition: &Vector, component: &Vector, strength: f64) -> Vector {
        Primitives::amplify(superposition, component, strength)
    }

    /// Extract the common pattern from a set of vectors.
    ///
    /// Keeps only dimensions where a majority of vectors agree.
    pub fn prototype(&self, vectors: &[&Vector], threshold: f64) -> Vector {
        Primitives::prototype(vectors, threshold)
    }

    /// Compute what changed between two states.
    ///
    /// Returns a vector highlighting additions (positive) and removals (negative).
    pub fn difference(&self, before: &Vector, after: &Vector) -> Vector {
        Primitives::difference(before, after)
    }

    /// Weighted interpolation between two vectors.
    ///
    /// `alpha = 0.0` returns vec1, `alpha = 1.0` returns vec2.
    pub fn blend(&self, vec1: &Vector, vec2: &Vector, alpha: f64) -> Vector {
        Primitives::blend(vec1, vec2, alpha)
    }

    /// Extract the part of vec that resonates with reference.
    ///
    /// Keeps only dimensions where both vectors agree.
    pub fn resonance(&self, vec: &Vector, reference: &Vector) -> Vector {
        Primitives::resonance(vec, reference)
    }

    /// Circular shift (permutation) of vector dimensions.
    pub fn permute(&self, vec: &Vector, k: i32) -> Vector {
        Primitives::permute(vec, k)
    }

    // =========================================================================
    // Accumulator Operations (Streaming)
    // =========================================================================

    /// Create a new empty accumulator for streaming operations.
    ///
    /// Accumulators preserve frequency information, making them ideal for
    /// anomaly detection where high-frequency patterns should dominate.
    pub fn create_accumulator(&self) -> Accumulator {
        Accumulator::new(self.dimensions)
    }

    /// Add an example to a running accumulator WITHOUT thresholding.
    ///
    /// Patterns seen 99 times contribute 99x more than patterns seen once.
    pub fn accumulate(&self, accumulator: &mut Accumulator, example: &Vector) {
        accumulator.add(example);
    }

    /// Normalize an accumulator for similarity queries.
    ///
    /// Returns a unit-normalized vector suitable for cosine similarity.
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
}

impl Default for Holon {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holon_creation() {
        let holon = Holon::new(4096);
        assert_eq!(holon.dimensions(), 4096);
    }

    #[test]
    fn test_deterministic_vectors() {
        let holon1 = Holon::with_seed(4096, 42);
        let holon2 = Holon::with_seed(4096, 42);

        let v1 = holon1.get_vector("test");
        let v2 = holon2.get_vector("test");

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_bind_unbind() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");

        let ab = holon.bind(&a, &b);
        let b_recovered = holon.unbind(&ab, &a);

        // Recovered should be similar to original
        let sim = holon.similarity(&b_recovered, &b);
        assert!(sim > 0.5, "Expected high similarity, got {}", sim);
    }

    #[test]
    fn test_bundle() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");
        let c = holon.get_vector("C");

        let abc = holon.bundle(&[&a, &b, &c]);

        // Bundle should be similar to all components
        assert!(holon.similarity(&abc, &a) > 0.3);
        assert!(holon.similarity(&abc, &b) > 0.3);
        assert!(holon.similarity(&abc, &c) > 0.3);
    }

    #[test]
    fn test_negate() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");
        let c = holon.get_vector("C");

        let abc = holon.bundle(&[&a, &b, &c]);
        let ac = holon.negate(&abc, &b);

        // After negation, B should have lower similarity
        assert!(holon.similarity(&ac, &b) < holon.similarity(&abc, &b));
    }

    #[test]
    fn test_accumulator() {
        let holon = Holon::new(4096);

        let common = holon.get_vector("common");
        let rare = holon.get_vector("rare");

        let mut accum = holon.create_accumulator();

        // Add common 10 times, rare once
        for _ in 0..10 {
            holon.accumulate(&mut accum, &common);
        }
        holon.accumulate(&mut accum, &rare);

        let baseline = holon.normalize_accumulator(&accum);

        // Common should have higher similarity
        let sim_common = holon.similarity(&common, &baseline);
        let sim_rare = holon.similarity(&rare, &baseline);

        assert!(
            sim_common > sim_rare,
            "Expected common > rare, got {} vs {}",
            sim_common,
            sim_rare
        );
    }
}
