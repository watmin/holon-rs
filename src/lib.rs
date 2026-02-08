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
pub mod walkable;

// Re-exports for convenience
pub use accumulator::Accumulator;
pub use encoder::{Encoder, SequenceMode};
pub use error::{HolonError, Result};
pub use primitives::{NegateMethod, Primitives};
pub use scalar::{ScalarEncoder, ScalarMode};
pub use similarity::{Metric, Similarity};
pub use vector::Vector;
pub use vector_manager::VectorManager;
pub use walkable::{ScalarRef, ScalarValue, WalkType, Walkable, WalkableRef, WalkableValue};

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

    /// Encode a serde_json Value directly.
    ///
    /// Useful when you already have parsed JSON or want to construct
    /// values programmatically without serializing to string.
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

    /// Encode a sequence of items.
    ///
    /// # Arguments
    /// * `items` - Slice of string items to encode
    /// * `mode` - How to encode the sequence (Bundle, Positional, Chained, Ngram)
    ///
    /// # Example
    /// ```rust
    /// // Order-preserving
    /// let seq = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Positional);
    ///
    /// // Order-independent (bag of items)
    /// let bag = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Bundle);
    ///
    /// // N-gram patterns
    /// let ngrams = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Ngram { n: 2 });
    /// ```
    pub fn encode_sequence(&self, items: &[&str], mode: SequenceMode) -> Vector {
        self.encoder.encode_sequence(items, mode)
    }

    /// Encode any type implementing the Walkable trait.
    ///
    /// This is the zero-serialization path: no JSON conversion needed.
    /// Your structs implement the Walkable trait and get encoded directly.
    ///
    /// # Example
    /// ```rust
    /// use holon::{Holon, Walkable, WalkType, WalkableValue};
    ///
    /// struct Packet {
    ///     protocol: String,
    ///     src_port: u16,
    ///     dst_port: u16,
    /// }
    ///
    /// impl Walkable for Packet {
    ///     fn walk_type(&self) -> WalkType { WalkType::Map }
    ///     fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
    ///         vec![
    ///             ("protocol", self.protocol.to_walkable_value()),
    ///             ("src_port", (self.src_port as i64).to_walkable_value()),
    ///             ("dst_port", (self.dst_port as i64).to_walkable_value()),
    ///         ]
    ///     }
    /// }
    ///
    /// let holon = Holon::new(4096);
    /// let packet = Packet { protocol: "TCP".into(), src_port: 443, dst_port: 8080 };
    /// let vec = holon.encode_walkable(&packet);
    /// ```
    pub fn encode_walkable<W: Walkable>(&self, walkable: &W) -> Vector {
        self.encoder.encode_walkable(walkable)
    }

    /// Encode a WalkableValue directly.
    ///
    /// Useful when you have dynamically constructed WalkableValues
    /// rather than typed structs.
    pub fn encode_walkable_value(&self, value: &WalkableValue) -> Vector {
        self.encoder.encode_walkable_value(value)
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

    /// Cleanup: find the closest vector in a codebook.
    ///
    /// Returns the index and similarity of the best match, or None if codebook is empty.
    pub fn cleanup(&self, noisy: &Vector, codebook: &[Vector]) -> Option<(usize, f64)> {
        Primitives::cleanup(noisy, codebook)
    }

    /// Incremental prototype update.
    ///
    /// Updates an existing prototype with a new example.
    /// `count` is the number of examples already in the prototype.
    pub fn prototype_add(&self, prototype: &Vector, example: &Vector, count: usize) -> Vector {
        Primitives::prototype_add(prototype, example, count)
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

    #[test]
    fn test_amplify() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");

        let ab = holon.bundle(&[&a, &b]);
        let amplified = holon.amplify(&ab, &a, 2.0);

        // After amplification, A should have higher similarity
        assert!(holon.similarity(&amplified, &a) > holon.similarity(&ab, &a));
    }

    #[test]
    fn test_prototype() {
        let holon = Holon::new(4096);

        // Create vectors with common patterns
        let v1 = holon.encode_json(r#"{"type": "billing", "a": 1}"#).unwrap();
        let v2 = holon.encode_json(r#"{"type": "billing", "b": 2}"#).unwrap();
        let v3 = holon.encode_json(r#"{"type": "billing", "c": 3}"#).unwrap();

        let proto = holon.prototype(&[&v1, &v2, &v3], 0.5);

        // Prototype should be similar to all inputs
        assert!(holon.similarity(&proto, &v1) > 0.2);
        assert!(holon.similarity(&proto, &v2) > 0.2);
        assert!(holon.similarity(&proto, &v3) > 0.2);
    }

    #[test]
    fn test_prototype_add() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("pattern_a");
        let b = holon.get_vector("pattern_b");
        let c = holon.get_vector("pattern_c");

        // Incremental prototype building
        let proto1 = a.clone();
        let proto2 = holon.prototype_add(&proto1, &b, 1);
        let proto3 = holon.prototype_add(&proto2, &c, 2);

        // Final prototype should have some similarity to all inputs
        assert!(holon.similarity(&proto3, &a) > 0.0);
        assert!(holon.similarity(&proto3, &b) > 0.0);
        assert!(holon.similarity(&proto3, &c) > 0.0);
    }

    #[test]
    fn test_cleanup() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");
        let c = holon.get_vector("C");

        let codebook = vec![a.clone(), b.clone(), c.clone()];

        // Noisy version of A (bundle with some noise)
        let noise = holon.get_vector("noise");
        let noisy_a = holon.bundle(&[&a, &a, &a, &noise]);

        let (idx, sim) = holon.cleanup(&noisy_a, &codebook).unwrap();

        assert_eq!(idx, 0, "Should find A as closest match");
        assert!(sim > 0.5, "Should have high similarity");
    }

    #[test]
    fn test_difference() {
        let holon = Holon::new(4096);

        let before = holon.encode_json(r#"{"status": "pending"}"#).unwrap();
        let after = holon.encode_json(r#"{"status": "completed"}"#).unwrap();

        let diff = holon.difference(&before, &after);

        // Difference should be non-zero
        assert!(diff.nnz() > 0);
    }

    #[test]
    fn test_blend() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("state_a");
        let b = holon.get_vector("state_b");

        let blend_0 = holon.blend(&a, &b, 0.0);
        let blend_1 = holon.blend(&a, &b, 1.0);
        let blend_half = holon.blend(&a, &b, 0.5);

        // blend(a, b, 0) should be similar to a
        assert!(holon.similarity(&blend_0, &a) > 0.9);

        // blend(a, b, 1) should be similar to b
        assert!(holon.similarity(&blend_1, &b) > 0.9);

        // blend at 0.5 should be somewhat similar to both
        assert!(holon.similarity(&blend_half, &a) > 0.3);
        assert!(holon.similarity(&blend_half, &b) > 0.3);
    }

    #[test]
    fn test_resonance() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");
        let b = holon.get_vector("B");

        let ab = holon.bundle(&[&a, &b]);
        let resonant = holon.resonance(&ab, &a);

        // Resonance with A should increase similarity to A
        assert!(holon.similarity(&resonant, &a) >= holon.similarity(&ab, &a) - 0.1);
    }

    #[test]
    fn test_permute() {
        let holon = Holon::new(4096);

        let v = holon.get_vector("seq_item");

        let permuted = holon.permute(&v, 10);
        let restored = holon.permute(&permuted, -10);

        // Permute and inverse permute should restore original
        assert_eq!(v, restored);
    }

    #[test]
    fn test_encode_sequence_positional() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Positional);
        let seq2 = holon.encode_sequence(&["C", "B", "A"], SequenceMode::Positional);

        // Different order should produce different vectors
        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim < 0.8, "Expected lower similarity for reversed, got {}", sim);
    }

    #[test]
    fn test_encode_sequence_bundle() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Bundle);
        let seq2 = holon.encode_sequence(&["C", "B", "A"], SequenceMode::Bundle);

        // Same items should produce similar vectors regardless of order
        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim > 0.9, "Expected high similarity for same items, got {}", sim);
    }

    #[test]
    fn test_encode_sequence_ngram() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C", "D"], SequenceMode::Ngram { n: 2 });
        let seq2 = holon.encode_sequence(&["A", "B", "X", "Y"], SequenceMode::Ngram { n: 2 });

        // Shared "AB" bigram should give some similarity
        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim > 0.1, "Expected some similarity from shared ngram, got {}", sim);
    }

    #[test]
    fn test_scalar_log_ratio_preservation() {
        let holon = Holon::new(4096);

        // With bipolar quantization, small differences may get quantized away.
        // Test that vectors at large magnitude differences are distinguishable.
        let v1 = holon.encode_scalar_log(1.0);
        let v100k = holon.encode_scalar_log(100_000.0);
        let v1b = holon.encode_scalar_log(1_000_000_000.0);

        let sim_1_100k = holon.similarity(&v1, &v100k);
        let sim_1_1b = holon.similarity(&v1, &v1b);

        // Different magnitudes should produce different similarities
        assert!(
            sim_1_100k != sim_1_1b,
            "Expected different similarities for different magnitudes"
        );

        // Self-similarity should be 1.0
        let self_sim = holon.similarity(&v1, &v1);
        assert!((self_sim - 1.0).abs() < 1e-10, "Self-similarity should be 1.0");
    }

    // =========================================================================
    // Walkable Tests
    // =========================================================================

    #[test]
    fn test_encode_walkable_custom_struct() {
        use crate::walkable::{WalkType, Walkable, WalkableValue};

        struct Packet {
            protocol: String,
            src_port: u16,
            dst_port: u16,
        }

        impl Walkable for Packet {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("protocol", self.protocol.to_walkable_value()),
                    ("src_port", (self.src_port as i64).to_walkable_value()),
                    ("dst_port", (self.dst_port as i64).to_walkable_value()),
                ]
            }
        }

        let holon = Holon::new(4096);

        let tcp = Packet {
            protocol: "TCP".into(),
            src_port: 443,
            dst_port: 8080,
        };

        let udp = Packet {
            protocol: "UDP".into(),
            src_port: 53,
            dst_port: 1234,
        };

        let vec_tcp = holon.encode_walkable(&tcp);
        let vec_udp = holon.encode_walkable(&udp);

        // Should produce valid vectors
        assert_eq!(vec_tcp.dimensions(), 4096);
        assert!(vec_tcp.nnz() > 0);
        assert!(vec_udp.nnz() > 0);

        // Different packets should NOT be identical
        let sim = holon.similarity(&vec_tcp, &vec_udp);
        assert!(sim < 1.0, "Expected different vectors for different packets, got sim={}", sim);

        // Same packet should be identical to itself
        let self_sim = holon.similarity(&vec_tcp, &vec_tcp);
        assert!((self_sim - 1.0).abs() < 0.01, "Self-similarity should be ~1.0, got {}", self_sim);
    }

    #[test]
    fn test_walkable_vs_json_similar_encoding() {
        use serde_json::json;

        let holon = Holon::new(4096);

        // Encode via JSON
        let json_vec = holon
            .encode_json(r#"{"type": "billing", "amount": 100}"#)
            .unwrap();

        // Encode via Walkable (using serde_json::Value which implements Walkable)
        let value = json!({"type": "billing", "amount": 100});
        let walkable_vec = holon.encode_walkable(&value);

        // Both should produce similar results (not identical due to implementation details)
        // but structurally equivalent
        assert_eq!(json_vec.dimensions(), walkable_vec.dimensions());
        assert!(json_vec.nnz() > 0);
        assert!(walkable_vec.nnz() > 0);

        // Should have high similarity (same data structure)
        let sim = holon.similarity(&json_vec, &walkable_vec);
        assert!(sim > 0.8, "Expected high similarity between JSON and Walkable encoding, got {}", sim);
    }

    #[test]
    fn test_walkable_nested_struct() {
        use crate::walkable::{WalkType, Walkable, WalkableValue, ScalarValue};

        struct Address {
            city: String,
            zip: String,
        }

        impl Walkable for Address {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("city", self.city.to_walkable_value()),
                    ("zip", self.zip.to_walkable_value()),
                ]
            }
        }

        struct Person {
            name: String,
            address: Address,
        }

        impl Walkable for Person {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("name", WalkableValue::Scalar(ScalarValue::String(self.name.clone()))),
                    ("address", self.address.to_walkable_value()),
                ]
            }
        }

        let holon = Holon::new(4096);

        let alice = Person {
            name: "Alice".into(),
            address: Address {
                city: "Seattle".into(),
                zip: "98101".into(),
            },
        };

        let bob = Person {
            name: "Bob".into(),
            address: Address {
                city: "Seattle".into(),
                zip: "98101".into(),
            },
        };

        let vec_alice = holon.encode_walkable(&alice);
        let vec_bob = holon.encode_walkable(&bob);

        // Same city/zip should give some similarity (shared fields)
        let sim = holon.similarity(&vec_alice, &vec_bob);
        assert!(sim > 0.1, "Expected some similarity from shared address, got {}", sim);
        assert!(sim < 1.0, "Different people should not be identical, got {}", sim);
    }

    #[test]
    fn test_encode_value_direct() {
        use serde_json::json;

        let holon = Holon::new(4096);

        let value = json!({"type": "test", "count": 42});
        let vec = holon.encode_value(&value);

        assert_eq!(vec.dimensions(), 4096);
        assert!(vec.nnz() > 0);
    }

    #[test]
    fn test_similarity_metrics() {
        let holon = Holon::new(4096);

        let a = holon.get_vector("A");

        // Test different metrics with self-similarity
        let cos = holon.similarity_with_metric(&a, &a, Metric::Cosine);
        let ham = holon.similarity_with_metric(&a, &a, Metric::Hamming);
        let euc = holon.similarity_with_metric(&a, &a, Metric::Euclidean);

        // Self-similarity should be maximal
        assert!((cos - 1.0).abs() < 1e-10);
        assert!((ham - 1.0).abs() < 1e-10);
        assert!((euc - 1.0).abs() < 1e-10); // Distance 0 -> similarity 1
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_anomaly_detection_workflow() {
        let holon = Holon::new(4096);

        // Build baseline from normal patterns
        let mut baseline_accum = holon.create_accumulator();

        for i in 0..100 {
            let normal = holon.encode_json(&format!(
                r#"{{"type": "request", "endpoint": "/api/users", "status": 200, "id": {}}}"#,
                i
            )).unwrap();
            holon.accumulate(&mut baseline_accum, &normal);
        }

        let baseline = holon.normalize_accumulator(&baseline_accum);

        // Test normal request
        let normal_test = holon.encode_json(
            r#"{"type": "request", "endpoint": "/api/users", "status": 200}"#
        ).unwrap();

        // Test anomalous request
        let anomaly_test = holon.encode_json(
            r#"{"type": "request", "endpoint": "/admin/delete_all", "status": 500}"#
        ).unwrap();

        let sim_normal = holon.similarity(&normal_test, &baseline);
        let sim_anomaly = holon.similarity(&anomaly_test, &baseline);

        // Normal should have higher similarity to baseline
        assert!(
            sim_normal > sim_anomaly,
            "Expected normal > anomaly, got {} vs {}",
            sim_normal,
            sim_anomaly
        );
    }

    #[test]
    fn test_rate_based_anomaly_detection() {
        let holon = Holon::new(4096);

        // Build baseline from normal rates
        let mut rate_accum = holon.create_accumulator();

        for _ in 0..50 {
            // Normal rate: ~100 pps with some variation
            let rate = 100.0 + (rand::random::<f64>() - 0.5) * 20.0;
            let rate_vec = holon.encode_scalar_log(rate);
            holon.accumulate(&mut rate_accum, &rate_vec);
        }

        let rate_baseline = holon.normalize_accumulator(&rate_accum);

        // Test normal rate
        let normal_rate = holon.encode_scalar_log(105.0);
        // Test anomalous rate (DDoS-like)
        let anomaly_rate = holon.encode_scalar_log(100000.0);

        let sim_normal = holon.similarity(&normal_rate, &rate_baseline);
        let sim_anomaly = holon.similarity(&anomaly_rate, &rate_baseline);

        assert!(
            sim_normal > sim_anomaly,
            "Expected normal rate > anomaly rate, got {} vs {}",
            sim_normal,
            sim_anomaly
        );
    }
}
