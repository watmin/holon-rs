//! # Holon: Programmatic Neural Memory
//!
//! Holon is a vector symbolic architecture (VSA) / hyperdimensional computing (HDC)
//! library for building deterministic, explainable AI systems.
//!
//! ## Three-layer architecture
//!
//! | Layer | Purpose |
//! |-------|---------|
//! | [`kernel`] | VSA primitives, encoding, similarity, accumulators, walkable trait |
//! | [`memory`] | Online subspace learning (CCIPCA), engram storage and recall |
//! | [`highlevel`] | [`Holon`] convenience wrapper that owns an encoder and delegates |
//!
//! ## Usage
//!
//! ```rust
//! // Direct kernel + memory imports (recommended for library / production code)
//! use holon::kernel::{Encoder, VectorManager, Primitives, Similarity};
//! use holon::memory::OnlineSubspace;
//!
//! let vm = VectorManager::new(4096);
//! let enc = Encoder::new(vm);
//!
//! let a = enc.encode_json(r#"{"role": "admin"}"#).unwrap();
//! let b = enc.encode_json(r#"{"role": "user"}"#).unwrap();
//! let sim = Similarity::cosine(&a, &b);
//! let bound = Primitives::bind(&a, &b);
//! ```
//!
//! ```rust
//! // Crate-level re-exports (less typing, same types)
//! use holon::{Encoder, VectorManager, OnlineSubspace};
//! ```
//!
//! ```rust
//! // Holon facade (most ergonomic for quick scripts)
//! use holon::Holon;
//!
//! let holon = Holon::new(4096);
//! let vec = holon.encode_json(r#"{"key": "value"}"#).unwrap();
//! let sim = holon.similarity(&vec, &vec);
//! ```

pub mod error;
pub mod highlevel;
pub mod kernel;
pub mod memory;

// ---------------------------------------------------------------------------
// Backward-compatible module re-exports.
//
// These let `use holon::primitives::Primitives` keep working even though
// the file now lives at `src/kernel/primitives.rs`.
// ---------------------------------------------------------------------------
pub use kernel::accumulator;
pub use kernel::encoder;
pub use kernel::primitives;
pub use kernel::scalar;
pub use kernel::similarity;
pub use kernel::vector;
pub use kernel::vector_manager;
pub use kernel::walkable;

// ---------------------------------------------------------------------------
// Backward-compatible type re-exports.
// ---------------------------------------------------------------------------
pub use error::{HolonError, Result};
pub use highlevel::Holon;
pub use kernel::{
    Accumulator, AttendMode, Encoder, GateMode, Metric, NegateMethod, Primitives, ScalarEncoder,
    ScalarMode, SegmentMethod, SequenceMode, Similarity, Vector, VectorManager,
};
pub use kernel::{ScalarRef, ScalarValue, TimeResolution, WalkType, Walkable, WalkableRef, WalkableValue};
pub use memory::{Engram, EngramLibrary, OnlineSubspace};

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

        assert!(holon.similarity(&ac, &b) < holon.similarity(&abc, &b));
    }

    #[test]
    fn test_accumulator() {
        let holon = Holon::new(4096);

        let common = holon.get_vector("common");
        let rare = holon.get_vector("rare");

        let mut accum = holon.create_accumulator();

        for _ in 0..10 {
            holon.accumulate(&mut accum, &common);
        }
        holon.accumulate(&mut accum, &rare);

        let baseline = holon.normalize_accumulator(&accum);

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

        assert!(holon.similarity(&amplified, &a) > holon.similarity(&ab, &a));
    }

    #[test]
    fn test_prototype() {
        let holon = Holon::new(4096);

        let v1 = holon.encode_json(r#"{"type": "billing", "a": 1}"#).unwrap();
        let v2 = holon.encode_json(r#"{"type": "billing", "b": 2}"#).unwrap();
        let v3 = holon.encode_json(r#"{"type": "billing", "c": 3}"#).unwrap();

        let proto = holon.prototype(&[&v1, &v2, &v3], 0.5);

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

        let proto1 = a.clone();
        let proto2 = holon.prototype_add(&proto1, &b, 1);
        let proto3 = holon.prototype_add(&proto2, &c, 2);

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

        assert!(holon.similarity(&blend_0, &a) > 0.9);
        assert!(holon.similarity(&blend_1, &b) > 0.9);
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

        assert!(holon.similarity(&resonant, &a) >= holon.similarity(&ab, &a) - 0.1);
    }

    #[test]
    fn test_permute() {
        let holon = Holon::new(4096);

        let v = holon.get_vector("seq_item");

        let permuted = holon.permute(&v, 10);
        let restored = holon.permute(&permuted, -10);

        assert_eq!(v, restored);
    }

    #[test]
    fn test_encode_sequence_positional() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Positional);
        let seq2 = holon.encode_sequence(&["C", "B", "A"], SequenceMode::Positional);

        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim < 0.8, "Expected lower similarity for reversed, got {}", sim);
    }

    #[test]
    fn test_encode_sequence_bundle() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C"], SequenceMode::Bundle);
        let seq2 = holon.encode_sequence(&["C", "B", "A"], SequenceMode::Bundle);

        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim > 0.9, "Expected high similarity for same items, got {}", sim);
    }

    #[test]
    fn test_encode_sequence_ngram() {
        let holon = Holon::new(4096);

        let seq1 = holon.encode_sequence(&["A", "B", "C", "D"], SequenceMode::Ngram { n: 2 });
        let seq2 = holon.encode_sequence(&["A", "B", "X", "Y"], SequenceMode::Ngram { n: 2 });

        let sim = holon.similarity(&seq1, &seq2);
        assert!(sim > 0.1, "Expected some similarity from shared ngram, got {}", sim);
    }

    #[test]
    fn test_scalar_log_ratio_preservation() {
        let holon = Holon::new(4096);

        let v1 = holon.encode_scalar_log(1.0);
        let v100k = holon.encode_scalar_log(100_000.0);
        let v1b = holon.encode_scalar_log(1_000_000_000.0);

        let sim_1_100k = holon.similarity(&v1, &v100k);
        let sim_1_1b = holon.similarity(&v1, &v1b);

        assert!(
            sim_1_100k != sim_1_1b,
            "Expected different similarities for different magnitudes"
        );

        let self_sim = holon.similarity(&v1, &v1);
        assert!((self_sim - 1.0).abs() < 1e-10, "Self-similarity should be 1.0");
    }

    // =========================================================================
    // Walkable Tests
    // =========================================================================

    #[test]
    fn test_encode_walkable_custom_struct() {
        use crate::kernel::walkable::{WalkType, Walkable, WalkableValue};

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

        assert_eq!(vec_tcp.dimensions(), 4096);
        assert!(vec_tcp.nnz() > 0);
        assert!(vec_udp.nnz() > 0);

        let sim = holon.similarity(&vec_tcp, &vec_udp);
        assert!(sim < 1.0, "Expected different vectors for different packets, got sim={}", sim);

        let self_sim = holon.similarity(&vec_tcp, &vec_tcp);
        assert!((self_sim - 1.0).abs() < 0.01, "Self-similarity should be ~1.0, got {}", self_sim);
    }

    #[test]
    fn test_walkable_vs_json_similar_encoding() {
        use serde_json::json;

        let holon = Holon::new(4096);

        let json_vec = holon
            .encode_json(r#"{"type": "billing", "amount": 100}"#)
            .unwrap();

        let value = json!({"type": "billing", "amount": 100});
        let walkable_vec = holon.encode_walkable(&value);

        assert_eq!(json_vec.dimensions(), walkable_vec.dimensions());
        assert!(json_vec.nnz() > 0);
        assert!(walkable_vec.nnz() > 0);

        let sim = holon.similarity(&json_vec, &walkable_vec);
        assert!(sim > 0.8, "Expected high similarity between JSON and Walkable encoding, got {}", sim);
    }

    #[test]
    fn test_walkable_nested_struct() {
        use crate::kernel::walkable::{WalkType, Walkable, WalkableValue, ScalarValue};

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

        let cos = holon.similarity_with_metric(&a, &a, Metric::Cosine);
        let ham = holon.similarity_with_metric(&a, &a, Metric::Hamming);
        let euc = holon.similarity_with_metric(&a, &a, Metric::Euclidean);

        assert!((cos - 1.0).abs() < 1e-10);
        assert!((ham - 1.0).abs() < 1e-10);
        assert!((euc - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_anomaly_detection_workflow() {
        let holon = Holon::new(4096);

        let mut baseline_accum = holon.create_accumulator();

        for i in 0..100 {
            let normal = holon.encode_json(&format!(
                r#"{{"type": "request", "endpoint": "/api/users", "status": 200, "id": {}}}"#,
                i
            )).unwrap();
            holon.accumulate(&mut baseline_accum, &normal);
        }

        let baseline = holon.normalize_accumulator(&baseline_accum);

        let normal_test = holon.encode_json(
            r#"{"type": "request", "endpoint": "/api/users", "status": 200}"#
        ).unwrap();

        let anomaly_test = holon.encode_json(
            r#"{"type": "request", "endpoint": "/admin/delete_all", "status": 500}"#
        ).unwrap();

        let sim_normal = holon.similarity(&normal_test, &baseline);
        let sim_anomaly = holon.similarity(&anomaly_test, &baseline);

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

        let mut rate_accum = holon.create_accumulator();

        for _ in 0..50 {
            let rate = 100.0 + (rand::random::<f64>() - 0.5) * 20.0;
            let rate_vec = holon.encode_scalar_log(rate);
            holon.accumulate(&mut rate_accum, &rate_vec);
        }

        let rate_baseline = holon.normalize_accumulator(&rate_accum);

        let normal_rate = holon.encode_scalar_log(105.0);
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

    #[test]
    fn test_kernel_direct_imports() {
        use crate::kernel::{Encoder, VectorManager, Primitives, Similarity};

        let vm = VectorManager::new(4096);
        let enc = Encoder::new(vm);

        let a = enc.encode_json(r#"{"role": "admin"}"#).unwrap();
        let b = enc.encode_json(r#"{"role": "user"}"#).unwrap();

        let sim = Similarity::cosine(&a, &b);
        assert!(sim < 1.0);

        let bound = Primitives::bind(&a, &b);
        assert!(bound.nnz() > 0);
    }
}
