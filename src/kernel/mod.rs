//! Kernel layer — foundational VSA/HDC primitives.
//!
//! The kernel provides the minimal, stable foundation for all Holon operations:
//! - VSA primitives (`bind`, `bundle`, `unbind`, `negate`, etc.)
//! - Structured data encoding ([`Encoder`])
//! - Deterministic vector management ([`VectorManager`])
//! - Continuous scalar encoding ([`ScalarEncoder`])
//! - Similarity metrics ([`Similarity`])
//! - Streaming accumulators ([`Accumulator`])
//! - Zero-serialization encoding ([`Walkable`] trait)
//!
//! This layer has no dependencies on [`memory`](crate::memory) or
//! [`highlevel`](crate::highlevel).
//!
//! # Example
//!
//! ```rust
//! use holon::kernel::{Encoder, VectorManager, Primitives, Similarity};
//!
//! let vm = VectorManager::new(4096);
//! let enc = Encoder::new(vm.clone());
//!
//! let a = enc.encode_json(r#"{"role": "admin"}"#).unwrap();
//! let b = enc.encode_json(r#"{"role": "user"}"#).unwrap();
//! let sim = Similarity::cosine(&a, &b);
//!
//! let role = vm.get_vector("role");
//! let filler = vm.get_vector("admin");
//! let bound = Primitives::bind(&role, &filler);
//! ```

pub mod accumulator;
pub mod encoder;
pub mod primitives;
pub mod scalar;
pub mod similarity;
pub mod vector;
pub mod vector_manager;
pub mod walkable;

pub use accumulator::Accumulator;
pub use encoder::{Encoder, SequenceMode};
pub use primitives::{AttendMode, GateMode, NegateMethod, Primitives, SegmentMethod};
pub use scalar::{ScalarEncoder, ScalarMode};
pub use similarity::{Metric, Similarity};
pub use vector::Vector;
pub use vector_manager::VectorManager;
pub use walkable::{
    ScalarRef, ScalarValue, TimeResolution, WalkType, Walkable, WalkableRef, WalkableValue,
};
