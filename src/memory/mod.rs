//! Memory layer: online subspace learning and engram pattern libraries.
//!
//! This module provides:
//!
//! - [`OnlineSubspace`] — incremental PCA (CCIPCA) that learns the manifold
//!   occupied by "normal" vectors and scores new vectors by their distance
//!   from that manifold (residual = anomaly score).
//!
//! - [`Engram`] — a named, serializable snapshot of a trained subspace,
//!   representing a learned pattern as a compact memory trace.
//!
//! - [`EngramLibrary`] — a collection of engrams with two-tier matching
//!   (eigenvalue pre-filter → full residual scoring).
//!
//! # Usage
//!
//! ```rust
//! use holon::{Encoder, VectorManager};
//! use holon::memory::{OnlineSubspace, EngramLibrary};
//!
//! let vm = VectorManager::new(4096);
//! let enc = Encoder::new(vm);
//!
//! // Learn normal behaviour
//! let mut subspace = OnlineSubspace::new(4096, 32);
//! for _ in 0..200 {
//!     let vec = enc.encode_json(r#"{"type": "normal", "val": 1}"#).unwrap();
//!     subspace.update(&vec.to_f64());
//! }
//!
//! // Score a new vector
//! let probe = enc.encode_json(r#"{"type": "anomaly", "val": 99999}"#).unwrap();
//! let residual = subspace.residual(&probe.to_f64());
//! let is_anomalous = residual > subspace.threshold();
//!
//! // Store in a library
//! let mut library = EngramLibrary::new(4096);
//! library.add("normal_ops", &subspace, None, Default::default());
//! let matches = library.match_vec(&probe.to_f64(), 3, 10);
//! ```

pub mod engram;
pub mod subspace;

pub use engram::{Engram, EngramLibrary};
pub use subspace::OnlineSubspace;
