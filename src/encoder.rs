//! Encoder: Structured data to vectors.
//!
//! The encoder converts structured data (JSON, key-value pairs) into
//! high-dimensional vectors while preserving semantic relationships.
//!
//! # Role-Filler Binding
//!
//! The key insight is **role-filler binding**: we bind each value (filler)
//! with its key (role) before bundling. This preserves structure:
//!
//! ```json
//! {"src_port": 53, "dst_port": 80}
//! ```
//!
//! Without role-filler binding, the "53" and "80" would be indistinguishable
//! as just numbers. With binding:
//! - `bind(role["src_port"], value["53"])` is different from
//! - `bind(role["dst_port"], value["53"])`

use crate::error::{HolonError, Result};
use crate::primitives::Primitives;
use crate::vector::Vector;
use crate::vector_manager::VectorManager;
use serde_json::Value;

/// Encoder for converting structured data to vectors.
#[derive(Clone)]
pub struct Encoder {
    vector_manager: VectorManager,
}

impl Encoder {
    /// Create a new encoder with the given vector manager.
    pub fn new(vector_manager: VectorManager) -> Self {
        Self { vector_manager }
    }

    /// Get the dimensionality.
    pub fn dimensions(&self) -> usize {
        self.vector_manager.dimensions()
    }

    /// Encode a JSON string into a vector.
    ///
    /// # Example
    /// ```rust
    /// let vec = encoder.encode_json(r#"{"type": "billing", "amount": 100}"#)?;
    /// ```
    pub fn encode_json(&self, json: &str) -> Result<Vector> {
        let value: Value = serde_json::from_str(json)?;
        Ok(self.encode_value(&value, None))
    }

    /// Encode a serde_json Value into a vector.
    pub fn encode_value(&self, value: &Value, prefix: Option<&str>) -> Vector {
        match value {
            Value::Null => self.encode_atom(&Self::make_path(prefix, "null")),
            Value::Bool(b) => self.encode_atom(&Self::make_path(prefix, &b.to_string())),
            Value::Number(n) => self.encode_atom(&Self::make_path(prefix, &n.to_string())),
            Value::String(s) => self.encode_atom(&Self::make_path(prefix, s)),
            Value::Array(arr) => self.encode_array(arr, prefix),
            Value::Object(obj) => self.encode_object(obj, prefix),
        }
    }

    fn make_path(prefix: Option<&str>, key: &str) -> String {
        match prefix {
            Some(p) => format!("{}.{}", p, key),
            None => key.to_string(),
        }
    }

    /// Encode an atomic value (string, number, etc.).
    fn encode_atom(&self, atom: &str) -> Vector {
        self.vector_manager.get_vector(atom)
    }

    /// Encode an array.
    ///
    /// Uses position-encoding for ordered arrays.
    fn encode_array(&self, arr: &[Value], prefix: Option<&str>) -> Vector {
        if arr.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "[]"));
        }

        let mut vectors: Vec<Vector> = Vec::new();

        for (i, item) in arr.iter().enumerate() {
            let pos_prefix = Self::make_path(prefix, &format!("[{}]", i));
            let item_vec = self.encode_value(item, Some(&pos_prefix));

            // Bind with position marker
            let pos_vec = self.encode_atom(&pos_prefix);
            let bound = Primitives::bind(&pos_vec, &item_vec);
            vectors.push(bound);
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        Primitives::bundle(&refs)
    }

    /// Encode an object using role-filler binding.
    fn encode_object(&self, obj: &serde_json::Map<String, Value>, prefix: Option<&str>) -> Vector {
        if obj.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "{}"));
        }

        let mut vectors: Vec<Vector> = Vec::new();

        for (key, value) in obj {
            let key_path = Self::make_path(prefix, key);

            // Role vector (the key)
            let role_vec = self.encode_atom(&key_path);

            // Filler vector (the value, with key as prefix for nested structure)
            let filler_vec = self.encode_value(value, Some(&key_path));

            // Role-filler binding: bind the key with its value
            let bound = Primitives::bind(&role_vec, &filler_vec);
            vectors.push(bound);
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        Primitives::bundle(&refs)
    }

    /// Encode a sequence of items with different modes.
    pub fn encode_sequence(&self, items: &[&str], mode: SequenceMode) -> Vector {
        if items.is_empty() {
            return Vector::zeros(self.dimensions());
        }

        match mode {
            SequenceMode::Bundle => {
                // Unordered: just bundle all items
                let vectors: Vec<Vector> = items.iter().map(|&s| self.encode_atom(s)).collect();
                let refs: Vec<&Vector> = vectors.iter().collect();
                Primitives::bundle(&refs)
            }
            SequenceMode::Positional => {
                // Ordered: bind each item with its position
                let mut vectors: Vec<Vector> = Vec::new();
                for (i, &item) in items.iter().enumerate() {
                    let item_vec = self.encode_atom(item);
                    let positioned = Primitives::permute(&item_vec, i as i32);
                    vectors.push(positioned);
                }
                let refs: Vec<&Vector> = vectors.iter().collect();
                Primitives::bundle(&refs)
            }
            SequenceMode::Chained => {
                // Chain: each element binds with the previous result
                let mut result = self.encode_atom(items[0]);
                for &item in items.iter().skip(1) {
                    let item_vec = self.encode_atom(item);
                    result = Primitives::bind(&result, &item_vec);
                }
                result
            }
            SequenceMode::Ngram { n } => {
                // N-grams: bundle all n-grams
                if items.len() < n {
                    // Not enough items for n-gram, fall back to bundle
                    let vectors: Vec<Vector> = items.iter().map(|&s| self.encode_atom(s)).collect();
                    let refs: Vec<&Vector> = vectors.iter().collect();
                    return Primitives::bundle(&refs);
                }

                let mut ngram_vecs: Vec<Vector> = Vec::new();
                for window in items.windows(n) {
                    // Create n-gram by binding items with position permutation
                    let mut ngram = self.encode_atom(window[0]);
                    for (j, &item) in window.iter().enumerate().skip(1) {
                        let item_vec = self.encode_atom(item);
                        let permuted = Primitives::permute(&item_vec, j as i32);
                        ngram = Primitives::bind(&ngram, &permuted);
                    }
                    ngram_vecs.push(ngram);
                }

                let refs: Vec<&Vector> = ngram_vecs.iter().collect();
                Primitives::bundle(&refs)
            }
        }
    }
}

/// Modes for encoding sequences.
#[derive(Clone, Copy, Debug)]
pub enum SequenceMode {
    /// Unordered bundle (order doesn't matter)
    Bundle,
    /// Position-encoded using permutation
    Positional,
    /// Chain-bound (each element depends on previous)
    Chained,
    /// N-gram encoding (captures local patterns)
    Ngram { n: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::Similarity;

    #[test]
    fn test_encode_json() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let vec = encoder
            .encode_json(r#"{"type": "billing"}"#)
            .expect("Failed to encode JSON");

        assert_eq!(vec.dimensions(), 4096);
    }

    #[test]
    fn test_role_filler_binding() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let src_53 = encoder
            .encode_json(r#"{"src_port": 53}"#)
            .expect("parse error");
        let dst_53 = encoder
            .encode_json(r#"{"dst_port": 53}"#)
            .expect("parse error");

        // These should be different because of role-filler binding
        let sim = Similarity::cosine(&src_53, &dst_53);
        assert!(
            sim < 0.5,
            "Expected low similarity for different roles, got {}",
            sim
        );
    }

    #[test]
    fn test_similar_structure() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let billing1 = encoder
            .encode_json(r#"{"type": "billing", "amount": 100}"#)
            .expect("parse error");
        let billing2 = encoder
            .encode_json(r#"{"type": "billing", "amount": 200}"#)
            .expect("parse error");
        let technical = encoder
            .encode_json(r#"{"type": "technical"}"#)
            .expect("parse error");

        // Two billing records should be more similar to each other than to technical
        let sim_billing = Similarity::cosine(&billing1, &billing2);
        let sim_cross = Similarity::cosine(&billing1, &technical);

        assert!(
            sim_billing > sim_cross,
            "Expected billing1↔billing2 > billing1↔technical, got {} vs {}",
            sim_billing,
            sim_cross
        );
    }

    #[test]
    fn test_sequence_positional() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let seq1 = encoder.encode_sequence(&["A", "B", "C"], SequenceMode::Positional);
        let seq2 = encoder.encode_sequence(&["C", "B", "A"], SequenceMode::Positional);

        // Order matters for positional encoding
        let sim = Similarity::cosine(&seq1, &seq2);
        assert!(
            sim < 0.8,
            "Expected lower similarity for reversed sequence, got {}",
            sim
        );
    }

    #[test]
    fn test_sequence_bundle() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let seq1 = encoder.encode_sequence(&["A", "B", "C"], SequenceMode::Bundle);
        let seq2 = encoder.encode_sequence(&["C", "B", "A"], SequenceMode::Bundle);

        // Order doesn't matter for bundle
        let sim = Similarity::cosine(&seq1, &seq2);
        assert!(
            sim > 0.9,
            "Expected high similarity for same items in different order, got {}",
            sim
        );
    }
}
