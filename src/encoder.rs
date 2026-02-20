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

use crate::error::Result;
use crate::primitives::Primitives;
use crate::vector::Vector;
use crate::vector_manager::VectorManager;
use crate::walkable::{
    ScalarRef, ScalarValue, TimeResolution, WalkType, Walkable, WalkableRef, WalkableValue,
};
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
    /// use holon::Holon;
    /// let holon = Holon::new(4096);
    /// let vec = holon.encode_json(r#"{"type": "billing", "amount": 100}"#).unwrap();
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

    // =========================================================================
    // Walkable Encoding (zero-serialization)
    // =========================================================================

    /// Encode any type implementing the Walkable trait.
    ///
    /// This is the zero-serialization path: your structs don't need to be
    /// converted to JSON first. Any type implementing Walkable can be
    /// encoded directly.
    ///
    /// # Example
    /// ```rust
    /// use holon::{Holon, Walkable, WalkType, WalkableValue, ScalarValue};
    ///
    /// struct Person { name: String, age: u32 }
    ///
    /// impl Walkable for Person {
    ///     fn walk_type(&self) -> WalkType { WalkType::Map }
    ///     fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
    ///         vec![
    ///             ("name", WalkableValue::Scalar(ScalarValue::String(self.name.clone()))),
    ///             ("age", WalkableValue::Scalar(ScalarValue::Int(self.age as i64))),
    ///         ]
    ///     }
    /// }
    ///
    /// let holon = Holon::new(4096);
    /// let person = Person { name: "Alice".into(), age: 30 };
    /// let vec = holon.encode_walkable(&person);
    /// ```
    pub fn encode_walkable<W: Walkable>(&self, walkable: &W) -> Vector {
        self.encode_walkable_recursive(walkable, None)
    }

    /// Encode a WalkableValue (for nested structures).
    pub fn encode_walkable_value(&self, value: &WalkableValue) -> Vector {
        self.encode_walkable_value_recursive(value, None)
    }

    fn encode_walkable_recursive<W: Walkable>(&self, walkable: &W, prefix: Option<&str>) -> Vector {
        match walkable.walk_type() {
            WalkType::Scalar => {
                let scalar = walkable.walk_scalar();
                self.encode_scalar_value(&scalar, prefix)
            }
            WalkType::Map => self.encode_walkable_map(walkable, prefix),
            WalkType::List => self.encode_walkable_list(walkable, prefix),
            WalkType::Set => self.encode_walkable_set(walkable, prefix),
        }
    }

    fn encode_walkable_value_recursive(&self, value: &WalkableValue, prefix: Option<&str>) -> Vector {
        match value {
            WalkableValue::Scalar(s) => self.encode_scalar_value(s, prefix),
            WalkableValue::Map(items) => {
                if items.is_empty() {
                    return self.encode_atom(&Self::make_path(prefix, "{}"));
                }

                let mut vectors: Vec<Vector> = Vec::new();

                for (key, val) in items {
                    let key_path = Self::make_path(prefix, key);
                    let role_vec = self.encode_atom(&key_path);
                    let filler_vec = self.encode_walkable_value_recursive(val, Some(&key_path));
                    let bound = Primitives::bind(&role_vec, &filler_vec);
                    vectors.push(bound);
                }

                let refs: Vec<&Vector> = vectors.iter().collect();
                Primitives::bundle(&refs)
            }
            WalkableValue::List(items) => {
                if items.is_empty() {
                    return self.encode_atom(&Self::make_path(prefix, "[]"));
                }

                let mut vectors: Vec<Vector> = Vec::new();

                for (i, item) in items.iter().enumerate() {
                    let pos_prefix = Self::make_path(prefix, &format!("[{}]", i));
                    let item_vec = self.encode_walkable_value_recursive(item, Some(&pos_prefix));
                    let pos_vec = self.encode_atom(&pos_prefix);
                    let bound = Primitives::bind(&pos_vec, &item_vec);
                    vectors.push(bound);
                }

                let refs: Vec<&Vector> = vectors.iter().collect();
                Primitives::bundle(&refs)
            }
            WalkableValue::Set(items) => {
                if items.is_empty() {
                    return self.encode_atom(&Self::make_path(prefix, "#{}"));
                }

                let set_indicator = self.encode_atom("set_indicator");
                let mut vectors: Vec<Vector> = Vec::new();

                for item in items {
                    let item_vec = self.encode_walkable_value_recursive(item, prefix);
                    vectors.push(item_vec);
                }

                let refs: Vec<&Vector> = vectors.iter().collect();
                let bundled = Primitives::bundle(&refs);
                Primitives::bind(&set_indicator, &bundled)
            }
        }
    }

    fn encode_scalar_value(&self, scalar: &ScalarValue, prefix: Option<&str>) -> Vector {
        // Handle numeric scalars with magnitude-aware encoding
        match scalar {
            ScalarValue::LogFloat { value, scale } => self.encode_scalar_log(*value, *scale),
            ScalarValue::LinearFloat { value, scale } => {
                self.encode_scalar_linear(*value, *scale)
            }
            ScalarValue::TimeFloat { value, resolution } => self.encode_time(*value, resolution),
            // Standard scalars use atom encoding
            _ => {
                let atom = scalar.to_atom();
                self.encode_atom(&Self::make_path(prefix, &atom))
            }
        }
    }

    /// Encode a scalar reference (zero-allocation path).
    #[inline]
    fn encode_scalar_ref(&self, scalar: ScalarRef<'_>, prefix: Option<&str>) -> Vector {
        // Handle numeric scalars with magnitude-aware encoding
        match scalar {
            ScalarRef::LogFloat { value, scale } => self.encode_scalar_log(value, scale),
            ScalarRef::LinearFloat { value, scale } => self.encode_scalar_linear(value, scale),
            ScalarRef::TimeFloat { value, resolution } => self.encode_time(value, &resolution),
            // Standard scalars use atom encoding
            _ => {
                let atom = scalar.to_atom();
                self.encode_atom(&Self::make_path(prefix, &atom))
            }
        }
    }

    fn encode_walkable_map<W: Walkable>(&self, walkable: &W, prefix: Option<&str>) -> Vector {
        // Use fast visitor path if available
        if walkable.has_fast_visitor() {
            return self.encode_walkable_map_fast(walkable, prefix);
        }

        // Fallback: use walk_map_items (allocates Vec)
        let items = walkable.walk_map_items();

        if items.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "{}"));
        }

        let mut vectors: Vec<Vector> = Vec::new();

        for (key, value) in items {
            let key_path = Self::make_path(prefix, key);

            // Role vector (the key)
            let role_vec = self.encode_atom(&key_path);

            // Filler vector (the value)
            let filler_vec = self.encode_walkable_value_recursive(&value, Some(&key_path));

            // Role-filler binding
            let bound = Primitives::bind(&role_vec, &filler_vec);
            vectors.push(bound);
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        Primitives::bundle(&refs)
    }

    /// Fast path using visitor - avoids Vec<WalkableValue> allocation for flat structs.
    fn encode_walkable_map_fast<W: Walkable>(&self, walkable: &W, prefix: Option<&str>) -> Vector {
        let mut vectors: Vec<Vector> = Vec::new();
        let mut has_items = false;

        walkable.walk_map_visitor(&mut |key, value_ref| {
            has_items = true;
            let key_path = Self::make_path(prefix, key);

            // Role vector (the key)
            let role_vec = self.encode_atom(&key_path);

            // Filler vector (the value) - inline encoding based on ref type
            let filler_vec = match value_ref {
                WalkableRef::Scalar(scalar_ref) => {
                    self.encode_scalar_ref(scalar_ref, Some(&key_path))
                }
                WalkableRef::Nested(nested_value) => {
                    self.encode_walkable_value_recursive(&nested_value, Some(&key_path))
                }
            };

            // Role-filler binding
            let bound = Primitives::bind(&role_vec, &filler_vec);
            vectors.push(bound);
        });

        if !has_items {
            return self.encode_atom(&Self::make_path(prefix, "{}"));
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        Primitives::bundle(&refs)
    }

    fn encode_walkable_list<W: Walkable>(&self, walkable: &W, prefix: Option<&str>) -> Vector {
        let items = walkable.walk_list_items();

        if items.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "[]"));
        }

        let mut vectors: Vec<Vector> = Vec::new();

        for (i, item) in items.iter().enumerate() {
            let pos_prefix = Self::make_path(prefix, &format!("[{}]", i));
            let item_vec = self.encode_walkable_value_recursive(item, Some(&pos_prefix));

            // Bind with position marker
            let pos_vec = self.encode_atom(&pos_prefix);
            let bound = Primitives::bind(&pos_vec, &item_vec);
            vectors.push(bound);
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        Primitives::bundle(&refs)
    }

    fn encode_walkable_set<W: Walkable>(&self, walkable: &W, prefix: Option<&str>) -> Vector {
        let items = walkable.walk_set_items();

        if items.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "#{}"));
        }

        let set_indicator = self.encode_atom("set_indicator");
        let mut vectors: Vec<Vector> = Vec::new();

        for item in items {
            let item_vec = self.encode_walkable_value_recursive(&item, prefix);
            vectors.push(item_vec);
        }

        let refs: Vec<&Vector> = vectors.iter().collect();
        let bundled = Primitives::bundle(&refs);
        Primitives::bind(&set_indicator, &bundled)
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

    // =========================================================================
    // Numeric Scalar Markers ($log, $linear, $scale)
    // =========================================================================

    /// Check if an object is a numeric scalar marker ($log or $linear).
    fn is_numeric_scalar_marker(obj: &serde_json::Map<String, Value>) -> bool {
        obj.contains_key("$log") || obj.contains_key("$linear")
    }

    /// Encode a numeric scalar marker.
    ///
    /// Supports:
    /// - `{"$log": value}` - Log10 encoding (equal ratios = equal similarity)
    /// - `{"$linear": value}` - Linear positional encoding
    /// - `{"$log": value, "$scale": 500}` - Custom scale parameter
    fn encode_numeric_scalar_marker(&self, obj: &serde_json::Map<String, Value>) -> Vector {
        let scale = obj
            .get("$scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1000.0);

        if let Some(value) = obj.get("$log") {
            if let Some(num) = value.as_f64() {
                return self.encode_scalar_log(num, scale);
            }
        }

        if let Some(value) = obj.get("$linear") {
            if let Some(num) = value.as_f64() {
                return self.encode_scalar_linear(num, scale);
            }
        }

        // Fallback: shouldn't happen if is_numeric_scalar_marker was checked
        Vector::zeros(self.dimensions())
    }

    /// Encode a scalar value using log10 scale.
    ///
    /// Equal ratios produce equal similarity drops:
    /// - 100 → 1000 (10x) has same similarity drop as 1000 → 10000 (10x)
    ///
    /// # Arguments
    /// * `value` - The value to encode (must be > 0)
    /// * `scale` - Controls similarity decay rate (default 1000.0)
    pub fn encode_scalar_log(&self, value: f64, scale: f64) -> Vector {
        if value <= 0.0 {
            return Vector::zeros(self.dimensions());
        }
        let log_value = value.log10();
        self.encode_scalar_positional(log_value, scale)
    }

    /// Encode a scalar value using linear positional encoding.
    ///
    /// Equal absolute differences produce equal similarity drops:
    /// - 10 → 20 (+10) has same similarity drop as 100 → 110 (+10)
    ///
    /// # Arguments
    /// * `value` - The value to encode
    /// * `scale` - Controls similarity decay rate (default 1000.0)
    pub fn encode_scalar_linear(&self, value: f64, scale: f64) -> Vector {
        self.encode_scalar_positional(value, scale)
    }

    /// Time-aware encoding: circular (hour-of-day, day-of-week, month) + positional.
    ///
    /// Matches Python's `_encode_time()` exactly, including role vector names.
    /// Role vector names are deliberately identical across languages for cross-language
    /// determinism: same seed → same role vectors → same encoded output.
    fn encode_time(&self, timestamp: f64, resolution: &TimeResolution) -> Vector {
        let dim = self.dimensions();

        // --- Decompose Unix timestamp with integer arithmetic (no chrono needed) ---
        let secs = timestamp as i64;

        // Seconds elapsed in current day (handles negative timestamps correctly)
        let sec_of_day = ((secs % 86400) + 86400) % 86400;
        let hour_frac = sec_of_day as f64 / 3600.0;

        // Day of week: Jan 1 1970 was Thursday (4). Add days elapsed.
        let days_elapsed = if secs >= 0 {
            secs / 86400
        } else {
            (secs - 86399) / 86400 // floor division for negative
        };
        let dow = ((days_elapsed % 7 + 4) % 7 + 7) % 7; // 0=Mon .. 6=Sun
        let dow_frac = dow as f64;

        // Approximate month (good enough for seasonal periodicity).
        // day_of_year is approximate since we skip leap-year logic.
        let day_of_year = ((days_elapsed % 365) + 365) % 365;
        let month_frac = day_of_year as f64 / 30.5;

        // --- Positional component (resolution-dependent) ---
        let position = match resolution {
            TimeResolution::Second => timestamp,
            TimeResolution::Minute => timestamp / 60.0,
            TimeResolution::Hour => timestamp / 3600.0,
            TimeResolution::Day => timestamp / 86400.0,
        };

        // --- Circular encoding (sin/cos interpolation between two random orthogonal vectors) ---
        let encode_circular = |value: f64, period: f64, base: &Vector, ortho: &Vector| -> Vector {
            let normalized = (value % period) / period;
            let angle = normalized * std::f64::consts::PI * 2.0;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let data: Vec<i8> = (0..dim)
                .map(|i| {
                    let v = base.data()[i] as f64 * cos_a + ortho.data()[i] as f64 * sin_a;
                    if v > 0.0 {
                        1
                    } else if v < 0.0 {
                        -1
                    } else {
                        0
                    }
                })
                .collect();
            Vector::from_data(data)
        };

        // Role vectors — names match Python exactly for cross-language determinism.
        let role_hour = self.vector_manager.get_vector("__time_role_hour__");
        let role_dow = self.vector_manager.get_vector("__time_role_dow__");
        let role_month = self.vector_manager.get_vector("__time_role_month__");
        let role_pos = self.vector_manager.get_vector("__time_role_position__");

        // Per-component base/ortho vectors (seeded from the role name for determinism)
        let base_hour = self.vector_manager.get_vector("__time_base_hour__");
        let ortho_hour = self.vector_manager.get_vector("__time_ortho_hour__");
        let base_dow = self.vector_manager.get_vector("__time_base_dow__");
        let ortho_dow = self.vector_manager.get_vector("__time_ortho_dow__");
        let base_month = self.vector_manager.get_vector("__time_base_month__");
        let ortho_month = self.vector_manager.get_vector("__time_ortho_month__");

        let hour_vec = encode_circular(hour_frac, 24.0, &base_hour, &ortho_hour);
        let dow_vec = encode_circular(dow_frac, 7.0, &base_dow, &ortho_dow);
        let month_vec = encode_circular(month_frac, 12.0, &base_month, &ortho_month);
        // Positional: transformer-style sin/cos at multiple frequencies
        let pos_vec = self.encode_scalar_positional(position, 10000.0);

        // Bind each component vector with its role, sum, then threshold to bipolar
        let mut sums = vec![0.0f64; dim];
        for (role, comp) in [
            (&role_hour, &hour_vec),
            (&role_dow, &dow_vec),
            (&role_month, &month_vec),
            (&role_pos, &pos_vec),
        ] {
            for i in 0..dim {
                sums[i] += role.data()[i] as f64 * comp.data()[i] as f64;
            }
        }

        Vector::from_data(
            sums.iter()
                .map(|&s| {
                    if s > 0.0 {
                        1
                    } else if s < 0.0 {
                        -1
                    } else {
                        0
                    }
                })
                .collect(),
        )
    }

    /// Transformer-style positional encoding.
    fn encode_scalar_positional(&self, position: f64, scale: f64) -> Vector {
        let dim = self.dimensions();
        let mut data = vec![0i8; dim];

        for i in 0..dim {
            let freq = 1.0 / scale.powf(i as f64 / dim as f64);
            let value = if i % 2 == 0 {
                (position * freq).sin()
            } else {
                (position * freq).cos()
            };
            data[i] = if value > 0.0 {
                1
            } else if value < 0.0 {
                -1
            } else {
                0
            };
        }

        Vector::from_data(data)
    }

    /// Encode an object using role-filler binding.
    fn encode_object(&self, obj: &serde_json::Map<String, Value>, prefix: Option<&str>) -> Vector {
        // Check for numeric scalar markers at top level
        if Self::is_numeric_scalar_marker(obj) {
            return self.encode_numeric_scalar_marker(obj);
        }

        if obj.is_empty() {
            return self.encode_atom(&Self::make_path(prefix, "{}"));
        }

        let mut vectors: Vec<Vector> = Vec::new();

        for (key, value) in obj {
            let key_path = Self::make_path(prefix, key);

            // Role vector (the key)
            let role_vec = self.encode_atom(&key_path);

            // Filler vector (the value, with key as prefix for nested structure)
            // Check if value is a numeric scalar marker
            let filler_vec = if let Value::Object(inner_obj) = value {
                if Self::is_numeric_scalar_marker(inner_obj) {
                    self.encode_numeric_scalar_marker(inner_obj)
                } else {
                    self.encode_value(value, Some(&key_path))
                }
            } else {
                self.encode_value(value, Some(&key_path))
            };

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

    // =========================================================================
    // Numeric Marker Tests ($log, $linear, $scale)
    // =========================================================================

    #[test]
    fn test_log_marker_basic() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // Values with 10x ratio should have similar distance
        let v100 = encoder.encode_json(r#"{"$log": 100}"#).unwrap();
        let v1000 = encoder.encode_json(r#"{"$log": 1000}"#).unwrap();
        let v10000 = encoder.encode_json(r#"{"$log": 10000}"#).unwrap();

        let sim_100_1000 = Similarity::cosine(&v100, &v1000);
        let sim_1000_10000 = Similarity::cosine(&v1000, &v10000);

        // 10x ratios should produce approximately equal similarity drops
        // (with some tolerance for quantization effects)
        let diff = (sim_100_1000 - sim_1000_10000).abs();
        assert!(
            diff < 0.1,
            "Expected similar similarity drops for 10x ratios, got {} vs {} (diff={})",
            sim_100_1000,
            sim_1000_10000,
            diff
        );
    }

    #[test]
    fn test_log_marker_magnitude_ordering() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let v100 = encoder.encode_json(r#"{"$log": 100}"#).unwrap();
        let v200 = encoder.encode_json(r#"{"$log": 200}"#).unwrap();
        let v10000 = encoder.encode_json(r#"{"$log": 10000}"#).unwrap();

        let sim_close = Similarity::cosine(&v100, &v200);
        let sim_far = Similarity::cosine(&v100, &v10000);

        // Closer values should have higher similarity
        assert!(
            sim_close > sim_far,
            "Expected closer values to be more similar, got {} vs {}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_log_marker_with_scale() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // Higher scale = slower similarity decay
        let v100_default = encoder.encode_json(r#"{"$log": 100}"#).unwrap();
        let v1000_default = encoder.encode_json(r#"{"$log": 1000}"#).unwrap();

        let v100_high = encoder.encode_json(r#"{"$log": 100, "$scale": 5000}"#).unwrap();
        let v1000_high = encoder.encode_json(r#"{"$log": 1000, "$scale": 5000}"#).unwrap();

        let sim_default = Similarity::cosine(&v100_default, &v1000_default);
        let sim_high_scale = Similarity::cosine(&v100_high, &v1000_high);

        // Higher scale should produce higher similarity for same ratio
        assert!(
            sim_high_scale > sim_default,
            "Expected higher scale to give higher similarity, got {} vs {}",
            sim_high_scale,
            sim_default
        );
    }

    #[test]
    fn test_linear_marker_basic() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let v0 = encoder.encode_json(r#"{"$linear": 0}"#).unwrap();
        let v10 = encoder.encode_json(r#"{"$linear": 10}"#).unwrap();
        let v50 = encoder.encode_json(r#"{"$linear": 50}"#).unwrap();

        let sim_0_10 = Similarity::cosine(&v0, &v10);
        let sim_0_50 = Similarity::cosine(&v0, &v50);

        // Closer values should have higher similarity
        assert!(
            sim_0_10 > sim_0_50,
            "Expected sim(0,10) > sim(0,50), got {} vs {}",
            sim_0_10,
            sim_0_50
        );
    }

    #[test]
    fn test_linear_vs_log_for_different_use_cases() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // Linear: 10 → 20 (+10) vs 100 → 110 (+10) should have similar drops
        let v10_lin = encoder.encode_json(r#"{"$linear": 10}"#).unwrap();
        let v20_lin = encoder.encode_json(r#"{"$linear": 20}"#).unwrap();
        let v100_lin = encoder.encode_json(r#"{"$linear": 100}"#).unwrap();
        let v110_lin = encoder.encode_json(r#"{"$linear": 110}"#).unwrap();

        let sim_lin_10_20 = Similarity::cosine(&v10_lin, &v20_lin);
        let sim_lin_100_110 = Similarity::cosine(&v100_lin, &v110_lin);

        // For linear encoding, equal absolute differences should give similar similarity
        let diff_lin = (sim_lin_10_20 - sim_lin_100_110).abs();
        assert!(
            diff_lin < 0.1,
            "Linear: Expected similar drops for +10 differences, got {} vs {} (diff={})",
            sim_lin_10_20,
            sim_lin_100_110,
            diff_lin
        );

        // Log: 10 → 20 (2x) vs 100 → 200 (2x) should have similar drops
        let v10_log = encoder.encode_json(r#"{"$log": 10}"#).unwrap();
        let v20_log = encoder.encode_json(r#"{"$log": 20}"#).unwrap();
        let v100_log = encoder.encode_json(r#"{"$log": 100}"#).unwrap();
        let v200_log = encoder.encode_json(r#"{"$log": 200}"#).unwrap();

        let sim_log_10_20 = Similarity::cosine(&v10_log, &v20_log);
        let sim_log_100_200 = Similarity::cosine(&v100_log, &v200_log);

        // For log encoding, equal ratios should give similar similarity
        let diff_log = (sim_log_10_20 - sim_log_100_200).abs();
        assert!(
            diff_log < 0.1,
            "Log: Expected similar drops for 2x ratios, got {} vs {} (diff={})",
            sim_log_10_20,
            sim_log_100_200,
            diff_log
        );
    }

    #[test]
    fn test_marker_in_record() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // Records with log-encoded rates
        let r1 = encoder
            .encode_json(r#"{"type": "traffic", "rate": {"$log": 1000}}"#)
            .unwrap();
        let r2 = encoder
            .encode_json(r#"{"type": "traffic", "rate": {"$log": 1100}}"#)
            .unwrap();
        let r3 = encoder
            .encode_json(r#"{"type": "traffic", "rate": {"$log": 50000}}"#)
            .unwrap();

        let sim_close = Similarity::cosine(&r1, &r2);
        let sim_far = Similarity::cosine(&r1, &r3);

        // Records with similar rates should be more similar
        assert!(
            sim_close > sim_far,
            "Expected records with similar rates to be more similar, got {} vs {}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_string_vs_log_encoding() {
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // String encoding: 100 and 101 are completely different strings
        let s100 = encoder.encode_json(r#"{"rate": 100}"#).unwrap();
        let s101 = encoder.encode_json(r#"{"rate": 101}"#).unwrap();
        let sim_string = Similarity::cosine(&s100, &s101);

        // Log encoding: 100 and 101 are very similar magnitudes
        let l100 = encoder.encode_json(r#"{"rate": {"$log": 100}}"#).unwrap();
        let l101 = encoder.encode_json(r#"{"rate": {"$log": 101}}"#).unwrap();
        let sim_log = Similarity::cosine(&l100, &l101);

        // Log should produce higher similarity for nearby values
        assert!(
            sim_log > sim_string,
            "Expected log encoding to show higher similarity for nearby values, got {} vs {}",
            sim_log,
            sim_string
        );
    }

    // =========================================================================
    // Walkable Numeric Scalar Tests
    // =========================================================================

    #[test]
    fn test_walkable_log_scalar() {
        use crate::walkable::{WalkType, Walkable, WalkableValue, ScalarValue};

        struct TrafficRecord {
            traffic_type: String,
            rate: f64,
        }

        impl Walkable for TrafficRecord {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("type", WalkableValue::Scalar(ScalarValue::String(self.traffic_type.clone()))),
                    ("rate", WalkableValue::Scalar(ScalarValue::log(self.rate))),
                ]
            }
        }

        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let r1 = TrafficRecord {
            traffic_type: "http".into(),
            rate: 1000.0,
        };
        let r2 = TrafficRecord {
            traffic_type: "http".into(),
            rate: 1100.0,
        };
        let r3 = TrafficRecord {
            traffic_type: "http".into(),
            rate: 50000.0,
        };

        let v1 = encoder.encode_walkable(&r1);
        let v2 = encoder.encode_walkable(&r2);
        let v3 = encoder.encode_walkable(&r3);

        let sim_close = Similarity::cosine(&v1, &v2);
        let sim_far = Similarity::cosine(&v1, &v3);

        // Similar rates should produce higher similarity
        assert!(
            sim_close > sim_far,
            "Expected similar rates to be more similar, got {} vs {}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_walkable_linear_scalar() {
        use crate::walkable::{WalkType, Walkable, WalkableValue, ScalarValue};

        struct Measurement {
            sensor: String,
            temperature: f64,
        }

        impl Walkable for Measurement {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("sensor", WalkableValue::Scalar(ScalarValue::String(self.sensor.clone()))),
                    ("temp", WalkableValue::Scalar(ScalarValue::linear(self.temperature))),
                ]
            }
        }

        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let m1 = Measurement {
            sensor: "room_a".into(),
            temperature: 20.0,
        };
        let m2 = Measurement {
            sensor: "room_a".into(),
            temperature: 22.0,
        };
        let m3 = Measurement {
            sensor: "room_a".into(),
            temperature: 50.0,
        };

        let v1 = encoder.encode_walkable(&m1);
        let v2 = encoder.encode_walkable(&m2);
        let v3 = encoder.encode_walkable(&m3);

        let sim_close = Similarity::cosine(&v1, &v2);
        let sim_far = Similarity::cosine(&v1, &v3);

        // Similar temperatures should produce higher similarity
        assert!(
            sim_close > sim_far,
            "Expected similar temperatures to be more similar, got {} vs {}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_walkable_visitor_with_log_scalar() {
        use crate::walkable::{WalkType, Walkable, WalkableValue, WalkableRef, ScalarValue};

        struct FastTrafficRecord {
            traffic_type: String,
            rate: f64,
        }

        impl Walkable for FastTrafficRecord {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("type", WalkableValue::Scalar(ScalarValue::String(self.traffic_type.clone()))),
                    ("rate", WalkableValue::Scalar(ScalarValue::log(self.rate))),
                ]
            }

            fn has_fast_visitor(&self) -> bool {
                true
            }

            fn walk_map_visitor(&self, visitor: &mut dyn FnMut(&str, WalkableRef<'_>)) {
                visitor("type", WalkableRef::string(&self.traffic_type));
                visitor("rate", WalkableRef::Scalar(crate::walkable::ScalarRef::log(self.rate)));
            }
        }

        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let r1 = FastTrafficRecord {
            traffic_type: "http".into(),
            rate: 1000.0,
        };
        let r2 = FastTrafficRecord {
            traffic_type: "http".into(),
            rate: 1100.0,
        };

        let v1 = encoder.encode_walkable(&r1);
        let v2 = encoder.encode_walkable(&r2);

        let sim = Similarity::cosine(&v1, &v2);

        // Similar rates should have high similarity
        assert!(
            sim > 0.5,
            "Expected similar rates to have high similarity, got {}",
            sim
        );
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

    // =========================================================================
    // TimeFloat Encoding Tests
    // =========================================================================

    #[test]
    fn test_time_same_hour_high_similarity() {
        use crate::walkable::ScalarValue;
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // Two timestamps exactly 1 week apart have the same hour and day-of-week.
        // The circular components should produce high similarity.
        let ts_base = 1_700_000_000.0f64; // arbitrary monday-morning-ish
        let ts_week_later = ts_base + 7.0 * 86400.0;

        let v1 = encoder.encode_scalar_value(
            &ScalarValue::time(ts_base),
            None,
        );
        let v2 = encoder.encode_scalar_value(
            &ScalarValue::time(ts_week_later),
            None,
        );

        let sim = Similarity::cosine(&v1, &v2);
        // Same circular components (hour+dow+month ≈ same), only positional differs.
        // Expect meaningful similarity.
        assert!(
            sim > 0.3,
            "Expected >0.3 similarity for timestamps 1 week apart (same circular components), got {}",
            sim
        );
    }

    #[test]
    fn test_time_positional_discrimination() {
        use crate::walkable::ScalarValue;
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let base = 1_700_000_000.0f64;
        let one_hour_later = base + 3600.0;
        let twelve_hours_later = base + 43200.0;

        let v_base = encoder.encode_scalar_value(&ScalarValue::time(base), None);
        let v_1h = encoder.encode_scalar_value(&ScalarValue::time(one_hour_later), None);
        let v_12h = encoder.encode_scalar_value(&ScalarValue::time(twelve_hours_later), None);

        let sim_1h = Similarity::cosine(&v_base, &v_1h);
        let sim_12h = Similarity::cosine(&v_base, &v_12h);

        // 1 hour apart should be more similar than 12 hours apart
        assert!(
            sim_1h > sim_12h,
            "Expected 1h-apart more similar than 12h-apart, got {} vs {}",
            sim_1h,
            sim_12h
        );
    }

    #[test]
    fn test_time_resolution_matters() {
        use crate::walkable::{ScalarValue, TimeResolution};
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let ts = 1_700_000_000.0f64;

        let v_second = encoder.encode_scalar_value(
            &ScalarValue::TimeFloat { value: ts, resolution: TimeResolution::Second },
            None,
        );
        let v_day = encoder.encode_scalar_value(
            &ScalarValue::TimeFloat { value: ts, resolution: TimeResolution::Day },
            None,
        );

        // Second and Day resolution produce different positional components
        let sim = Similarity::cosine(&v_second, &v_day);
        assert!(
            sim < 0.99,
            "Expected Second vs Day resolution to differ, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_time_scalar_ref_encodes() {
        use crate::walkable::ScalarRef;
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        // ScalarRef::time should produce a valid vector
        let v = encoder.encode_scalar_ref(ScalarRef::time(1_700_000_000.0), None);
        assert_eq!(v.dimensions(), 4096);
        assert!(v.nnz() > 0, "TimeFloat vector should be non-zero");
    }

    #[test]
    fn test_time_midnight_epoch_vs_midnight_year2() {
        // Midnight on Jan 1 1970 (epoch) vs Jan 1 1971 (year 2):
        // Both are midnight, both are Thursday, roughly same month position.
        // Circular components should be quite similar.
        use crate::walkable::ScalarValue;
        let vm = VectorManager::new(4096);
        let encoder = Encoder::new(vm);

        let epoch = 0.0f64;              // Jan 1 1970 00:00:00 UTC
        let year2 = 365.0 * 86400.0;    // Jan 1 1971 00:00:00 UTC (approx)

        let v_epoch = encoder.encode_scalar_value(&ScalarValue::time(epoch), None);
        let v_year2 = encoder.encode_scalar_value(&ScalarValue::time(year2), None);

        let sim = Similarity::cosine(&v_epoch, &v_year2);
        // Same hour (0), same day-of-week (Thursday), same month-fraction (~0).
        // Should share high circular similarity.
        assert!(
            sim > 0.3,
            "Expected >0.3 for same circular components (midnight Thursday, month 0), got {}",
            sim
        );
    }
}
