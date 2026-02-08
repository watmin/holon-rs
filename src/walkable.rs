//! Walkable trait for zero-serialization encoding.
//!
//! This module provides the Walkable trait that allows encoding in-memory data
//! structures directly without JSON/EDN serialization.
//!
//! # Design Goals
//!
//! - **Zero-cost abstraction**: Trait dispatch is compile-time where possible
//! - **Ergonomic**: Native Rust types work automatically
//! - **Extensible**: Users can implement for their own types
//!
//! # Example
//!
//! ```rust
//! use holon::{Walkable, WalkType, ScalarValue};
//!
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! impl Walkable for Person {
//!     fn walk_type(&self) -> WalkType {
//!         WalkType::Map
//!     }
//!
//!     fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
//!         vec![
//!             ("name", WalkableValue::Scalar(ScalarValue::String(self.name.clone()))),
//!             ("age", WalkableValue::Scalar(ScalarValue::Int(self.age as i64))),
//!         ]
//!     }
//! }
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};

/// The structural type of a walkable value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkType {
    /// Atomic values (string, int, float, bool, null)
    Scalar,
    /// Key-value pairs (like HashMap, structs)
    Map,
    /// Ordered sequences (like Vec, arrays)
    List,
    /// Unordered unique items (like HashSet)
    Set,
}

/// Scalar value types for encoding.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl ScalarValue {
    /// Convert to string for vector lookup.
    pub fn to_atom(&self) -> String {
        match self {
            ScalarValue::String(s) => s.clone(),
            ScalarValue::Int(i) => i.to_string(),
            ScalarValue::Float(f) => f.to_string(),
            ScalarValue::Bool(b) => b.to_string(),
            ScalarValue::Null => "null".to_string(),
        }
    }
}

/// A value that can be walked during encoding.
///
/// This enum allows returning owned values from `walk_*` methods,
/// making it easier to implement Walkable for custom types.
#[derive(Clone, Debug)]
pub enum WalkableValue {
    /// A scalar value
    Scalar(ScalarValue),
    /// A map of key-value pairs
    Map(Vec<(String, WalkableValue)>),
    /// An ordered list of items
    List(Vec<WalkableValue>),
    /// An unordered set of items
    Set(Vec<WalkableValue>),
}

impl WalkableValue {
    /// Get the walk type of this value.
    pub fn walk_type(&self) -> WalkType {
        match self {
            WalkableValue::Scalar(_) => WalkType::Scalar,
            WalkableValue::Map(_) => WalkType::Map,
            WalkableValue::List(_) => WalkType::List,
            WalkableValue::Set(_) => WalkType::Set,
        }
    }
}

/// Trait for types that can be walked and encoded by Holon.
///
/// Implement this trait to make your types directly encodable without
/// converting to JSON first.
///
/// # Which method to implement
///
/// Based on `walk_type()`, implement the corresponding method:
/// - `WalkType::Scalar` → `walk_scalar()`
/// - `WalkType::Map` → `walk_map_items()`
/// - `WalkType::List` → `walk_list_items()`
/// - `WalkType::Set` → `walk_set_items()`
pub trait Walkable {
    /// Return the structural type of this value.
    fn walk_type(&self) -> WalkType;

    /// Return the scalar value. Only valid for `WalkType::Scalar`.
    fn walk_scalar(&self) -> ScalarValue {
        panic!(
            "walk_scalar() called on {:?} (not a Scalar)",
            self.walk_type()
        )
    }

    /// Return (key, value) pairs. Only valid for `WalkType::Map`.
    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        panic!(
            "walk_map_items() called on {:?} (not a Map)",
            self.walk_type()
        )
    }

    /// Return items in order. Only valid for `WalkType::List`.
    fn walk_list_items(&self) -> Vec<WalkableValue> {
        panic!(
            "walk_list_items() called on {:?} (not a List)",
            self.walk_type()
        )
    }

    /// Return items (order not guaranteed). Only valid for `WalkType::Set`.
    fn walk_set_items(&self) -> Vec<WalkableValue> {
        panic!(
            "walk_set_items() called on {:?} (not a Set)",
            self.walk_type()
        )
    }

    /// Convert to a WalkableValue (for nested structures).
    fn to_walkable_value(&self) -> WalkableValue {
        match self.walk_type() {
            WalkType::Scalar => WalkableValue::Scalar(self.walk_scalar()),
            WalkType::Map => {
                let items: Vec<(String, WalkableValue)> = self
                    .walk_map_items()
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect();
                WalkableValue::Map(items)
            }
            WalkType::List => WalkableValue::List(self.walk_list_items()),
            WalkType::Set => WalkableValue::Set(self.walk_set_items()),
        }
    }
}

// =============================================================================
// Built-in implementations for Rust primitives
// =============================================================================

impl Walkable for String {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::String(self.clone())
    }
}

impl Walkable for &str {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::String(self.to_string())
    }
}

impl Walkable for i64 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Int(*self)
    }
}

impl Walkable for i32 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Int(*self as i64)
    }
}

impl Walkable for u32 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Int(*self as i64)
    }
}

impl Walkable for u64 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Int(*self as i64)
    }
}

impl Walkable for usize {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Int(*self as i64)
    }
}

impl Walkable for f64 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Float(*self)
    }
}

impl Walkable for f32 {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Float(*self as f64)
    }
}

impl Walkable for bool {
    fn walk_type(&self) -> WalkType {
        WalkType::Scalar
    }

    fn walk_scalar(&self) -> ScalarValue {
        ScalarValue::Bool(*self)
    }
}

impl<T: Walkable> Walkable for Option<T> {
    fn walk_type(&self) -> WalkType {
        match self {
            Some(v) => v.walk_type(),
            None => WalkType::Scalar,
        }
    }

    fn walk_scalar(&self) -> ScalarValue {
        match self {
            Some(v) => v.walk_scalar(),
            None => ScalarValue::Null,
        }
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        self.as_ref()
            .map(|v| v.walk_map_items())
            .unwrap_or_default()
    }

    fn walk_list_items(&self) -> Vec<WalkableValue> {
        self.as_ref()
            .map(|v| v.walk_list_items())
            .unwrap_or_default()
    }

    fn walk_set_items(&self) -> Vec<WalkableValue> {
        self.as_ref()
            .map(|v| v.walk_set_items())
            .unwrap_or_default()
    }
}

// Vec<T> as List
impl<T: Walkable> Walkable for Vec<T> {
    fn walk_type(&self) -> WalkType {
        WalkType::List
    }

    fn walk_list_items(&self) -> Vec<WalkableValue> {
        self.iter().map(|item| item.to_walkable_value()).collect()
    }
}

// Slice as List
impl<T: Walkable> Walkable for [T] {
    fn walk_type(&self) -> WalkType {
        WalkType::List
    }

    fn walk_list_items(&self) -> Vec<WalkableValue> {
        self.iter().map(|item| item.to_walkable_value()).collect()
    }
}

// HashMap<String, V> as Map
impl<V: Walkable> Walkable for HashMap<String, V> {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        self.iter()
            .map(|(k, v)| (k.as_str(), v.to_walkable_value()))
            .collect()
    }
}

// HashMap<&str, V> as Map
impl<V: Walkable> Walkable for HashMap<&str, V> {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        self.iter()
            .map(|(k, v)| (*k, v.to_walkable_value()))
            .collect()
    }
}

// HashSet<T> as Set
impl<T: Walkable> Walkable for HashSet<T> {
    fn walk_type(&self) -> WalkType {
        WalkType::Set
    }

    fn walk_set_items(&self) -> Vec<WalkableValue> {
        self.iter().map(|item| item.to_walkable_value()).collect()
    }
}

// BTreeSet<T> as Set
impl<T: Walkable> Walkable for BTreeSet<T> {
    fn walk_type(&self) -> WalkType {
        WalkType::Set
    }

    fn walk_set_items(&self) -> Vec<WalkableValue> {
        self.iter().map(|item| item.to_walkable_value()).collect()
    }
}

// WalkableValue is itself Walkable
impl Walkable for WalkableValue {
    fn walk_type(&self) -> WalkType {
        match self {
            WalkableValue::Scalar(_) => WalkType::Scalar,
            WalkableValue::Map(_) => WalkType::Map,
            WalkableValue::List(_) => WalkType::List,
            WalkableValue::Set(_) => WalkType::Set,
        }
    }

    fn walk_scalar(&self) -> ScalarValue {
        match self {
            WalkableValue::Scalar(s) => s.clone(),
            _ => panic!("walk_scalar() called on non-scalar WalkableValue"),
        }
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        match self {
            WalkableValue::Map(items) => items.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(),
            _ => panic!("walk_map_items() called on non-map WalkableValue"),
        }
    }

    fn walk_list_items(&self) -> Vec<WalkableValue> {
        match self {
            WalkableValue::List(items) => items.clone(),
            _ => panic!("walk_list_items() called on non-list WalkableValue"),
        }
    }

    fn walk_set_items(&self) -> Vec<WalkableValue> {
        match self {
            WalkableValue::Set(items) => items.clone(),
            _ => panic!("walk_set_items() called on non-set WalkableValue"),
        }
    }

    fn to_walkable_value(&self) -> WalkableValue {
        self.clone()
    }
}

// serde_json::Value as Walkable (integration with existing JSON path)
impl Walkable for serde_json::Value {
    fn walk_type(&self) -> WalkType {
        match self {
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => WalkType::Scalar,
            serde_json::Value::Array(_) => WalkType::List,
            serde_json::Value::Object(_) => WalkType::Map,
        }
    }

    fn walk_scalar(&self) -> ScalarValue {
        match self {
            serde_json::Value::Null => ScalarValue::Null,
            serde_json::Value::Bool(b) => ScalarValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ScalarValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    ScalarValue::Float(f)
                } else {
                    ScalarValue::String(n.to_string())
                }
            }
            serde_json::Value::String(s) => ScalarValue::String(s.clone()),
            _ => panic!("walk_scalar() called on non-scalar JSON value"),
        }
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        match self {
            serde_json::Value::Object(obj) => obj
                .iter()
                .map(|(k, v)| (k.as_str(), v.to_walkable_value()))
                .collect(),
            _ => panic!("walk_map_items() called on non-object JSON value"),
        }
    }

    fn walk_list_items(&self) -> Vec<WalkableValue> {
        match self {
            serde_json::Value::Array(arr) => {
                arr.iter().map(|v| v.to_walkable_value()).collect()
            }
            _ => panic!("walk_list_items() called on non-array JSON value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_string() {
        let s = "hello".to_string();
        assert_eq!(s.walk_type(), WalkType::Scalar);
        assert_eq!(s.walk_scalar(), ScalarValue::String("hello".to_string()));
    }

    #[test]
    fn test_scalar_int() {
        let n: i64 = 42;
        assert_eq!(n.walk_type(), WalkType::Scalar);
        assert_eq!(n.walk_scalar(), ScalarValue::Int(42));
    }

    #[test]
    fn test_vec_as_list() {
        let v = vec!["a".to_string(), "b".to_string()];
        assert_eq!(v.walk_type(), WalkType::List);

        let items = v.walk_list_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_hashmap_as_map() {
        let mut m: HashMap<String, i64> = HashMap::new();
        m.insert("x".to_string(), 10);
        m.insert("y".to_string(), 20);

        assert_eq!(m.walk_type(), WalkType::Map);

        let items = m.walk_map_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_hashset_as_set() {
        let mut s: HashSet<String> = HashSet::new();
        s.insert("a".to_string());
        s.insert("b".to_string());

        assert_eq!(s.walk_type(), WalkType::Set);

        let items = s.walk_set_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_custom_struct() {
        struct Person {
            name: String,
            age: u32,
        }

        impl Walkable for Person {
            fn walk_type(&self) -> WalkType {
                WalkType::Map
            }

            fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
                vec![
                    ("name", self.name.to_walkable_value()),
                    ("age", self.age.to_walkable_value()),
                ]
            }
        }

        let p = Person {
            name: "Alice".to_string(),
            age: 30,
        };

        assert_eq!(p.walk_type(), WalkType::Map);
        let items = p.walk_map_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_json_value() {
        let v: serde_json::Value = serde_json::json!({
            "name": "test",
            "count": 42
        });

        assert_eq!(v.walk_type(), WalkType::Map);
        let items = v.walk_map_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_nested_structure() {
        let mut inner: HashMap<String, i64> = HashMap::new();
        inner.insert("x".to_string(), 1);
        inner.insert("y".to_string(), 2);

        let outer = vec![inner.to_walkable_value()];

        assert_eq!(outer.walk_type(), WalkType::List);
        let items = outer.walk_list_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].walk_type(), WalkType::Map);
    }
}
