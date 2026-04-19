//! Atom type registry — canonical-bytes dispatch for parametric `Atom<T>`.
//!
//! The `HolonAST::Atom(Arc<dyn Any>)` variant stores arbitrary T per 058-001's
//! parametric acceptance. To deterministically hash an Atom's payload, the
//! encoder looks up T's canonical-bytes function in this registry.
//!
//! # What registers when
//!
//! - **Rust primitives** — registered unconditionally by [`AtomTypeRegistry::with_builtins`]
//!   at construction. Every wat program can atomize `i64`, `f64`, `bool`,
//!   `String`, `char`, all signed/unsigned integer widths, and `f32`.
//! - **`HolonAST` itself** — also registered in `with_builtins`. This enables
//!   programs-as-atoms: `Atom(some_holon_ast)` produces an opaque-identity
//!   vector whose hash is over the holon's canonical-EDN form (recursive).
//! - **User-declared types** — `struct`, `enum`, `newtype`, `typealias`
//!   declarations in wat source will register their canonicalizers at
//!   startup (a later slice, once the wat-vm frontend exists). For now,
//!   applications using this crate directly can call
//!   [`AtomTypeRegistry::register`] at startup to add their own types.
//!
//! # The `:Any` ban extends here
//!
//! Users never reach for `std::any::Any` in wat source — the wat type checker
//! refuses `:Any` as an annotation. The `Any` trait here is pure Rust
//! substrate plumbing: it exists so the `HolonAST::Atom` variant can hold
//! heterogeneous T at runtime, and the registry provides the TypeId-dispatched
//! downcast path. Every call site's T is guaranteed by the type checker, so
//! the registry lookup never fails in well-typed programs.

use super::holon_ast::{canonical_edn_holon, HolonAST};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Signature of a canonical-bytes function.
///
/// Takes the payload (as `&dyn Any`) and the registry itself (for recursive
/// canonicalization when the payload is a `HolonAST`). Returns deterministic
/// bytes that hash to the atom's identity vector.
pub type CanonicalFn =
    Arc<dyn Fn(&(dyn Any + Send + Sync), &AtomTypeRegistry) -> Vec<u8> + Send + Sync>;

struct RegistryEntry {
    type_tag: &'static str,
    canonical: CanonicalFn,
}

/// Registry of canonical-bytes handlers keyed by `TypeId`.
///
/// Construct with [`AtomTypeRegistry::with_builtins`] to get the Rust primitives
/// and `HolonAST` pre-registered; or [`AtomTypeRegistry::new`] for a bare
/// registry (tests, specialized uses).
pub struct AtomTypeRegistry {
    entries: HashMap<TypeId, RegistryEntry>,
}

impl Default for AtomTypeRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl AtomTypeRegistry {
    /// Create an empty registry. Most callers want [`with_builtins`] instead.
    pub fn new() -> Self {
        AtomTypeRegistry {
            entries: HashMap::new(),
        }
    }

    /// Create a registry with all built-in handlers registered.
    ///
    /// Covers every Rust primitive (all integer widths, both float widths,
    /// `bool`, `char`, `String`, `&'static str`, `()` unit) plus `HolonAST`
    /// itself for programs-as-atoms.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register_builtins();
        r
    }

    /// Register a canonical-bytes handler for type `T`.
    ///
    /// - `type_tag` is the stable wire-format name used in the atom's hash
    ///   input. Use the wat spec name (`"i64"`, `"f64"`, `"wat/algebra/Holon"`)
    ///   or the user type's keyword-path (`"project/market/Candle"`). The tag
    ///   must be stable across nodes running the same program; it is not
    ///   derived from Rust's `TypeId` or `type_name` (both are unstable
    ///   across Rust versions).
    /// - `canonical` produces deterministic bytes from a `&T`. Same value
    ///   must always produce the same bytes.
    pub fn register<T: Any + Send + Sync + 'static>(
        &mut self,
        type_tag: &'static str,
        canonical: impl Fn(&T, &AtomTypeRegistry) -> Vec<u8> + Send + Sync + 'static,
    ) {
        let wrapped: CanonicalFn = Arc::new(move |v, registry| {
            let typed = v.downcast_ref::<T>().expect(
                "AtomTypeRegistry: type mismatch at downcast. \
                 This is a registry bug — TypeId matched but downcast failed.",
            );
            canonical(typed, registry)
        });
        self.entries.insert(
            TypeId::of::<T>(),
            RegistryEntry {
                type_tag,
                canonical: wrapped,
            },
        );
    }

    /// Get the stable type tag and canonical-bytes function for a payload.
    fn lookup(&self, tid: TypeId) -> Option<(&'static str, &CanonicalFn)> {
        self.entries.get(&tid).map(|e| (e.type_tag, &e.canonical))
    }

    /// Canonicalize an atom payload to bytes.
    ///
    /// Panics if T is unregistered. Well-typed programs never hit this panic
    /// because the wat type checker guarantees every atom's T is registered.
    /// Applications calling this crate directly must register their types
    /// before atomizing them.
    pub fn canonical_bytes(&self, value: &(dyn Any + Send + Sync)) -> Vec<u8> {
        let tid = value.type_id();
        match self.lookup(tid) {
            Some((_, f)) => f(value, self),
            None => panic!(
                "AtomTypeRegistry: unregistered type (TypeId={:?}). \
                 Built-in primitives and HolonAST are registered by \
                 AtomTypeRegistry::with_builtins(); user types must register \
                 explicitly via AtomTypeRegistry::register::<T>(\"your/type/tag\", ...).",
                tid
            ),
        }
    }

    /// Look up a payload's stable type tag.
    ///
    /// Same panic semantics as [`canonical_bytes`].
    pub fn type_tag(&self, value: &(dyn Any + Send + Sync)) -> &'static str {
        self.lookup(value.type_id())
            .map(|(tag, _)| tag)
            .unwrap_or_else(|| {
                panic!(
                    "AtomTypeRegistry: unregistered type (TypeId={:?}).",
                    value.type_id()
                )
            })
    }

    /// Test whether type `T` is registered.
    pub fn contains<T: Any + Send + Sync + 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    fn register_builtins(&mut self) {
        // Signed integers
        self.register::<i8>("i8", |v, _| v.to_le_bytes().to_vec());
        self.register::<i16>("i16", |v, _| v.to_le_bytes().to_vec());
        self.register::<i32>("i32", |v, _| v.to_le_bytes().to_vec());
        self.register::<i64>("i64", |v, _| v.to_le_bytes().to_vec());
        self.register::<i128>("i128", |v, _| v.to_le_bytes().to_vec());
        self.register::<isize>("isize", |v, _| v.to_le_bytes().to_vec());

        // Unsigned integers
        self.register::<u8>("u8", |v, _| v.to_le_bytes().to_vec());
        self.register::<u16>("u16", |v, _| v.to_le_bytes().to_vec());
        self.register::<u32>("u32", |v, _| v.to_le_bytes().to_vec());
        self.register::<u64>("u64", |v, _| v.to_le_bytes().to_vec());
        self.register::<u128>("u128", |v, _| v.to_le_bytes().to_vec());
        self.register::<usize>("usize", |v, _| v.to_le_bytes().to_vec());

        // Floats
        self.register::<f32>("f32", |v, _| v.to_le_bytes().to_vec());
        self.register::<f64>("f64", |v, _| v.to_le_bytes().to_vec());

        // Misc primitives
        self.register::<bool>("bool", |v, _| vec![*v as u8]);
        self.register::<char>("char", |v, _| (*v as u32).to_le_bytes().to_vec());
        self.register::<()>("unit", |_, _| Vec::new());

        // Strings
        self.register::<String>("String", |v, _| v.as_bytes().to_vec());
        self.register::<&'static str>("&str", |v, _| v.as_bytes().to_vec());

        // HolonAST — programs-as-atoms. Canonical-EDN walks the AST recursively
        // through this same registry, so an Atom wrapping a Holon whose leaves
        // are themselves typed atoms (i64, f64, user struct, another Holon)
        // resolves all the way down.
        self.register::<HolonAST>("wat/algebra/Holon", |v, registry| {
            canonical_edn_holon(v, registry)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_registered() {
        let r = AtomTypeRegistry::with_builtins();
        assert!(r.contains::<i64>());
        assert!(r.contains::<f64>());
        assert!(r.contains::<bool>());
        assert!(r.contains::<String>());
        assert!(r.contains::<HolonAST>());
    }

    #[test]
    fn canonical_bytes_i64_deterministic() {
        let r = AtomTypeRegistry::with_builtins();
        let a: i64 = 42;
        let b: i64 = 42;
        assert_eq!(r.canonical_bytes(&a), r.canonical_bytes(&b));
    }

    #[test]
    fn canonical_bytes_different_types_produce_different_tags() {
        let r = AtomTypeRegistry::with_builtins();
        let as_i64: i64 = 42;
        let as_i32: i32 = 42;
        // Same value, different types — type tags differ.
        assert_ne!(r.type_tag(&as_i64), r.type_tag(&as_i32));
    }

    #[test]
    fn register_user_type() {
        #[derive(Debug)]
        struct MyCandle {
            open: f64,
            close: f64,
        }

        let mut r = AtomTypeRegistry::with_builtins();
        r.register::<MyCandle>("project/market/Candle", |v, _| {
            let mut out = Vec::new();
            out.extend_from_slice(&v.open.to_le_bytes());
            out.extend_from_slice(&v.close.to_le_bytes());
            out
        });

        let c = MyCandle {
            open: 100.0,
            close: 101.5,
        };
        let bytes = r.canonical_bytes(&c);
        assert_eq!(bytes.len(), 16); // two f64s
        assert_eq!(r.type_tag(&c), "project/market/Candle");
    }

    #[test]
    #[should_panic(expected = "unregistered type")]
    fn unregistered_type_panics() {
        #[derive(Debug)]
        struct Unknown;
        let r = AtomTypeRegistry::new();
        let u = Unknown;
        r.canonical_bytes(&u);
    }
}
