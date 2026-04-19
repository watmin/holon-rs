//! HolonAST — the universal AST for the wat algebra's 6 core forms.
//!
//! Under 058 FOUNDATION's "Two Tiers of wat" split, HolonAST is the UpperCase
//! tier — AST constructors that do not run. The vector projection materializes
//! on demand via [`encode`].
//!
//! The 6 variants correspond to algebra core per FOUNDATION.md:
//! `Atom`, `Bind`, `Bundle`, `Permute`, `Thermometer`, `Blend`.
//!
//! # Parametric Atom
//!
//! `Atom<T>` (058-001 ACCEPTED as parametric) stores any Rust type T. Rust-level
//! storage is `Arc<dyn Any + Send + Sync>`; the wat type-checker guarantees the
//! concrete T at every use site, so runtime downcasts never fail. Hash input is
//! `(TypeId, canonical bytes of value)` so `Atom(42_i64)`, `Atom("42".to_string())`,
//! `Atom(42.0_f64)`, and `Atom(true)` all produce distinct vectors.
//!
//! `:Any` is NOT a wat-level type. The grammar refuses it; the type checker
//! refuses it. `dyn Any` is Rust substrate plumbing, never wat-visible.

use super::primitives::Primitives;
use super::scalar::{ScalarEncoder, ScalarMode};
use super::similarity::Similarity;
use super::vector::Vector;
use super::vector_manager::VectorManager;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::sync::Arc;

/// The universal AST for the 6 algebra core forms.
pub enum HolonAST {
    /// `Atom(T)` — parametric over any Rust type T. See module doc.
    Atom(Arc<dyn Any + Send + Sync>),

    /// `Bind(a, b)` — elementwise multiplication; MAP's "M".
    Bind(Arc<HolonAST>, Arc<HolonAST>),

    /// `Bundle(xs)` — elementwise sum + ternary threshold; MAP's "A".
    /// Takes a list per 058-003.
    Bundle(Arc<Vec<HolonAST>>),

    /// `Permute(child, k)` — cyclic shift; MAP's "P".
    /// `Permute(v, k)[i] = v[(i + k) mod d]` per CORE-AUDIT.
    Permute(Arc<HolonAST>, i32),

    /// `Thermometer(value, min, max)` — gradient encoding.
    /// First `N = round(d · clamp((value-min)/(max-min), 0, 1))` dims are `+1`,
    /// remaining `d - N` dims are `-1`. Canonical layout; bit-identical across
    /// nodes at the same d. See CORE-AUDIT / 058-023.
    Thermometer { value: f64, min: f64, max: f64 },

    /// `Blend(a, b, w1, w2)` — `threshold(w1·a + w2·b)`.
    /// Option B per 058-002: two independent real-valued weights; negative allowed.
    Blend(Arc<HolonAST>, Arc<HolonAST>, f64, f64),
}

impl HolonAST {
    /// Construct an Atom from any Rust type that is `Any + Send + Sync`.
    ///
    /// Canonicalization during encoding dispatches on `TypeId`; only types with
    /// registered canonical-bytes handlers can be encoded. Built-in handlers
    /// cover Rust primitives; user-declared types will register at startup
    /// (later slice).
    pub fn atom<T: Any + Send + Sync + 'static>(value: T) -> Self {
        HolonAST::Atom(Arc::new(value))
    }

    /// Construct a Bind node.
    pub fn bind(a: HolonAST, b: HolonAST) -> Self {
        HolonAST::Bind(Arc::new(a), Arc::new(b))
    }

    /// Construct a Bundle node from a list of children.
    pub fn bundle(children: Vec<HolonAST>) -> Self {
        HolonAST::Bundle(Arc::new(children))
    }

    /// Construct a Permute node with step k.
    pub fn permute(child: HolonAST, k: i32) -> Self {
        HolonAST::Permute(Arc::new(child), k)
    }

    /// Construct a Thermometer node.
    pub fn thermometer(value: f64, min: f64, max: f64) -> Self {
        HolonAST::Thermometer { value, min, max }
    }

    /// Construct a Blend node with two independent weights (Option B).
    pub fn blend(a: HolonAST, b: HolonAST, w1: f64, w2: f64) -> Self {
        HolonAST::Blend(Arc::new(a), Arc::new(b), w1, w2)
    }
}

/// Canonical-bytes function for Atom payloads.
///
/// Dispatches on `TypeId` to produce a deterministic byte representation of
/// the atom's value. Type-tag is the `TypeId`'s `u64` bits; value-bytes are
/// the little-endian or UTF-8 representation of the concrete T.
///
/// Built-in types covered in this slice: `String`, `&'static str`, `i8`, `i16`,
/// `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `f32`, `f64`,
/// `bool`, `char`, `isize`, `usize`.
///
/// Other types (HolonAST, user-declared types) will be supported by a registry
/// populated at startup; for now, unsupported types panic with a clear message.
fn atom_canonical_bytes(value: &(dyn Any + Send + Sync)) -> Vec<u8> {
    if let Some(s) = value.downcast_ref::<String>() {
        return s.as_bytes().to_vec();
    }
    if let Some(s) = value.downcast_ref::<&'static str>() {
        return s.as_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<i8>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<i16>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<i32>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<i64>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<i128>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<u8>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<u16>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<u32>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<u64>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<u128>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<isize>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<usize>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<f32>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<f64>() {
        return v.to_le_bytes().to_vec();
    }
    if let Some(v) = value.downcast_ref::<bool>() {
        return vec![*v as u8];
    }
    if let Some(v) = value.downcast_ref::<char>() {
        return (*v as u32).to_le_bytes().to_vec();
    }
    panic!(
        "HolonAST::Atom: unsupported payload type (TypeId={:?}). \
         Only Rust primitives are supported in this slice; HolonAST payloads \
         and user-declared types require the atom type registry (not yet implemented).",
        value.type_id()
    );
}

/// Hash an Atom's payload to a seed for vector generation.
///
/// Seed input: `(TypeId bits, canonical bytes)`. Different types with identical
/// bytes produce different seeds — `(Atom 42_i64)` ≠ `(Atom "42".to_string())`.
fn atom_seed(value: &(dyn Any + Send + Sync), global_seed: u64) -> u64 {
    let type_id = value.type_id();
    let bytes = atom_canonical_bytes(value);

    let mut hasher = Sha256::new();
    hasher.update(global_seed.to_le_bytes());
    // TypeId's internal u128 bits aren't stable across Rust versions, but
    // within a single compilation they're deterministic — good enough for
    // in-process type discrimination. Future slices may swap to a stable
    // type-tag string (canonical type name) once user types register.
    hasher.update(format!("{:?}", type_id).as_bytes());
    hasher.update(&bytes);
    let hash = hasher.finalize();

    u64::from_le_bytes(hash[0..8].try_into().unwrap())
}

/// Realize a HolonAST into a Vector by walking the AST and dispatching to
/// the lowercase Rust primitives.
///
/// Encoding is deterministic: same AST ⇒ same vector. No caching in this slice
/// (L1/L2 cache lands later per FOUNDATION's "Caching Is Memoization" section).
pub fn encode(ast: &HolonAST, vm: &VectorManager, scalar: &ScalarEncoder) -> Vector {
    match ast {
        HolonAST::Atom(payload) => {
            // Seed a deterministic vector from (TypeId, canonical bytes, global seed).
            let seed = atom_seed(payload.as_ref(), vm.global_seed());
            let dims = vm.dimensions();
            deterministic_vector_from_seed(seed, dims)
        }
        HolonAST::Bind(a, b) => {
            let va = encode(a, vm, scalar);
            let vb = encode(b, vm, scalar);
            Primitives::bind(&va, &vb)
        }
        HolonAST::Bundle(children) => {
            let vectors: Vec<Vector> = children.iter().map(|c| encode(c, vm, scalar)).collect();
            let refs: Vec<&Vector> = vectors.iter().collect();
            Primitives::bundle(&refs)
        }
        HolonAST::Permute(child, k) => {
            let vc = encode(child, vm, scalar);
            Primitives::permute(&vc, *k)
        }
        HolonAST::Thermometer { value, min, max } => {
            scalar.encode(*value, ScalarMode::Thermometer { min: *min, max: *max })
        }
        HolonAST::Blend(a, b, w1, w2) => {
            let va = encode(a, vm, scalar);
            let vb = encode(b, vm, scalar);
            Primitives::blend_weighted(&va, &vb, *w1, *w2)
        }
    }
}

/// Recover the inner T from an Atom node.
///
/// Returns `Some(&T)` if the atom's payload is of type T; `None` otherwise.
/// The wat type-checker guarantees T at every call site in well-typed programs,
/// so `None` indicates a bug or a non-Atom variant.
pub fn atom_value<T: Any + Send + Sync + 'static>(ast: &HolonAST) -> Option<&T> {
    match ast {
        HolonAST::Atom(payload) => payload.downcast_ref::<T>(),
        _ => None,
    }
}

/// Generate a deterministic ternary vector from a u64 seed.
///
/// Produces values in `{-1, 0, +1}^d` with uniform 1/3 probability each,
/// matching the distribution used by `VectorManager::compute_vector`.
fn deterministic_vector_from_seed(seed: u64, dims: usize) -> Vector {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0i8; dims];
    for d in data.iter_mut().take(dims) {
        let r = rng.next_u32() % 3;
        *d = match r {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => unreachable!(),
        };
    }
    Vector::from_data(data)
}

// Keep Similarity in scope so downstream users can `use` this module and get
// `cosine` / `dot` together with `encode`.
#[allow(unused_imports)]
use Similarity as _;

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 1024;

    fn fresh_vm_and_scalar() -> (VectorManager, ScalarEncoder) {
        (VectorManager::with_seed(D, 42), ScalarEncoder::with_seed(D, 42))
    }

    #[test]
    fn atom_type_discrimination_int_vs_string() {
        let (vm, se) = fresh_vm_and_scalar();
        let v_int = encode(&HolonAST::atom(42_i64), &vm, &se);
        let v_str = encode(&HolonAST::atom("42".to_string()), &vm, &se);
        assert_ne!(v_int, v_str, "Atom(42_i64) must differ from Atom(\"42\")");
    }

    #[test]
    fn atom_type_discrimination_int_vs_float() {
        let (vm, se) = fresh_vm_and_scalar();
        let v_int = encode(&HolonAST::atom(42_i64), &vm, &se);
        let v_float = encode(&HolonAST::atom(42.0_f64), &vm, &se);
        assert_ne!(v_int, v_float, "Atom(42_i64) must differ from Atom(42.0_f64)");
    }

    #[test]
    fn atom_type_discrimination_bool_vs_int() {
        let (vm, se) = fresh_vm_and_scalar();
        let v_bool = encode(&HolonAST::atom(true), &vm, &se);
        let v_int = encode(&HolonAST::atom(1_i64), &vm, &se);
        assert_ne!(v_bool, v_int, "Atom(true) must differ from Atom(1_i64)");
    }

    #[test]
    fn atom_deterministic() {
        let (vm1, se1) = fresh_vm_and_scalar();
        let (vm2, se2) = fresh_vm_and_scalar();
        let v1 = encode(&HolonAST::atom("hello".to_string()), &vm1, &se1);
        let v2 = encode(&HolonAST::atom("hello".to_string()), &vm2, &se2);
        assert_eq!(v1, v2, "Same AST + same seed must produce same vector");
    }

    #[test]
    fn atom_value_recovery() {
        let node = HolonAST::atom(42_i64);
        let recovered: Option<&i64> = atom_value(&node);
        assert_eq!(recovered, Some(&42_i64));
    }

    #[test]
    fn atom_value_wrong_type_none() {
        let node = HolonAST::atom(42_i64);
        let wrong: Option<&String> = atom_value(&node);
        assert!(wrong.is_none());
    }

    #[test]
    fn atom_value_non_atom_none() {
        let a = HolonAST::atom(1_i64);
        let b = HolonAST::atom(2_i64);
        let bound = HolonAST::bind(a, b);
        let v: Option<&i64> = atom_value(&bound);
        assert!(v.is_none());
    }

    #[test]
    fn bind_composes_atoms() {
        let (vm, se) = fresh_vm_and_scalar();
        let a = HolonAST::atom("role".to_string());
        let b = HolonAST::atom("filler".to_string());
        let v_bound = encode(&HolonAST::bind(a, b), &vm, &se);
        assert_eq!(v_bound.dimensions(), D);
    }

    #[test]
    fn bundle_list_form() {
        let (vm, se) = fresh_vm_and_scalar();
        let children = vec![
            HolonAST::atom("a".to_string()),
            HolonAST::atom("b".to_string()),
            HolonAST::atom("c".to_string()),
        ];
        let v = encode(&HolonAST::bundle(children), &vm, &se);
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    fn permute_by_zero_is_identity() {
        let (vm, se) = fresh_vm_and_scalar();
        let a = HolonAST::atom("x".to_string());
        let v_original = encode(&a, &vm, &se);
        let v_permuted = encode(&HolonAST::permute(a, 0), &vm, &se);
        assert_eq!(v_original, v_permuted, "Permute(v, 0) must be identity");
    }

    #[test]
    fn permute_is_invertible() {
        let (vm, se) = fresh_vm_and_scalar();
        let a = HolonAST::atom("x".to_string());
        // Permute by +k then by -k should return the original.
        let forward = HolonAST::permute(HolonAST::atom("x".to_string()), 7);
        let round_trip = HolonAST::permute(forward, -7);
        let v_original = encode(&a, &vm, &se);
        let v_round_trip = encode(&round_trip, &vm, &se);
        assert_eq!(
            v_original, v_round_trip,
            "Permute(Permute(v, k), -k) must equal v"
        );
    }

    #[test]
    fn thermometer_endpoints() {
        let (vm, se) = fresh_vm_and_scalar();
        let v_min = encode(&HolonAST::thermometer(0.0, 0.0, 100.0), &vm, &se);
        let v_max = encode(&HolonAST::thermometer(100.0, 0.0, 100.0), &vm, &se);
        // All -1 at min, all +1 at max, so cosine = -1
        let sim = Similarity::cosine(&v_min, &v_max);
        assert!(
            sim < -0.99,
            "Thermometer at endpoints should have cosine ≈ -1; got {}",
            sim
        );
    }

    #[test]
    fn blend_option_b_subtract() {
        // Subtract = Blend(x, y, 1, -1)
        let (vm, se) = fresh_vm_and_scalar();
        let x = HolonAST::atom("x".to_string());
        let y = HolonAST::atom("y".to_string());
        let v_sub = encode(&HolonAST::blend(x, y, 1.0, -1.0), &vm, &se);
        assert_eq!(v_sub.dimensions(), D);
    }

    #[test]
    fn blend_option_b_circular_weights() {
        // Circular uses (cos θ, sin θ) — sum ≠ 1; proves Option B needed.
        let (vm, se) = fresh_vm_and_scalar();
        let cos_basis = HolonAST::atom("wat/std/circular-cos-basis".to_string());
        let sin_basis = HolonAST::atom("wat/std/circular-sin-basis".to_string());
        let theta = std::f64::consts::FRAC_PI_4; // cos + sin ≈ 1.414 at π/4
        let v = encode(
            &HolonAST::blend(cos_basis, sin_basis, theta.cos(), theta.sin()),
            &vm,
            &se,
        );
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    #[should_panic(expected = "unsupported payload type")]
    fn unsupported_payload_type_panics() {
        // A type with no canonical handler (custom struct, not a Rust primitive)
        // panics with a clear message. Future slices register user types.
        #[derive(Debug)]
        struct Unknown(i64);
        let (vm, se) = fresh_vm_and_scalar();
        let _ = encode(&HolonAST::atom(Unknown(1)), &vm, &se);
    }
}
