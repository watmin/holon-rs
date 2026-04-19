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
//! concrete T at every use site, so runtime downcasts never fail.
//!
//! Hash input is `(type_tag, canonical bytes of value)` via [`AtomTypeRegistry`].
//! Different types with identical bytes produce different vectors —
//! `Atom(42_i64)`, `Atom("42".to_string())`, `Atom(42.0_f64)`, `Atom(true)`
//! are four distinct atoms.
//!
//! Programs-as-atoms: `Atom(some_holon_ast)` produces an opaque-identity vector
//! whose hash is over the holon's canonical-EDN form. Two legitimate encodings
//! for any composite Holon — direct (structural; children composed, sub-parts
//! recoverable via unbind) or atomized wrap (opaque; EDN-hashed, structure not
//! recoverable from the vector). Applications pick per use case (058-001).
//!
//! `:Any` is NOT a wat-level type. The grammar refuses it; the type checker
//! refuses it. `dyn Any` is Rust substrate plumbing, never wat-visible.
//!
//! # Keywords
//!
//! Rust has no symbol type. Wat keywords (`:foo`, `:wat/std/cos-basis`) are
//! represented as `String` values whose content begins with `:`. The leading
//! `:` is part of the canonical bytes, so `HolonAST::keyword("foo")` and
//! `HolonAST::atom("foo".to_string())` produce different vectors — the byte
//! layouts differ. Use [`HolonAST::keyword`] to construct keyword atoms; the
//! wat-vm parser will produce them from source like `:foo`. String atoms use
//! [`HolonAST::atom`] with a `String` that does not begin with `:`.

use super::atom_registry::AtomTypeRegistry;
use super::primitives::Primitives;
use super::scalar::{ScalarEncoder, ScalarMode};
use super::vector::Vector;
use super::vector_manager::VectorManager;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// The universal AST for the 6 algebra core forms.
///
/// `Clone` is derived — all variants are `Arc`-wrapped, so clone is
/// O(1) refcount-increment regardless of tree depth.
#[derive(Clone)]
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

impl fmt::Debug for HolonAST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolonAST::Atom(_) => {
                // Payload type can't be named without a registry lookup — keep
                // the variant identifiable without exposing the opaque payload.
                write!(f, "Atom(<opaque>)")
            }
            HolonAST::Bind(a, b) => f.debug_tuple("Bind").field(a).field(b).finish(),
            HolonAST::Bundle(children) => f.debug_tuple("Bundle").field(children).finish(),
            HolonAST::Permute(child, k) => {
                f.debug_tuple("Permute").field(child).field(k).finish()
            }
            HolonAST::Thermometer { value, min, max } => f
                .debug_struct("Thermometer")
                .field("value", value)
                .field("min", min)
                .field("max", max)
                .finish(),
            HolonAST::Blend(a, b, w1, w2) => f
                .debug_tuple("Blend")
                .field(a)
                .field(b)
                .field(w1)
                .field(w2)
                .finish(),
        }
    }
}

impl HolonAST {
    /// Construct an Atom from any Rust type that is `Any + Send + Sync`.
    ///
    /// Canonicalization during encoding dispatches on the registry. Using a
    /// type that isn't registered panics at encode time with a clear message.
    pub fn atom<T: Any + Send + Sync + 'static>(value: T) -> Self {
        HolonAST::Atom(Arc::new(value))
    }

    /// Construct an Atom wrapping a wat keyword.
    ///
    /// Rust has no symbol type; wat keywords are represented at the Rust
    /// level as `String` values whose content begins with `:`. The leading
    /// `:` is part of the stored bytes and the hash input, so
    /// `HolonAST::keyword("foo")` (stored as `":foo"`) produces a different
    /// vector from `HolonAST::atom("foo".to_string())` (stored as `"foo"`).
    ///
    /// The input may or may not already carry the leading `:`:
    ///
    /// ```
    /// use holon::HolonAST;
    /// // All three create the same keyword atom — leading colon enforced:
    /// let k1 = HolonAST::keyword("foo/bar");
    /// let k2 = HolonAST::keyword(":foo/bar");
    /// // k1 and k2 hold identical String payloads ":foo/bar".
    /// ```
    ///
    /// This is the storage convention the wat-vm parser will produce: source
    /// `:foo/bar` → `HolonAST::keyword("foo/bar")`; source `"foo/bar"` →
    /// `HolonAST::atom("foo/bar".to_string())`. Different content → different
    /// canonical bytes → different vectors, no collision.
    pub fn keyword(name: &str) -> Self {
        let stored = if name.starts_with(':') {
            name.to_string()
        } else {
            let mut s = String::with_capacity(name.len() + 1);
            s.push(':');
            s.push_str(name);
            s
        };
        HolonAST::atom(stored)
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

// Variant tags for canonical-EDN serialization. Distinct bytes per variant so
// two different variants with identical inner bytes cannot collide.
const TAG_ATOM: u8 = 0x01;
const TAG_BIND: u8 = 0x02;
const TAG_BUNDLE: u8 = 0x03;
const TAG_PERMUTE: u8 = 0x04;
const TAG_THERMOMETER: u8 = 0x05;
const TAG_BLEND: u8 = 0x06;

/// Canonical-EDN bytes for a HolonAST.
///
/// Deterministic across all constructions of the same AST shape — same
/// children, same weights, same atom payloads all produce the same bytes.
/// Two different ASTs produce different bytes.
///
/// This is the hash input for `Atom(Arc<HolonAST>)` — programs-as-atoms. It
/// is also the substrate for cryptographic-provenance hashing per FOUNDATION's
/// "The Algebra Is Immutable" section (EDN is the transport form; the hash IS
/// the holon's identity).
pub fn canonical_edn_holon(ast: &HolonAST, registry: &AtomTypeRegistry) -> Vec<u8> {
    let mut out = Vec::new();
    match ast {
        HolonAST::Atom(payload) => {
            out.push(TAG_ATOM);
            let tag = registry.type_tag(payload.as_ref());
            // Length-prefixed tag so a tag name can contain the null byte
            // without breaking parsing (defensive; tags in practice are ASCII).
            out.extend_from_slice(&(tag.len() as u32).to_le_bytes());
            out.extend_from_slice(tag.as_bytes());
            let bytes = registry.canonical_bytes(payload.as_ref());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        HolonAST::Bind(a, b) => {
            out.push(TAG_BIND);
            let a_bytes = canonical_edn_holon(a, registry);
            let b_bytes = canonical_edn_holon(b, registry);
            out.extend_from_slice(&(a_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&a_bytes);
            out.extend_from_slice(&(b_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&b_bytes);
        }
        HolonAST::Bundle(children) => {
            out.push(TAG_BUNDLE);
            out.extend_from_slice(&(children.len() as u32).to_le_bytes());
            for c in children.iter() {
                let c_bytes = canonical_edn_holon(c, registry);
                out.extend_from_slice(&(c_bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(&c_bytes);
            }
        }
        HolonAST::Permute(child, k) => {
            out.push(TAG_PERMUTE);
            let c_bytes = canonical_edn_holon(child, registry);
            out.extend_from_slice(&(c_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&c_bytes);
            out.extend_from_slice(&k.to_le_bytes());
        }
        HolonAST::Thermometer { value, min, max } => {
            out.push(TAG_THERMOMETER);
            out.extend_from_slice(&value.to_le_bytes());
            out.extend_from_slice(&min.to_le_bytes());
            out.extend_from_slice(&max.to_le_bytes());
        }
        HolonAST::Blend(a, b, w1, w2) => {
            out.push(TAG_BLEND);
            let a_bytes = canonical_edn_holon(a, registry);
            let b_bytes = canonical_edn_holon(b, registry);
            out.extend_from_slice(&(a_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&a_bytes);
            out.extend_from_slice(&(b_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&b_bytes);
            out.extend_from_slice(&w1.to_le_bytes());
            out.extend_from_slice(&w2.to_le_bytes());
        }
    }
    out
}

/// Hash an Atom's payload to a seed for vector generation.
///
/// Seed input: `(type_tag, canonical bytes, global seed)`. Different types
/// with identical bytes produce different seeds.
fn atom_seed(
    value: &(dyn Any + Send + Sync),
    registry: &AtomTypeRegistry,
    global_seed: u64,
) -> u64 {
    let tag = registry.type_tag(value);
    let bytes = registry.canonical_bytes(value);

    let mut hasher = Sha256::new();
    hasher.update(global_seed.to_le_bytes());
    hasher.update(tag.as_bytes());
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(&bytes);
    let hash = hasher.finalize();

    u64::from_le_bytes(hash[0..8].try_into().unwrap())
}

/// Realize a HolonAST into a Vector by walking the AST and dispatching to
/// the lowercase Rust primitives.
///
/// Encoding is deterministic: same AST ⇒ same vector. No caching in this slice
/// (L1/L2 cache lands later per FOUNDATION's "Caching Is Memoization" section).
pub fn encode(
    ast: &HolonAST,
    vm: &VectorManager,
    scalar: &ScalarEncoder,
    registry: &AtomTypeRegistry,
) -> Vector {
    match ast {
        HolonAST::Atom(payload) => {
            let seed = atom_seed(payload.as_ref(), registry, vm.global_seed());
            let dims = vm.dimensions();
            deterministic_vector_from_seed(seed, dims)
        }
        HolonAST::Bind(a, b) => {
            let va = encode(a, vm, scalar, registry);
            let vb = encode(b, vm, scalar, registry);
            Primitives::bind(&va, &vb)
        }
        HolonAST::Bundle(children) => {
            let vectors: Vec<Vector> = children
                .iter()
                .map(|c| encode(c, vm, scalar, registry))
                .collect();
            let refs: Vec<&Vector> = vectors.iter().collect();
            Primitives::bundle(&refs)
        }
        HolonAST::Permute(child, k) => {
            let vc = encode(child, vm, scalar, registry);
            Primitives::permute(&vc, *k)
        }
        HolonAST::Thermometer { value, min, max } => scalar.encode(
            *value,
            ScalarMode::Thermometer {
                min: *min,
                max: *max,
            },
        ),
        HolonAST::Blend(a, b, w1, w2) => {
            let va = encode(a, vm, scalar, registry);
            let vb = encode(b, vm, scalar, registry);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Similarity;

    const D: usize = 1024;

    fn fresh_env() -> (VectorManager, ScalarEncoder, AtomTypeRegistry) {
        (
            VectorManager::with_seed(D, 42),
            ScalarEncoder::with_seed(D, 42),
            AtomTypeRegistry::with_builtins(),
        )
    }

    #[test]
    fn atom_type_discrimination_int_vs_string() {
        let (vm, se, reg) = fresh_env();
        let v_int = encode(&HolonAST::atom(42_i64), &vm, &se, &reg);
        let v_str = encode(&HolonAST::atom("42".to_string()), &vm, &se, &reg);
        assert_ne!(v_int, v_str, "Atom(42_i64) must differ from Atom(\"42\")");
    }

    #[test]
    fn atom_type_discrimination_int_vs_float() {
        let (vm, se, reg) = fresh_env();
        let v_int = encode(&HolonAST::atom(42_i64), &vm, &se, &reg);
        let v_float = encode(&HolonAST::atom(42.0_f64), &vm, &se, &reg);
        assert_ne!(v_int, v_float);
    }

    #[test]
    fn atom_type_discrimination_int_widths() {
        // i32 and i64 with the same numeric value produce different vectors.
        let (vm, se, reg) = fresh_env();
        let v_i32 = encode(&HolonAST::atom(42_i32), &vm, &se, &reg);
        let v_i64 = encode(&HolonAST::atom(42_i64), &vm, &se, &reg);
        assert_ne!(v_i32, v_i64);
    }

    #[test]
    fn keyword_vs_string_discrimination() {
        // Wat keywords are stored as Strings with leading `:`; wat strings are
        // Strings without. Content differs → hash differs → vectors differ.
        let (vm, se, reg) = fresh_env();
        let v_kw = encode(&HolonAST::keyword("foo/bar"), &vm, &se, &reg);
        let v_str = encode(&HolonAST::atom("foo/bar".to_string()), &vm, &se, &reg);
        assert_ne!(
            v_kw, v_str,
            "Keyword :foo/bar must differ from string \"foo/bar\" — \
             the leading `:` is part of the stored bytes"
        );
    }

    #[test]
    fn keyword_normalization() {
        // HolonAST::keyword accepts the name with or without the leading `:`
        // and always stores with the colon. Both spellings produce the same atom.
        let (vm, se, reg) = fresh_env();
        let v_no_colon = encode(&HolonAST::keyword("foo"), &vm, &se, &reg);
        let v_with_colon = encode(&HolonAST::keyword(":foo"), &vm, &se, &reg);
        assert_eq!(
            v_no_colon, v_with_colon,
            "HolonAST::keyword must normalize the leading colon"
        );
    }

    #[test]
    fn keyword_vs_prefixed_string() {
        // Atom(":foo".to_string()) and HolonAST::keyword("foo") both store
        // ":foo" as a String — they are the same atom, same vector. This is
        // the parser's contract: wat source `:foo` produces the colon-prefixed
        // String regardless of which helper constructs it.
        let (vm, se, reg) = fresh_env();
        let v_kw = encode(&HolonAST::keyword("foo"), &vm, &se, &reg);
        let v_direct = encode(&HolonAST::atom(":foo".to_string()), &vm, &se, &reg);
        assert_eq!(v_kw, v_direct);
    }

    #[test]
    fn atom_type_discrimination_bool_vs_int() {
        let (vm, se, reg) = fresh_env();
        let v_bool = encode(&HolonAST::atom(true), &vm, &se, &reg);
        let v_int = encode(&HolonAST::atom(1_i64), &vm, &se, &reg);
        assert_ne!(v_bool, v_int);
    }

    #[test]
    fn atom_deterministic() {
        let (vm1, se1, reg1) = fresh_env();
        let (vm2, se2, reg2) = fresh_env();
        let v1 = encode(&HolonAST::atom("hello".to_string()), &vm1, &se1, &reg1);
        let v2 = encode(&HolonAST::atom("hello".to_string()), &vm2, &se2, &reg2);
        assert_eq!(v1, v2);
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
        let (vm, se, reg) = fresh_env();
        let a = HolonAST::atom("role".to_string());
        let b = HolonAST::atom("filler".to_string());
        let v_bound = encode(&HolonAST::bind(a, b), &vm, &se, &reg);
        assert_eq!(v_bound.dimensions(), D);
    }

    #[test]
    fn bundle_list_form() {
        let (vm, se, reg) = fresh_env();
        let children = vec![
            HolonAST::atom("a".to_string()),
            HolonAST::atom("b".to_string()),
            HolonAST::atom("c".to_string()),
        ];
        let v = encode(&HolonAST::bundle(children), &vm, &se, &reg);
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    fn permute_by_zero_is_identity() {
        let (vm, se, reg) = fresh_env();
        let a = HolonAST::atom("x".to_string());
        let v_original = encode(&a, &vm, &se, &reg);
        let v_permuted = encode(&HolonAST::permute(a, 0), &vm, &se, &reg);
        assert_eq!(v_original, v_permuted);
    }

    #[test]
    fn permute_is_invertible() {
        let (vm, se, reg) = fresh_env();
        let a = HolonAST::atom("x".to_string());
        let forward = HolonAST::permute(HolonAST::atom("x".to_string()), 7);
        let round_trip = HolonAST::permute(forward, -7);
        let v_original = encode(&a, &vm, &se, &reg);
        let v_round_trip = encode(&round_trip, &vm, &se, &reg);
        assert_eq!(v_original, v_round_trip);
    }

    #[test]
    fn thermometer_endpoints() {
        let (vm, se, reg) = fresh_env();
        let v_min = encode(&HolonAST::thermometer(0.0, 0.0, 100.0), &vm, &se, &reg);
        let v_max = encode(&HolonAST::thermometer(100.0, 0.0, 100.0), &vm, &se, &reg);
        let sim = Similarity::cosine(&v_min, &v_max);
        assert!(sim < -0.99, "cosine ≈ -1 expected, got {}", sim);
    }

    #[test]
    fn blend_option_b_subtract() {
        let (vm, se, reg) = fresh_env();
        let x = HolonAST::atom("x".to_string());
        let y = HolonAST::atom("y".to_string());
        let v_sub = encode(&HolonAST::blend(x, y, 1.0, -1.0), &vm, &se, &reg);
        assert_eq!(v_sub.dimensions(), D);
    }

    #[test]
    fn blend_option_b_circular_weights() {
        let (vm, se, reg) = fresh_env();
        let cos_basis = HolonAST::atom("wat/std/circular-cos-basis".to_string());
        let sin_basis = HolonAST::atom("wat/std/circular-sin-basis".to_string());
        let theta = std::f64::consts::FRAC_PI_4;
        let v = encode(
            &HolonAST::blend(cos_basis, sin_basis, theta.cos(), theta.sin()),
            &vm,
            &se,
            &reg,
        );
        assert_eq!(v.dimensions(), D);
    }

    // ─── Programs-as-atoms tests — the substrate commit of 058-001 ──────────

    #[test]
    fn atom_holon_encodes() {
        // Atom wrapping a HolonAST encodes without panic.
        let (vm, se, reg) = fresh_env();
        let inner = HolonAST::bundle(vec![
            HolonAST::atom("a".to_string()),
            HolonAST::atom("b".to_string()),
        ]);
        let atomized = HolonAST::atom(inner);
        let v = encode(&atomized, &vm, &se, &reg);
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    fn atom_holon_identical_programs_same_vector() {
        // Two Atoms wrapping structurally-identical Holons produce identical vectors.
        let (vm, se, reg) = fresh_env();
        let make = || {
            HolonAST::bundle(vec![
                HolonAST::atom(42_i64),
                HolonAST::atom("rsi".to_string()),
            ])
        };
        let v1 = encode(&HolonAST::atom(make()), &vm, &se, &reg);
        let v2 = encode(&HolonAST::atom(make()), &vm, &se, &reg);
        assert_eq!(v1, v2);
    }

    #[test]
    fn atom_holon_different_programs_different_vectors() {
        // Two Atoms wrapping structurally-different Holons produce different vectors.
        let (vm, se, reg) = fresh_env();
        let prog_a = HolonAST::bundle(vec![
            HolonAST::atom(42_i64),
            HolonAST::atom("rsi".to_string()),
        ]);
        let prog_b = HolonAST::bundle(vec![
            HolonAST::atom(43_i64),
            HolonAST::atom("rsi".to_string()),
        ]);
        let v_a = encode(&HolonAST::atom(prog_a), &vm, &se, &reg);
        let v_b = encode(&HolonAST::atom(prog_b), &vm, &se, &reg);
        assert_ne!(v_a, v_b);
    }

    #[test]
    fn atom_holon_differs_from_direct_encoding() {
        // 058-001: two legitimate encodings — direct (structural) and atomized (opaque).
        // They must produce DIFFERENT vectors to represent the two different framings.
        let (vm, se, reg) = fresh_env();
        let make = || {
            HolonAST::bundle(vec![
                HolonAST::atom("x".to_string()),
                HolonAST::atom("y".to_string()),
            ])
        };
        let v_direct = encode(&make(), &vm, &se, &reg);
        let v_atomized = encode(&HolonAST::atom(make()), &vm, &se, &reg);
        assert_ne!(
            v_direct, v_atomized,
            "Direct encoding (structural) and Atom wrapping (opaque identity) \
             must produce different vectors — 058-001 two-encodings principle"
        );
    }

    #[test]
    fn user_type_registration_path() {
        // Applications register their own types before atomizing them.
        #[derive(Debug)]
        struct Candle {
            open: f64,
            close: f64,
        }

        let mut reg = AtomTypeRegistry::with_builtins();
        reg.register::<Candle>("project/market/Candle", |v, _| {
            let mut out = Vec::new();
            out.extend_from_slice(&v.open.to_le_bytes());
            out.extend_from_slice(&v.close.to_le_bytes());
            out
        });

        let vm = VectorManager::with_seed(D, 42);
        let se = ScalarEncoder::with_seed(D, 42);

        let c1 = Candle {
            open: 100.0,
            close: 101.0,
        };
        let c2 = Candle {
            open: 100.0,
            close: 101.0,
        };
        let c3 = Candle {
            open: 100.0,
            close: 102.0,
        };

        let v1 = encode(&HolonAST::atom(c1), &vm, &se, &reg);
        let v2 = encode(&HolonAST::atom(c2), &vm, &se, &reg);
        let v3 = encode(&HolonAST::atom(c3), &vm, &se, &reg);

        assert_eq!(v1, v2, "identical Candles produce identical vectors");
        assert_ne!(v1, v3, "different Candles produce different vectors");
    }

    #[test]
    #[should_panic(expected = "unregistered type")]
    fn unregistered_type_panics() {
        #[derive(Debug)]
        struct Unknown(#[allow(dead_code)] i64);
        let (vm, se, reg) = fresh_env();
        // Unknown was not registered; encode must panic with a clear message.
        let _ = encode(&HolonAST::atom(Unknown(1)), &vm, &se, &reg);
    }

    // ─── Canonical-EDN tests ────────────────────────────────────────────────

    #[test]
    fn canonical_edn_deterministic() {
        let reg = AtomTypeRegistry::with_builtins();
        let make = || {
            HolonAST::bundle(vec![
                HolonAST::atom(42_i64),
                HolonAST::atom("rsi".to_string()),
            ])
        };
        assert_eq!(
            canonical_edn_holon(&make(), &reg),
            canonical_edn_holon(&make(), &reg)
        );
    }

    #[test]
    fn canonical_edn_variants_distinguished() {
        // Two different variants with similar inner structure must produce
        // different bytes.
        let reg = AtomTypeRegistry::with_builtins();
        let a = HolonAST::atom(1_i64);
        let b = HolonAST::atom(2_i64);
        let bound = HolonAST::bind(
            HolonAST::atom(1_i64),
            HolonAST::atom(2_i64),
        );
        let bundled = HolonAST::bundle(vec![a, b]);
        assert_ne!(
            canonical_edn_holon(&bound, &reg),
            canonical_edn_holon(&bundled, &reg),
            "Bind and Bundle of the same atoms must canonicalize differently"
        );
    }
}
