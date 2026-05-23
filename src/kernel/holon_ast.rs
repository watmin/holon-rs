//! HolonAST — the universal AST for the wat algebra. Closed under itself.
//!
//! Per arc 057 (typed HolonAST leaves): primitives ARE HolonAST. The number
//! 42 is the simplest possible AST — a leaf. So is `true`, so is `"foo"`,
//! so is `:outcome`. They have well-defined canonical encodings; they are
//! terms in the algebra; they are HolonAST.
//!
//! **Twelve true primitives** (arc 230 — typed-entities doctrine):
//!
//! - **Holder** (arc 225): `Atom(Arc<HolonAST>)`. The algebra's quote —
//!   minimal holder, repeatable holds compose. Wraps any HolonAST as an
//!   opaque-identity unit: the resulting vector hashes the canonical-EDN
//!   bytes of the wrapped AST as a single seed instead of recursively
//!   encoding sub-parts.
//! - **Composers**: `Bind`, `Bundle`, `Permute`, `Thermometer`, `Blend`.
//!   Similarity-preserving recursive encoding. Sub-parts recoverable via
//!   `unbind`.
//! - **Raw carriers**: `String`, `I64`, `F64`, `Bool`, `Char`. Irreducible
//!   content the encoder hashes to a deterministic identity vector.
//! - **Substrate-internal sentinel** (arc 073): `SlotMarker { min, max }`.
//!   The placeholder a `term::template` operation produces in place of a
//!   Thermometer's `value`. User-unconstructible at the wat surface; encoder
//!   panics on it (templates are query keys, not encodable values).
//!
//! **Retired convenience variants** (arc 230 supersession):
//! `Symbol`, `Keyword`, `Tag`, `Nil` were convenience shortcuts over
//! `Bind(Atom, Atom)` compositions. Per the typed-entities doctrine every
//! typed value at user-surface compiles to `(Bind (Atom class) (Atom data))`.
//! The constructor helpers (`symbol()`, `keyword()`, `tag()`, `nil()`)
//! are preserved API-surface — they now produce the Bind composition.
//! The accessor helpers (`as_symbol()`, `as_keyword()`, `as_tag()`,
//! `is_nil()`) are preserved — they now recognise the Bind composition.
//!
//! The algebra is closed: every term in `HolonAST` is itself `HolonAST`.
//! `Hash + Eq` derive directly (manual impls only because f64 fields use
//! `to_bits` per the standard NaN-Hash dance). Cache keys, engram
//! libraries, persistence, and cross-process AST handoff all work because
//! a HolonAST has structural identity.

use super::primitives::Primitives;
use super::scalar::{ScalarEncoder, ScalarMode};
use super::vector::Vector;
use super::vector_manager::VectorManager;
use sha2::{Digest, Sha256};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The universal AST. Twelve true primitives. Closed under itself.
///
/// Arc 230 (typed-entities doctrine): the four convenience variants
/// `Symbol`, `Keyword`, `Tag`, `Nil` are retired. Their constructor helpers
/// (`symbol()`, `keyword()`, `tag()`, `nil()`) now produce pure
/// `Bind(Atom, Atom)` compositions. Accessors (`as_symbol()`,
/// `as_keyword()`, `as_tag()`, `is_nil()`) recognise the compositions.
///
/// `Clone` is O(1) — every recursive payload is `Arc`-wrapped.
#[derive(Clone)]
pub enum HolonAST {
    // ─── Raw carriers ───────────────────────────────────────────────────
    /// String literal content. Stored bytes are exactly the string.
    String(Arc<str>),

    /// 64-bit signed integer leaf.
    I64(i64),

    /// 64-bit float leaf. Hash uses `to_bits` (NaN-safe per standard
    /// HashMap-with-f64 dance).
    F64(f64),

    /// Boolean leaf.
    Bool(bool),

    /// Char leaf — single Unicode scalar value. EDN-literal form `\a`,
    /// `\newline`, `\u{NNNN}` per Clojure-EDN spec. BMP-only is a wat-rs
    /// surface concern (arc 220 Stone 220.2); holon-rs accepts full `char`.
    Char(char),

    // ─── Opaque-identity wrap ───────────────────────────────────────────
    /// Wrap a HolonAST as an opaque-identity unit. The wrapped AST's
    /// canonical bytes feed a single SHA-256; the resulting vector
    /// represents "this program as one identity," distinct from the
    /// structural vector of the unwrapped form.
    ///
    /// `Atom(Atom(x))` differs from `Atom(x)` differs from `x` —
    /// quote-wrapping is repeatable and meaningful (Lisp's `'(quote x)`
    /// ≠ `'x`).
    Atom(Arc<HolonAST>),

    // ─── Composites ─────────────────────────────────────────────────────
    /// `Bind(a, b)` — elementwise multiplication; MAP's "M".
    Bind(Arc<HolonAST>, Arc<HolonAST>),

    /// `Bundle(xs)` — elementwise sum + ternary threshold; MAP's "A".
    Bundle(Arc<Vec<HolonAST>>),

    /// `Permute(child, k)` — cyclic shift; MAP's "P".
    /// `Permute(v, k)[i] = v[(i + k) mod d]`.
    Permute(Arc<HolonAST>, i32),

    /// `Thermometer(value, min, max)` — gradient encoding. First
    /// `N = round(d · clamp((value-min)/(max-min), 0, 1))` dims are `+1`,
    /// remaining `d - N` dims are `-1`.
    Thermometer { value: f64, min: f64, max: f64 },

    /// `Blend(a, b, w1, w2)` — `threshold(w1·a + w2·b)`. Two independent
    /// real-valued weights; negative allowed (058-002 Option B).
    Blend(Arc<HolonAST>, Arc<HolonAST>, f64, f64),

    // ─── Substrate-internal sentinel (arc 073) ──────────────────────────
    /// `SlotMarker { min, max }` — placeholder for a Thermometer's `value`
    /// in a `term::template` output. The receptive field (min, max) is
    /// preserved (templates with different ranges are distinct cell
    /// types); the `value` is discarded (templates with same range and
    /// different values share a cell type, differing only in tuning).
    ///
    /// Not user-constructible at the wat surface. Returned by
    /// `:wat::holon::term::template`; consumed by `TermStore::get`'s
    /// template-equality check. Encoder panics on it — templates are
    /// query keys, not encodable values; reaching the encoder with a
    /// SlotMarker is a category error and surfacing it loudly beats
    /// producing a silent zero vector.
    SlotMarker { min: f64, max: f64 },
}

impl fmt::Debug for HolonAST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolonAST::String(s) => f.debug_tuple("String").field(&&**s).finish(),
            HolonAST::I64(n) => f.debug_tuple("I64").field(n).finish(),
            HolonAST::F64(x) => f.debug_tuple("F64").field(x).finish(),
            HolonAST::Bool(b) => f.debug_tuple("Bool").field(b).finish(),
            HolonAST::Char(c) => f.debug_tuple("Char").field(c).finish(),
            HolonAST::Atom(h) => f.debug_tuple("Atom").field(h).finish(),
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
            HolonAST::SlotMarker { min, max } => f
                .debug_struct("SlotMarker")
                .field("min", min)
                .field("max", max)
                .finish(),
        }
    }
}

impl PartialEq for HolonAST {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (HolonAST::String(a), HolonAST::String(b)) => a == b,
            (HolonAST::I64(a), HolonAST::I64(b)) => a == b,
            (HolonAST::F64(a), HolonAST::F64(b)) => a.to_bits() == b.to_bits(),
            (HolonAST::Bool(a), HolonAST::Bool(b)) => a == b,
            (HolonAST::Char(a), HolonAST::Char(b)) => a == b,
            (HolonAST::Atom(a), HolonAST::Atom(b)) => a == b,
            (HolonAST::Bind(a1, b1), HolonAST::Bind(a2, b2)) => a1 == a2 && b1 == b2,
            (HolonAST::Bundle(xs), HolonAST::Bundle(ys)) => xs == ys,
            (HolonAST::Permute(a, k1), HolonAST::Permute(b, k2)) => a == b && k1 == k2,
            (
                HolonAST::Thermometer {
                    value: v1,
                    min: m1,
                    max: x1,
                },
                HolonAST::Thermometer {
                    value: v2,
                    min: m2,
                    max: x2,
                },
            ) => v1.to_bits() == v2.to_bits() && m1.to_bits() == m2.to_bits() && x1.to_bits() == x2.to_bits(),
            (HolonAST::Blend(a1, b1, w1a, w2a), HolonAST::Blend(a2, b2, w1b, w2b)) => {
                a1 == a2 && b1 == b2 && w1a.to_bits() == w1b.to_bits() && w2a.to_bits() == w2b.to_bits()
            }
            (
                HolonAST::SlotMarker { min: m1, max: x1 },
                HolonAST::SlotMarker { min: m2, max: x2 },
            ) => m1.to_bits() == m2.to_bits() && x1.to_bits() == x2.to_bits(),
            _ => false,
        }
    }
}

impl Eq for HolonAST {}

impl Hash for HolonAST {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            HolonAST::String(s) => s.hash(state),
            HolonAST::I64(n) => n.hash(state),
            HolonAST::F64(x) => x.to_bits().hash(state),
            HolonAST::Bool(b) => b.hash(state),
            HolonAST::Char(c) => (*c as u32).hash(state),
            HolonAST::Atom(h) => h.hash(state),
            HolonAST::Bind(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            HolonAST::Bundle(xs) => xs.hash(state),
            HolonAST::Permute(a, k) => {
                a.hash(state);
                k.hash(state);
            }
            HolonAST::Thermometer { value, min, max } => {
                value.to_bits().hash(state);
                min.to_bits().hash(state);
                max.to_bits().hash(state);
            }
            HolonAST::Blend(a, b, w1, w2) => {
                a.hash(state);
                b.hash(state);
                w1.to_bits().hash(state);
                w2.to_bits().hash(state);
            }
            HolonAST::SlotMarker { min, max } => {
                min.to_bits().hash(state);
                max.to_bits().hash(state);
            }
        }
    }
}

impl HolonAST {
    // ─── Internal helper: build a classified Bind composition ───────────
    //
    // `Bind(Atom(String(classifier)), Atom(String(content)))` is the
    // canonical encoding for Symbol / Keyword / Tag / Nil per arc 230
    // (typed-entities doctrine). All four retired variant constructors
    // and all four accessors go through this shape.
    fn classified(classifier: &str, content: &str) -> Self {
        HolonAST::Bind(
            Arc::new(HolonAST::Atom(Arc::new(HolonAST::String(Arc::from(classifier))))),
            Arc::new(HolonAST::Atom(Arc::new(HolonAST::String(Arc::from(content))))),
        )
    }

    // Extract `content` from `Bind(Atom(String(expected_cls)), Atom(String(content)))`.
    // Returns None if the shape doesn't match or the classifier differs.
    fn extract_classified<'a>(h: &'a HolonAST, expected_cls: &str) -> Option<&'a str> {
        match h {
            HolonAST::Bind(a, b) => match (a.as_ref(), b.as_ref()) {
                (HolonAST::Atom(ac), HolonAST::Atom(bc)) => {
                    match (ac.as_ref(), bc.as_ref()) {
                        (HolonAST::String(cls), HolonAST::String(val))
                            if cls.as_ref() == expected_cls =>
                        {
                            Some(val.as_ref())
                        }
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Construct a Symbol composition — arc 230 supersession of `HolonAST::Symbol(s)`.
    ///
    /// Returns `Bind(Atom(String("Symbol")), Atom(String(content)))`.
    /// The `Symbol` variant is retired (arc 230); this helper preserves the
    /// constructor API while producing the pure Bind composition per the
    /// typed-entities doctrine.
    pub fn symbol(content: impl Into<Arc<str>>) -> Self {
        let s: Arc<str> = content.into();
        HolonAST::classified("Symbol", s.as_ref())
    }

    /// Construct a `String` leaf.
    pub fn string(content: impl Into<Arc<str>>) -> Self {
        HolonAST::String(content.into())
    }

    /// Construct an `I64` leaf.
    pub fn i64(n: i64) -> Self {
        HolonAST::I64(n)
    }

    /// Construct an `F64` leaf.
    pub fn f64(x: f64) -> Self {
        HolonAST::F64(x)
    }

    /// Construct a `Bool` leaf.
    pub fn bool_(b: bool) -> Self {
        HolonAST::Bool(b)
    }

    /// Construct a `Char` leaf from a Rust `char`. The substrate accepts
    /// full Unicode; BMP-only enforcement is a wat-rs surface concern.
    pub fn char_(c: char) -> Self {
        HolonAST::Char(c)
    }

    /// Construct a Keyword composition — arc 230 supersession of `HolonAST::Keyword(s)`.
    ///
    /// Returns `Bind(Atom(String("Keyword")), Atom(String(name_without_colon)))`.
    /// `HolonAST::keyword("foo")` and `HolonAST::keyword(":foo")` produce
    /// the same composition (leading colon stripped).
    ///
    /// Arc 221 minted `HolonAST::Keyword`; arc 230 supersedes with Bind composition
    /// per the typed-entities doctrine.
    pub fn keyword(name: &str) -> Self {
        let stored = if let Some(stripped) = name.strip_prefix(':') {
            stripped
        } else {
            name
        };
        HolonAST::classified("Keyword", stored)
    }

    /// Construct a Nil composition — arc 230 supersession of `HolonAST::Nil`.
    ///
    /// Returns `Bind(Atom(String("Symbol")), Atom(String("nil")))`.
    /// Per user 2026-05-22 articulation: nil encodes as Symbol("nil") composition.
    ///
    /// Arc 221 minted `HolonAST::Nil`; arc 230 supersedes with Bind composition.
    pub fn nil() -> Self {
        HolonAST::classified("Symbol", "nil")
    }

    /// Construct a Tag composition — arc 230 supersession of `HolonAST::Tag(s)`.
    ///
    /// Returns `Bind(Atom(String("Tag")), Atom(String(name_without_hash)))`.
    /// `HolonAST::tag("uuid")` and `HolonAST::tag("#uuid")` produce the same
    /// composition (leading `#` stripped).
    ///
    /// Arc 221 minted `HolonAST::Tag`; arc 230 supersedes with Bind composition.
    pub fn tag(name: &str) -> Self {
        let stored = if let Some(stripped) = name.strip_prefix('#') {
            stripped
        } else {
            name
        };
        HolonAST::classified("Tag", stored)
    }

    /// Wrap a HolonAST as an opaque-identity Atom. See the [`HolonAST::Atom`]
    /// variant docs for the algebraic significance of the wrap.
    pub fn atom(h: HolonAST) -> Self {
        HolonAST::Atom(Arc::new(h))
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

    // ─── Per-variant accessors (Option<T>) ──────────────────────────────
    //
    // Arc 230: Symbol/Keyword/Tag/Nil variants retired. Accessors now
    // recognise the Bind(Atom(String(classifier)), Atom(String(content)))
    // composition produced by the updated constructors.

    /// If this is a Symbol composition, return the content.
    ///
    /// Arc 230: recognises `Bind(Atom(String("Symbol")), Atom(String(s)))`.
    pub fn as_symbol(&self) -> Option<&str> {
        HolonAST::extract_classified(self, "Symbol")
    }

    /// If this is a `String` leaf, return the content.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            HolonAST::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    /// If this is an `I64` leaf, return the value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            HolonAST::I64(n) => Some(*n),
            _ => None,
        }
    }

    /// If this is an `F64` leaf, return the value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            HolonAST::F64(x) => Some(*x),
            _ => None,
        }
    }

    /// If this is a `Bool` leaf, return the value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HolonAST::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// If this is an `Atom`, return the wrapped HolonAST.
    pub fn atom_inner(&self) -> Option<&HolonAST> {
        match self {
            HolonAST::Atom(h) => Some(h.as_ref()),
            _ => None,
        }
    }

    /// If this is a Keyword composition, return the content (without leading colon).
    ///
    /// Arc 230: recognises `Bind(Atom(String("Keyword")), Atom(String(s)))`.
    pub fn as_keyword(&self) -> Option<&str> {
        HolonAST::extract_classified(self, "Keyword")
    }

    /// If this is a Tag composition, return the content (without leading `#`).
    ///
    /// Arc 230: recognises `Bind(Atom(String("Tag")), Atom(String(s)))`.
    pub fn as_tag(&self) -> Option<&str> {
        HolonAST::extract_classified(self, "Tag")
    }

    /// Returns true if this is the Nil composition.
    ///
    /// Arc 230: recognises `Bind(Atom(String("Symbol")), Atom(String("nil")))`.
    pub fn is_nil(&self) -> bool {
        HolonAST::extract_classified(self, "Symbol") == Some("nil")
    }

    // ─── Term decomposition (arc 073) ───────────────────────────────────

    /// Return this AST with every `Thermometer { value, min, max }` leaf
    /// replaced by `SlotMarker { min, max }`. Same structure otherwise.
    /// Two thoughts that differ only in Thermometer values produce
    /// identical templates; thoughts that differ in receptive fields
    /// (min/max) or in any other variant produce distinct templates.
    ///
    /// The output is itself a `HolonAST` (the algebra remains closed)
    /// but is INTENTIONALLY non-encodable: feeding a SlotMarker-bearing
    /// template to `encode` panics with a category-error message.
    pub fn template(&self) -> HolonAST {
        match self {
            HolonAST::String(_)
            | HolonAST::I64(_)
            | HolonAST::F64(_)
            | HolonAST::Bool(_)
            | HolonAST::Char(_) => self.clone(),
            HolonAST::Atom(inner) => HolonAST::Atom(Arc::new(inner.template())),
            HolonAST::Bind(a, b) => HolonAST::Bind(Arc::new(a.template()), Arc::new(b.template())),
            HolonAST::Bundle(xs) => {
                HolonAST::Bundle(Arc::new(xs.iter().map(|c| c.template()).collect()))
            }
            HolonAST::Permute(child, k) => HolonAST::Permute(Arc::new(child.template()), *k),
            HolonAST::Thermometer { min, max, .. } => HolonAST::SlotMarker {
                min: *min,
                max: *max,
            },
            HolonAST::Blend(a, b, w1, w2) => HolonAST::Blend(
                Arc::new(a.template()),
                Arc::new(b.template()),
                *w1,
                *w2,
            ),
            HolonAST::SlotMarker { min, max } => HolonAST::SlotMarker {
                min: *min,
                max: *max,
            },
        }
    }

    /// Pre-order list of every `Thermometer` value in this AST. Empty
    /// for forms with no Thermometer leaves. Parallel in length and
    /// order to `ranges`.
    pub fn slots(&self) -> Vec<f64> {
        let mut out = Vec::new();
        collect_slots(self, &mut out);
        out
    }

    /// Pre-order list of every `Thermometer (min, max)` pair in this
    /// AST. Empty for forms with no Thermometer leaves. Parallel in
    /// length and order to `slots`.
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        collect_ranges(self, &mut out);
        out
    }
}

fn collect_slots(ast: &HolonAST, out: &mut Vec<f64>) {
    match ast {
        HolonAST::String(_)
        | HolonAST::I64(_)
        | HolonAST::F64(_)
        | HolonAST::Bool(_)
        | HolonAST::Char(_)
        | HolonAST::SlotMarker { .. } => {}
        HolonAST::Atom(inner) => collect_slots(inner, out),
        HolonAST::Bind(a, b) => {
            collect_slots(a, out);
            collect_slots(b, out);
        }
        HolonAST::Bundle(xs) => {
            for c in xs.iter() {
                collect_slots(c, out);
            }
        }
        HolonAST::Permute(child, _) => collect_slots(child, out),
        HolonAST::Thermometer { value, .. } => out.push(*value),
        HolonAST::Blend(a, b, _, _) => {
            collect_slots(a, out);
            collect_slots(b, out);
        }
    }
}

fn collect_ranges(ast: &HolonAST, out: &mut Vec<(f64, f64)>) {
    match ast {
        HolonAST::String(_)
        | HolonAST::I64(_)
        | HolonAST::F64(_)
        | HolonAST::Bool(_)
        | HolonAST::Char(_)
        | HolonAST::SlotMarker { .. } => {}
        HolonAST::Atom(inner) => collect_ranges(inner, out),
        HolonAST::Bind(a, b) => {
            collect_ranges(a, out);
            collect_ranges(b, out);
        }
        HolonAST::Bundle(xs) => {
            for c in xs.iter() {
                collect_ranges(c, out);
            }
        }
        HolonAST::Permute(child, _) => collect_ranges(child, out),
        HolonAST::Thermometer { min, max, .. } => out.push((*min, *max)),
        HolonAST::Blend(a, b, _, _) => {
            collect_ranges(a, out);
            collect_ranges(b, out);
        }
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
const TAG_SLOT_MARKER: u8 = 0x07;

// Type tags for primitive leaves (the 8 true primitive raw carriers + Atom).
// Arc 230: PRIM_TAG_SYMBOL / PRIM_TAG_KEYWORD / PRIM_TAG_NIL / PRIM_TAG_TAG
// removed — those variants are retired. Symbol/Keyword/Tag/Nil now encode as
// Bind compositions; the canonical bytes come from the Bind/Atom/String
// structure without any PRIM_TAG seed of their own.
const PRIM_TAG_STRING: &str = "String";
const PRIM_TAG_I64: &str = "i64";
const PRIM_TAG_F64: &str = "f64";
const PRIM_TAG_BOOL: &str = "bool";
const PRIM_TAG_CHAR: &str = "char";
const ATOM_INNER_TAG: &str = "wat/algebra/Holon";

fn write_atom_payload(out: &mut Vec<u8>, type_tag: &str, payload: &[u8]) {
    out.push(TAG_ATOM);
    out.extend_from_slice(&(type_tag.len() as u32).to_le_bytes());
    out.extend_from_slice(type_tag.as_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}

/// Canonical-EDN bytes for a HolonAST.
///
/// Deterministic across all constructions of the same AST shape. Two
/// different ASTs produce different bytes. Same input → same bytes →
/// same vector seed.
///
/// Arc 230: Symbol/Keyword/Tag/Nil variants retired. Those forms now
/// encode as `Bind(Atom(String(cls)), Atom(String(val)))` compositions;
/// their canonical bytes come from the Bind/Atom/String structure naturally.
pub fn canonical_edn_holon(ast: &HolonAST) -> Vec<u8> {
    let mut out = Vec::new();
    match ast {
        // Primitive leaves use the `[TAG_ATOM, type_tag, payload]` shape.
        HolonAST::String(s) => write_atom_payload(&mut out, PRIM_TAG_STRING, s.as_bytes()),
        HolonAST::I64(n) => write_atom_payload(&mut out, PRIM_TAG_I64, &n.to_le_bytes()),
        HolonAST::F64(x) => write_atom_payload(&mut out, PRIM_TAG_F64, &x.to_le_bytes()),
        HolonAST::Bool(b) => write_atom_payload(&mut out, PRIM_TAG_BOOL, &[*b as u8]),
        HolonAST::Char(c) => write_atom_payload(&mut out, PRIM_TAG_CHAR, &(*c as u32).to_le_bytes()),
        HolonAST::Atom(inner) => {
            let inner_bytes = canonical_edn_holon(inner);
            write_atom_payload(&mut out, ATOM_INNER_TAG, &inner_bytes);
        }
        HolonAST::Bind(a, b) => {
            out.push(TAG_BIND);
            let a_bytes = canonical_edn_holon(a);
            let b_bytes = canonical_edn_holon(b);
            out.extend_from_slice(&(a_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&a_bytes);
            out.extend_from_slice(&(b_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&b_bytes);
        }
        HolonAST::Bundle(children) => {
            out.push(TAG_BUNDLE);
            out.extend_from_slice(&(children.len() as u32).to_le_bytes());
            for c in children.iter() {
                let c_bytes = canonical_edn_holon(c);
                out.extend_from_slice(&(c_bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(&c_bytes);
            }
        }
        HolonAST::Permute(child, k) => {
            out.push(TAG_PERMUTE);
            let c_bytes = canonical_edn_holon(child);
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
            let a_bytes = canonical_edn_holon(a);
            let b_bytes = canonical_edn_holon(b);
            out.extend_from_slice(&(a_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&a_bytes);
            out.extend_from_slice(&(b_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&b_bytes);
            out.extend_from_slice(&w1.to_le_bytes());
            out.extend_from_slice(&w2.to_le_bytes());
        }
        HolonAST::SlotMarker { min, max } => {
            out.push(TAG_SLOT_MARKER);
            out.extend_from_slice(&min.to_le_bytes());
            out.extend_from_slice(&max.to_le_bytes());
        }
    }
    out
}

/// Hash a leaf's canonical bytes to a vector seed.
fn leaf_seed(type_tag: &str, payload: &[u8], global_seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(global_seed.to_le_bytes());
    hasher.update(type_tag.as_bytes());
    hasher.update((payload.len() as u32).to_le_bytes());
    hasher.update(payload);
    let hash = hasher.finalize();
    u64::from_le_bytes(hash[0..8].try_into().unwrap())
}

/// Realize a HolonAST into a Vector by walking the AST and dispatching to
/// the lowercase Rust primitives.
///
/// Arc 230: Symbol/Keyword/Tag/Nil variants retired. Those forms now
/// encode as `Bind(Atom(String(cls)), Atom(String(val)))` compositions
/// which are handled by the `Bind` and `Atom` arms naturally — no
/// special PRIM_TAG seed needed. The structural Bind encoding is the
/// discriminator (STOP-8 check: Symbol("foo") vs Keyword("foo") differ
/// because their classifier atoms differ: "Symbol" vs "Keyword").
pub fn encode(ast: &HolonAST, vm: &VectorManager, scalar: &ScalarEncoder) -> Vector {
    match ast {
        HolonAST::String(s) => {
            let seed = leaf_seed(PRIM_TAG_STRING, s.as_bytes(), vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::I64(n) => {
            let seed = leaf_seed(PRIM_TAG_I64, &n.to_le_bytes(), vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::F64(x) => {
            let seed = leaf_seed(PRIM_TAG_F64, &x.to_le_bytes(), vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::Bool(b) => {
            let seed = leaf_seed(PRIM_TAG_BOOL, &[*b as u8], vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::Char(c) => {
            let seed = leaf_seed(PRIM_TAG_CHAR, &(*c as u32).to_le_bytes(), vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::Atom(inner) => {
            let inner_bytes = canonical_edn_holon(inner);
            let seed = leaf_seed(ATOM_INNER_TAG, &inner_bytes, vm.global_seed());
            deterministic_vector_from_seed(seed, vm.dimensions())
        }
        HolonAST::Bind(a, b) => {
            let va = encode(a, vm, scalar);
            let vb = encode(b, vm, scalar);
            Primitives::bind(&va, &vb)
        }
        HolonAST::Bundle(children) => {
            // Empty Bundle = the algebra's identity element (no
            // information). Materializes as a zero ternary vector so
            // structurally-lowered wat forms containing `()` survive
            // encoding without panicking.
            if children.is_empty() {
                Vector::zeros(vm.dimensions())
            } else {
                let vectors: Vec<Vector> =
                    children.iter().map(|c| encode(c, vm, scalar)).collect();
                let refs: Vec<&Vector> = vectors.iter().collect();
                Primitives::bundle(&refs)
            }
        }
        HolonAST::Permute(child, k) => {
            let vc = encode(child, vm, scalar);
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
            let va = encode(a, vm, scalar);
            let vb = encode(b, vm, scalar);
            Primitives::blend_weighted(&va, &vb, *w1, *w2)
        }
        HolonAST::SlotMarker { .. } => panic!(
            "encode: HolonAST::SlotMarker is a query-key sentinel, not an \
             encodable value. Templates produced by `term::template` are not \
             encodable; call `encode` on the original Thermometer-bearing form."
        ),
    }
}

/// Generate a deterministic ternary vector from a u64 seed.
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
    use std::collections::HashMap;

    const D: usize = 1024;

    fn fresh_env() -> (VectorManager, ScalarEncoder) {
        (
            VectorManager::with_seed(D, 42),
            ScalarEncoder::with_seed(D, 42),
        )
    }

    #[test]
    fn leaf_int_vs_string_distinct() {
        let (vm, se) = fresh_env();
        let v_int = encode(&HolonAST::i64(42), &vm, &se);
        let v_str = encode(&HolonAST::string("42"), &vm, &se);
        assert_ne!(v_int, v_str);
    }

    #[test]
    fn leaf_int_vs_float_distinct() {
        let (vm, se) = fresh_env();
        let v_int = encode(&HolonAST::i64(42), &vm, &se);
        let v_float = encode(&HolonAST::f64(42.0), &vm, &se);
        assert_ne!(v_int, v_float);
    }

    #[test]
    fn keyword_vs_string_distinct_by_content() {
        // Arc 230: keyword("foo/bar") = Bind(Atom("Keyword"), Atom("foo/bar")).
        // string("foo/bar") = raw String leaf. Structurally distinct.
        let (vm, se) = fresh_env();
        let v_kw = encode(&HolonAST::keyword("foo/bar"), &vm, &se);
        let v_str = encode(&HolonAST::string("foo/bar"), &vm, &se);
        assert_ne!(v_kw, v_str);
    }

    #[test]
    fn keyword_normalization() {
        // HolonAST::keyword normalizes the leading colon either way — both
        // produce Keyword("foo") with the same stored content.
        let (vm, se) = fresh_env();
        let v_no_colon = encode(&HolonAST::keyword("foo"), &vm, &se);
        let v_with_colon = encode(&HolonAST::keyword(":foo"), &vm, &se);
        assert_eq!(v_no_colon, v_with_colon);
    }

    #[test]
    fn keyword_distinct_from_symbol_at_type_level() {
        // Arc 230: keyword("foo") = Bind(Atom("Keyword"), Atom("foo"));
        // symbol(":foo") = Bind(Atom("Symbol"), Atom(":foo")).
        // Classifier atoms differ → distinct canonical bytes + vectors.
        let (vm, se) = fresh_env();
        let v_kw = encode(&HolonAST::keyword("foo"), &vm, &se);
        let v_sym = encode(&HolonAST::symbol(":foo"), &vm, &se);
        assert_ne!(v_kw, v_sym,
            "keyword(\"foo\") and symbol(\":foo\") must be distinct (arc 230)");
    }

    #[test]
    fn leaf_bool_vs_int_distinct() {
        let (vm, se) = fresh_env();
        let v_bool = encode(&HolonAST::bool_(true), &vm, &se);
        let v_int = encode(&HolonAST::i64(1), &vm, &se);
        assert_ne!(v_bool, v_int);
    }

    #[test]
    fn leaf_deterministic() {
        let (vm1, se1) = fresh_env();
        let (vm2, se2) = fresh_env();
        let v1 = encode(&HolonAST::string("hello"), &vm1, &se1);
        let v2 = encode(&HolonAST::string("hello"), &vm2, &se2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn per_variant_accessors_recover_payloads() {
        assert_eq!(HolonAST::i64(42).as_i64(), Some(42));
        assert_eq!(HolonAST::f64(3.14).as_f64(), Some(3.14));
        assert_eq!(HolonAST::bool_(true).as_bool(), Some(true));
        assert_eq!(HolonAST::string("foo").as_string(), Some("foo"));
        assert_eq!(HolonAST::keyword("k").as_keyword(), Some("k"));
    }

    #[test]
    fn per_variant_accessors_reject_wrong_variant() {
        let n = HolonAST::i64(42);
        assert!(n.as_string().is_none());
        assert!(n.as_bool().is_none());
        assert!(n.atom_inner().is_none());

        let bound = HolonAST::bind(HolonAST::i64(1), HolonAST::i64(2));
        assert!(bound.as_i64().is_none());
        assert!(bound.atom_inner().is_none());
    }

    #[test]
    fn atom_inner_recovers_wrapped_holon() {
        let inner = HolonAST::bundle(vec![HolonAST::i64(1), HolonAST::i64(2)]);
        let wrapped = HolonAST::atom(inner.clone());
        assert_eq!(wrapped.atom_inner(), Some(&inner));
    }

    #[test]
    fn bind_composes_atoms() {
        let (vm, se) = fresh_env();
        let v_bound = encode(
            &HolonAST::bind(HolonAST::string("role"), HolonAST::string("filler")),
            &vm,
            &se,
        );
        assert_eq!(v_bound.dimensions(), D);
    }

    #[test]
    fn bundle_list_form() {
        let (vm, se) = fresh_env();
        let v = encode(
            &HolonAST::bundle(vec![
                HolonAST::string("a"),
                HolonAST::string("b"),
                HolonAST::string("c"),
            ]),
            &vm,
            &se,
        );
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    fn permute_by_zero_is_identity() {
        let (vm, se) = fresh_env();
        let a = HolonAST::string("x");
        let v_original = encode(&a, &vm, &se);
        let v_permuted = encode(&HolonAST::permute(a, 0), &vm, &se);
        assert_eq!(v_original, v_permuted);
    }

    #[test]
    fn permute_is_invertible() {
        let (vm, se) = fresh_env();
        let a = HolonAST::string("x");
        let forward = HolonAST::permute(HolonAST::string("x"), 7);
        let round_trip = HolonAST::permute(forward, -7);
        let v_original = encode(&a, &vm, &se);
        let v_round_trip = encode(&round_trip, &vm, &se);
        assert_eq!(v_original, v_round_trip);
    }

    #[test]
    fn thermometer_endpoints() {
        let (vm, se) = fresh_env();
        let v_min = encode(&HolonAST::thermometer(0.0, 0.0, 100.0), &vm, &se);
        let v_max = encode(&HolonAST::thermometer(100.0, 0.0, 100.0), &vm, &se);
        let sim = Similarity::cosine(&v_min, &v_max);
        assert!(sim < -0.99, "cosine ≈ -1 expected, got {}", sim);
    }

    #[test]
    fn blend_option_b_subtract() {
        let (vm, se) = fresh_env();
        let v_sub = encode(
            &HolonAST::blend(
                HolonAST::string("x"),
                HolonAST::string("y"),
                1.0,
                -1.0,
            ),
            &vm,
            &se,
        );
        assert_eq!(v_sub.dimensions(), D);
    }

    #[test]
    fn blend_option_b_circular_weights() {
        let (vm, se) = fresh_env();
        let cos_basis = HolonAST::string("wat/std/circular-cos-basis");
        let sin_basis = HolonAST::string("wat/std/circular-sin-basis");
        let theta = std::f64::consts::FRAC_PI_4;
        let v = encode(
            &HolonAST::blend(cos_basis, sin_basis, theta.cos(), theta.sin()),
            &vm,
            &se,
        );
        assert_eq!(v.dimensions(), D);
    }

    // ─── Programs-as-atoms tests — Atom(Arc<HolonAST>) opaque-identity ─

    #[test]
    fn atom_holon_encodes() {
        let (vm, se) = fresh_env();
        let inner = HolonAST::bundle(vec![HolonAST::string("a"), HolonAST::string("b")]);
        let v = encode(&HolonAST::atom(inner), &vm, &se);
        assert_eq!(v.dimensions(), D);
    }

    #[test]
    fn atom_holon_identical_programs_same_vector() {
        let (vm, se) = fresh_env();
        let make = || {
            HolonAST::bundle(vec![HolonAST::i64(42), HolonAST::string("rsi")])
        };
        let v1 = encode(&HolonAST::atom(make()), &vm, &se);
        let v2 = encode(&HolonAST::atom(make()), &vm, &se);
        assert_eq!(v1, v2);
    }

    #[test]
    fn atom_holon_different_programs_different_vectors() {
        let (vm, se) = fresh_env();
        let prog_a = HolonAST::bundle(vec![HolonAST::i64(42), HolonAST::string("rsi")]);
        let prog_b = HolonAST::bundle(vec![HolonAST::i64(43), HolonAST::string("rsi")]);
        let v_a = encode(&HolonAST::atom(prog_a), &vm, &se);
        let v_b = encode(&HolonAST::atom(prog_b), &vm, &se);
        assert_ne!(v_a, v_b);
    }

    #[test]
    fn atom_holon_differs_from_direct_encoding() {
        // BOOK Ch.54: opaque-identity wrap and structural encoding must
        // produce distinct vectors. Atom(prog) ≠ prog at the geometric
        // level — they answer different questions.
        let (vm, se) = fresh_env();
        let make = || {
            HolonAST::bundle(vec![HolonAST::string("x"), HolonAST::string("y")])
        };
        let v_direct = encode(&make(), &vm, &se);
        let v_atomized = encode(&HolonAST::atom(make()), &vm, &se);
        assert_ne!(v_direct, v_atomized);
    }

    #[test]
    fn atom_wrap_is_repeatable() {
        // Atom(Atom(x)) ≠ Atom(x) ≠ x — quote-wrapping is meaningful.
        let (vm, se) = fresh_env();
        let leaf = HolonAST::string("x");
        let v_leaf = encode(&leaf, &vm, &se);
        let v_atom = encode(&HolonAST::atom(leaf.clone()), &vm, &se);
        let v_atom2 = encode(&HolonAST::atom(HolonAST::atom(leaf.clone())), &vm, &se);
        assert_ne!(v_leaf, v_atom);
        assert_ne!(v_atom, v_atom2);
        assert_ne!(v_leaf, v_atom2);
    }

    // ─── Hash + Eq + cache key tests ────────────────────────────────────

    #[test]
    fn derive_hash_eq_via_hashmap() {
        let mut m: HashMap<HolonAST, i64> = HashMap::new();
        m.insert(HolonAST::string("foo"), 1);
        m.insert(HolonAST::i64(42), 2);
        m.insert(
            HolonAST::bind(HolonAST::keyword("k"), HolonAST::i64(7)),
            3,
        );
        assert_eq!(m.get(&HolonAST::string("foo")), Some(&1));
        assert_eq!(m.get(&HolonAST::i64(42)), Some(&2));
        assert_eq!(
            m.get(&HolonAST::bind(HolonAST::keyword("k"), HolonAST::i64(7))),
            Some(&3)
        );
        assert!(m.get(&HolonAST::string("bar")).is_none());
    }

    #[test]
    fn f64_to_bits_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let h = |a: &HolonAST| {
            let mut hasher = DefaultHasher::new();
            a.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(h(&HolonAST::f64(0.1)), h(&HolonAST::f64(0.1)));
        assert_ne!(h(&HolonAST::f64(0.1)), h(&HolonAST::f64(0.2)));
    }

    #[test]
    fn f64_nan_hash_consistent_for_same_bits() {
        use std::collections::hash_map::DefaultHasher;
        let h = |a: &HolonAST| {
            let mut hasher = DefaultHasher::new();
            a.hash(&mut hasher);
            hasher.finish()
        };
        let nan_a = HolonAST::f64(f64::NAN);
        let nan_b = HolonAST::f64(f64::from_bits(f64::NAN.to_bits()));
        assert_eq!(h(&nan_a), h(&nan_b));
    }

    // ─── Canonical-EDN tests ────────────────────────────────────────────

    #[test]
    fn canonical_edn_deterministic() {
        let make = || {
            HolonAST::bundle(vec![HolonAST::i64(42), HolonAST::string("rsi")])
        };
        assert_eq!(canonical_edn_holon(&make()), canonical_edn_holon(&make()));
    }

    #[test]
    fn canonical_edn_variants_distinguished() {
        let bound = HolonAST::bind(HolonAST::i64(1), HolonAST::i64(2));
        let bundled = HolonAST::bundle(vec![HolonAST::i64(1), HolonAST::i64(2)]);
        assert_ne!(
            canonical_edn_holon(&bound),
            canonical_edn_holon(&bundled),
            "Bind and Bundle of the same atoms must canonicalize differently"
        );
    }

    // ─── Term decomposition tests (arc 073) ─────────────────────────────

    fn rsi_thought(value: f64) -> HolonAST {
        HolonAST::bind(
            HolonAST::keyword("rsi-thought"),
            HolonAST::thermometer(value, 0.0, 100.0),
        )
    }

    #[test]
    fn template_replaces_thermometer_with_slot_marker() {
        let form = rsi_thought(70.0);
        let tpl = form.template();
        match &tpl {
            HolonAST::Bind(a, b) => {
                assert_eq!(a.as_keyword(), Some("rsi-thought"));
                assert!(matches!(
                    **b,
                    HolonAST::SlotMarker { min, max } if min == 0.0 && max == 100.0
                ));
            }
            other => panic!("expected Bind(_, SlotMarker), got {:?}", other),
        }
    }

    #[test]
    fn template_collapses_thoughts_with_different_tuning() {
        // Different value, same range → identical templates.
        assert_eq!(rsi_thought(70.0).template(), rsi_thought(30.0).template());
    }

    #[test]
    fn template_distinguishes_different_ranges() {
        // Same shape, same value, different range → distinct templates.
        let a = HolonAST::bind(
            HolonAST::keyword("x"),
            HolonAST::thermometer(0.5, 0.0, 1.0),
        );
        let b = HolonAST::bind(
            HolonAST::keyword("x"),
            HolonAST::thermometer(0.5, -1.0, 1.0),
        );
        assert_ne!(a.template(), b.template());
    }

    #[test]
    fn template_distinguishes_different_atoms() {
        // Same range, different keyword → distinct templates.
        let a = rsi_thought(70.0);
        let b = HolonAST::bind(
            HolonAST::keyword("macd-thought"),
            HolonAST::thermometer(70.0, 0.0, 100.0),
        );
        assert_ne!(a.template(), b.template());
    }

    #[test]
    fn slots_pre_order() {
        // Bundle of two thoughts: pre-order yields rsi.value then macd.value.
        let form = HolonAST::bundle(vec![
            rsi_thought(70.0),
            HolonAST::bind(
                HolonAST::keyword("macd-thought"),
                HolonAST::thermometer(0.25, -1.0, 1.0),
            ),
        ]);
        assert_eq!(form.slots(), vec![70.0, 0.25]);
    }

    #[test]
    fn slots_and_ranges_parallel() {
        let form = HolonAST::bundle(vec![
            rsi_thought(70.0),
            HolonAST::thermometer(0.25, -1.0, 1.0),
        ]);
        let slots = form.slots();
        let ranges = form.ranges();
        assert_eq!(slots.len(), ranges.len());
        assert_eq!(slots, vec![70.0, 0.25]);
        assert_eq!(ranges, vec![(0.0, 100.0), (-1.0, 1.0)]);
    }

    #[test]
    fn empty_slots_for_thermometer_free_form() {
        let form = HolonAST::bind(HolonAST::keyword("x"), HolonAST::i64(42));
        assert!(form.slots().is_empty());
        assert!(form.ranges().is_empty());
    }

    #[test]
    fn slot_marker_does_not_re_emit_slot() {
        // Decomposing a template (which already contains SlotMarker)
        // produces empty slots — SlotMarker carries no value to extract.
        let tpl = rsi_thought(70.0).template();
        assert!(tpl.slots().is_empty());
        assert!(tpl.ranges().is_empty());
    }

    #[test]
    #[should_panic(expected = "SlotMarker is a query-key sentinel")]
    fn encode_template_panics() {
        let (vm, se) = fresh_env();
        let tpl = rsi_thought(70.0).template();
        let _ = encode(&tpl, &vm, &se);
    }

    // ─── Char leaf tests (arc 221 Stone 221.1) ──────────────────────────

    #[test]
    fn char_leaf_round_trip() {
        let h = HolonAST::char_('a');
        assert_eq!(h, HolonAST::Char('a'));
        assert_ne!(h, HolonAST::char_('b'));
        // Hash determinism: same char → same hash
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut h1);
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        HolonAST::char_('a').hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn char_distinct_from_string() {
        // Char('a') and String("a") MUST produce distinct canonical bytes
        // (and therefore distinct VSA vectors). PRIM_TAG_CHAR ≠ PRIM_TAG_STRING.
        let char_bytes = canonical_edn_holon(&HolonAST::char_('a'));
        let str_bytes = canonical_edn_holon(&HolonAST::string("a"));
        assert_ne!(
            char_bytes,
            str_bytes,
            "Char('a') and String(\"a\") MUST differ in canonical bytes"
        );
    }

    #[test]
    fn char_distinct_from_symbol() {
        // Char('a') is a raw Char leaf; symbol("a") = Bind(Atom("Symbol"), Atom("a")).
        // Structurally distinct — Char leaf vs Bind composition.
        let char_bytes = canonical_edn_holon(&HolonAST::char_('a'));
        let sym_bytes = canonical_edn_holon(&HolonAST::symbol("a"));
        assert_ne!(
            char_bytes,
            sym_bytes,
            "char_('a') and symbol(\"a\") MUST differ in canonical bytes (arc 230)"
        );
    }

    // ─── Keyword / Nil / Tag composition tests (arc 230 supersession) ───
    //
    // Arc 221 minted HolonAST::Keyword/Nil/Tag variants; arc 230 supersedes
    // with Bind(Atom(String(cls)), Atom(String(val))) compositions.
    // Tests updated accordingly — no direct variant pattern matching.

    #[test]
    fn keyword_composition_round_trip() {
        // Arc 230: keyword() produces Bind composition; as_keyword() recognises it.
        let h = HolonAST::keyword("foo");
        assert_eq!(h.as_keyword(), Some("foo"));
        // Leading colon stripped — both produce the same composition.
        assert_eq!(HolonAST::keyword(":foo"), HolonAST::keyword("foo"));
        // Hash determinism: same composition → same hash.
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut h1);
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        HolonAST::keyword("foo").hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn nil_composition_round_trip() {
        // Arc 230: nil() produces Symbol("nil") composition; is_nil() recognises it.
        let h = HolonAST::nil();
        assert!(h.is_nil());
        // Two nil() calls produce equal compositions.
        assert_eq!(h, HolonAST::nil());
        // Hash determinism.
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut h1);
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        HolonAST::nil().hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn tag_composition_round_trip() {
        // Arc 230: tag() produces Bind composition; as_tag() recognises it.
        let h = HolonAST::tag("uuid");
        assert_eq!(h.as_tag(), Some("uuid"));
        // Leading # stripped — both produce the same composition.
        assert_eq!(HolonAST::tag("#uuid"), HolonAST::tag("uuid"));
    }

    #[test]
    fn keyword_distinct_from_symbol() {
        // Keyword("foo") = Bind(Atom("Keyword"), Atom("foo"))
        // Symbol("foo") = Bind(Atom("Symbol"), Atom("foo"))
        // Classifier atoms differ → distinct canonical bytes + vectors.
        let kw_bytes = canonical_edn_holon(&HolonAST::keyword("foo"));
        let sym_bytes = canonical_edn_holon(&HolonAST::symbol("foo"));
        assert_ne!(kw_bytes, sym_bytes,
            "keyword(\"foo\") and symbol(\"foo\") MUST differ in canonical bytes");
    }

    #[test]
    fn nil_equals_symbol_nil() {
        // Arc 230: nil() = symbol("nil") = Bind(Atom("Symbol"), Atom("nil")).
        // This is the honest encoding per user 2026-05-22 articulation.
        let nil_bytes = canonical_edn_holon(&HolonAST::nil());
        let sym_nil_bytes = canonical_edn_holon(&HolonAST::symbol("nil"));
        assert_eq!(nil_bytes, sym_nil_bytes,
            "nil() and symbol(\"nil\") MUST be identical compositions post-arc-230");
    }

    #[test]
    fn tag_distinct_from_symbol() {
        // Tag("uuid") = Bind(Atom("Tag"), Atom("uuid"))
        // Symbol("uuid") = Bind(Atom("Symbol"), Atom("uuid"))
        // Classifier atoms differ → distinct canonical bytes.
        let tag_bytes = canonical_edn_holon(&HolonAST::tag("uuid"));
        let sym_bytes = canonical_edn_holon(&HolonAST::symbol("uuid"));
        assert_ne!(tag_bytes, sym_bytes,
            "tag(\"uuid\") and symbol(\"uuid\") MUST differ in canonical bytes");
    }

    #[test]
    fn keyword_distinct_from_nil() {
        // Keyword("nil") = Bind(Atom("Keyword"), Atom("nil"))
        // nil() = Bind(Atom("Symbol"), Atom("nil"))
        // Classifier atoms differ → distinct.
        let kw_bytes = canonical_edn_holon(&HolonAST::keyword("nil"));
        let nil_bytes = canonical_edn_holon(&HolonAST::nil());
        assert_ne!(kw_bytes, nil_bytes,
            "keyword(\"nil\") and nil() MUST differ in canonical bytes");
    }

    #[test]
    fn tag_distinct_from_keyword() {
        let tag_bytes = canonical_edn_holon(&HolonAST::tag("foo"));
        let kw_bytes = canonical_edn_holon(&HolonAST::keyword("foo"));
        assert_ne!(tag_bytes, kw_bytes,
            "tag(\"foo\") and keyword(\"foo\") MUST differ in canonical bytes");
    }

    #[test]
    fn nil_distinct_from_bool() {
        let nil_bytes = canonical_edn_holon(&HolonAST::nil());
        let true_bytes = canonical_edn_holon(&HolonAST::bool_(true));
        let false_bytes = canonical_edn_holon(&HolonAST::bool_(false));
        assert_ne!(nil_bytes, true_bytes);
        assert_ne!(nil_bytes, false_bytes);
    }

    #[test]
    fn as_keyword_returns_content_without_colon() {
        assert_eq!(HolonAST::keyword("foo").as_keyword(), Some("foo"));
        assert_eq!(HolonAST::keyword(":foo").as_keyword(), Some("foo"));
        // symbol("foo") has classifier "Symbol", not "Keyword" → None.
        assert_eq!(HolonAST::symbol("foo").as_keyword(), None);
        // nil() = symbol("nil") → as_keyword() returns None (wrong classifier).
        assert_eq!(HolonAST::nil().as_keyword(), None);
    }

    #[test]
    fn as_tag_returns_content_without_hash() {
        assert_eq!(HolonAST::tag("uuid").as_tag(), Some("uuid"));
        assert_eq!(HolonAST::tag("#uuid").as_tag(), Some("uuid"));
        // symbol("uuid") has classifier "Symbol" → as_tag() returns None.
        assert_eq!(HolonAST::symbol("uuid").as_tag(), None);
    }

    // ─── Symbol/String canonical-bytes distinction (arc 230 supersession) ─
    //
    // Arc 221 Stone 221.5 used PRIM_TAG_SYMBOL vs PRIM_TAG_STRING.
    // Arc 230: symbol() = Bind(Atom("Symbol"), Atom(s)) — structurally
    // distinct from bare String(s) because Bind ≠ leaf-atom shape.

    #[test]
    fn symbol_string_canonical_bytes_distinct() {
        // Arc 230: symbol("x") = Bind(Atom("Symbol"), Atom("x")).
        // String("x") = String leaf. Structurally distinct.
        let sym_bytes = canonical_edn_holon(&HolonAST::symbol("x"));
        let str_bytes = canonical_edn_holon(&HolonAST::string("x"));
        assert_ne!(
            sym_bytes,
            str_bytes,
            "symbol(\"x\") and string(\"x\") MUST differ in canonical bytes (arc 230)"
        );
    }

    #[test]
    fn symbol_string_vectors_distinct() {
        // Arc 230: symbol("x") and string("x") produce distinct VSA vectors.
        // Distinction comes from Bind structure (not PRIM_TAG_SYMBOL seed).
        let (vm, se) = fresh_env();
        let v_sym = encode(&HolonAST::symbol("x"), &vm, &se);
        let v_str = encode(&HolonAST::string("x"), &vm, &se);
        assert_ne!(
            v_sym,
            v_str,
            "symbol(\"x\") and string(\"x\") MUST produce distinct vectors (arc 230)"
        );
    }

    #[test]
    fn vsa_identity_no_collision_between_classifiers() {
        // STOP-8 check: arc 230 requires that Symbol("foo") and Keyword("foo")
        // produce distinct VSA vectors under the new Bind-composition encoding.
        // The classifier atom ("Symbol" vs "Keyword") is the discriminator.
        let (vm, se) = fresh_env();
        let v_sym = encode(&HolonAST::symbol("foo"), &vm, &se);
        let v_kw = encode(&HolonAST::keyword("foo"), &vm, &se);
        let v_tag = encode(&HolonAST::tag("foo"), &vm, &se);
        let v_nil = encode(&HolonAST::nil(), &vm, &se);
        assert_ne!(v_sym, v_kw, "symbol(\"foo\") must not collide with keyword(\"foo\")");
        assert_ne!(v_sym, v_tag, "symbol(\"foo\") must not collide with tag(\"foo\")");
        assert_ne!(v_kw, v_tag, "keyword(\"foo\") must not collide with tag(\"foo\")");
        // nil = symbol("nil"); keyword("nil") is distinct.
        let v_kw_nil = encode(&HolonAST::keyword("nil"), &vm, &se);
        assert_ne!(v_nil, v_kw_nil, "nil() must not collide with keyword(\"nil\")");
    }
}
