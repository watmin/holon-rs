# Holon (Rust)

**VSA for structured data.** JSON becomes geometry — role-filler binding preserves structure, similarity search finds shape, not just content.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python_version-holon-blue.svg)](https://github.com/watmin/holon)

<div align="center">
<img src="assets/superposition-incantation.gif" alt="Superposition Incantation">

*The sorcerer computes in high dimensions. Similarity becomes destiny.*
</div>

High-performance Vector Symbolic Architecture library in Rust. Encode structured data as vectors, compose with algebraic primitives, learn patterns from streams. Same foundations as Python Holon, but **12x faster**.

## Quick Start

```rust
use holon::Holon;

fn main() -> holon::Result<()> {
    let holon = Holon::new(4096);

    // Encode structured data as vectors
    let normal = holon.encode_json(r#"{"type": "login", "user": "alice"}"#)?;
    let suspicious = holon.encode_json(r#"{"type": "login", "attempts": 1000}"#)?;

    // Similarity: 0.0 = orthogonal, 1.0 = identical
    let sim = holon.similarity(&normal, &suspicious);
    println!("Similarity: {:.3}", sim);  // Different structures → low similarity

    // Learn what's "normal" from a stream
    let mut baseline = holon.create_accumulator();
    for _ in 0..1000 {
        let event = holon.encode_json(r#"{"type": "normal"}"#)?;
        holon.accumulate(&mut baseline, &event);
    }
    let normal_pattern = holon.normalize_accumulator(&baseline);

    // Detect anomalies
    let anomaly = holon.encode_json(r#"{"type": "attack", "payload": "DROP TABLE"}"#)?;
    let anomaly_score = 1.0 - holon.similarity(&anomaly, &normal_pattern);
    println!("Anomaly score: {:.3}", anomaly_score);  // High = anomalous

    Ok(())
}
```

<div align="center">
<img src="assets/time-bending-lattices.gif" alt="Time-Bending Lattices">

*Patterns encoded into geometry. What was noise becomes signal.*
</div>

## Performance

Holon-rs is designed for **real-time streaming** at production scale:

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| encode_json | 75 µs | 7 µs | **10x** |
| similarity | 15 µs | 1.4 µs | **11x** |
| bind | 12 µs | 0.8 µs | **15x** |
| Full detection pipeline | 15s | 1.2s | **12x** |

With SIMD acceleration (`--features simd`), similarity operations get an additional **5x** boost.

```bash
# Enable SIMD (AVX2/NEON)
cargo build --release --features simd
```

## Zero-Hardcode Anomaly Detection

The crown jewel: **100% attack recall, 4% false positive rate** - with ZERO domain knowledge.

```bash
cargo run --example zero_hardcode_detection --release --features simd
```

```
DETECTION RESULTS
------------------------------------------------------------
Phase              Packets   Detected       Rate       Status
warmup                 300          0         0% ○ LEARNING
DNS Attack           15000      14993       100% ✓ DETECTED
recovery-1             150          6         4% ✓ CLEAN
SYN Flood            18000      17993       100% ✓ DETECTED
recovery-2             150          6         4% ✓ CLEAN
NTP Attack           15000      14993       100% ✓ DETECTED
final                  150          6         4% ✓ CLEAN
------------------------------------------------------------
ATTACK RECALL                               100%
FALSE POSITIVE RATE                           4%
```

**What the detector doesn't know:**
- Port 53 = DNS
- Port 123 = NTP
- What "attack" means

**What it does know:**
- How to learn "normal" from a baseline
- How to detect when current traffic differs
- How to explain what changed (without interpretation)

## Core Concepts

### Deterministic Vectors

Same atom → same vector. Always. Everywhere.

```rust
// Tokyo data center
let tokyo = Holon::with_seed(4096, 42);
let vec_tokyo = tokyo.get_vector("suspicious_pattern");

// NYC data center
let nyc = Holon::with_seed(4096, 42);
let vec_nyc = nyc.get_vector("suspicious_pattern");

assert_eq!(vec_tokyo, vec_nyc);  // Always true - no sync needed!
```

This enables **distributed consensus without synchronization**. Every node with the same seed generates identical representations.

### Role-Filler Binding

Structure matters. `{"src_port": 53}` is different from `{"dst_port": 53}`:

```rust
let src = holon.encode_json(r#"{"src_port": 53}"#)?;
let dst = holon.encode_json(r#"{"dst_port": 53}"#)?;

let sim = holon.similarity(&src, &dst);
assert!(sim < 0.5);  // Same value, different role → different vector
```

### VSA Primitives

All the mystical operations:

| Primitive | Spell | Effect |
|-----------|-------|--------|
| `bind(a, b)` | Entanglement | Creates reversible association |
| `unbind(ab, a)` | Unraveling | Retrieves bound value |
| `bundle([a,b,c])` | Superposition | Combines into one (similar to all) |
| `negate(abc, b)` | Banishment | Removes component |
| `amplify(abc, a, 2.0)` | Empowerment | Strengthens component |
| `prototype([...])` | Essence Extraction | Finds common pattern |
| `difference(a, b)` | Delta Vision | The change is a vector |
| `blend(a, b, 0.7)` | Fusion | Weighted combination |

### Extended Primitives (Batch 014)

New operations for explainable anomaly forensics:

| Primitive | Effect |
|-----------|--------|
| `segment(stream, w, t)` | Find WHEN behavior changed |
| `complexity(v)` | Measure HOW mixed the signal is (0.0-1.0) |
| `invert(v, codebook)` | Decompose WHAT patterns are present |
| `similarity_profile(a, b)` | See WHERE dimensions agree/disagree |
| `attend(q, m, s, mode)` | Soft attention - focus on relevant dims |
| `project(v, subspace)` | Check IF in known subspace |
| `analogy(a, b, c)` | A:B::C:? pattern transfer |
| `conditional_bind(a, b, g)` | Gated binding by condition |

### Continuous Scalar Encoding

Encode rates, temperatures, angles - where similar values have similar vectors:

```rust
// Log-scale: equal ratios = equal similarity
let r100 = holon.encode_scalar_log(100.0);
let r1000 = holon.encode_scalar_log(1000.0);
let r10000 = holon.encode_scalar_log(10000.0);

// 100→1000 similarity ≈ 1000→10000 (both 10x ratio)

// Circular: hour 23 is similar to hour 0
let h23 = holon.encode_scalar(23.0, ScalarMode::Circular { period: 24.0 });
let h0 = holon.encode_scalar(0.0, ScalarMode::Circular { period: 24.0 });
```

### Accumulators: Frequency Matters

The secret weapon for anomaly detection:

```rust
// Bundle is idempotent - 100x same = 1x
let bundled = holon.bundle(&[&a, &a, &a, /* ...100x... */ &a]);

// Accumulator preserves frequency - 100x normal drowns out 1x rare
let mut acc = holon.create_accumulator();
for _ in 0..100 {
    holon.accumulate(&mut acc, &normal);
}
holon.accumulate(&mut acc, &rare);

// 99% of the signal is "normal" - rare barely registers
```

## Examples

```bash
# Zero-hardcode detection (DNS, SYN, NTP attacks)
cargo run --example zero_hardcode_detection --release --features simd

# Pure vector rate detection
cargo run --example pure_vector_rate --release --features simd

# Explainable anomaly forensics (Batch 014 - extended primitives)
cargo run --example explainable_forensics --release

# Zero-shot attack variant detection (analogy)
cargo run --example attack_variant_detection --release

# Improved detection with pattern attribution
cargo run --example improved_detection --release

# Targeted rate limiting (wider separation)
cargo run --example targeted_rate_limiting --release

# Run all tests
cargo test

# Benchmarks
cargo bench
```

## Building

```bash
# Standard build
cargo build --release

# With SIMD acceleration (recommended)
cargo build --release --features simd

# Run tests
cargo test

# Run with SIMD tests
cargo test --features simd
```

## Project Structure

```
holon-rs/
├── src/
│   ├── lib.rs           # Main Holon interface
│   ├── vector.rs        # Bipolar vectors {-1, 0, 1}
│   ├── vector_manager.rs # Deterministic atom→vector
│   ├── primitives.rs    # VSA operations
│   ├── encoder.rs       # JSON→vector encoding
│   ├── scalar.rs        # Continuous value encoding
│   ├── accumulator.rs   # Streaming primitives
│   ├── similarity.rs    # Metrics (cosine, dot, etc.)
│   └── error.rs         # Error types
├── examples/
│   ├── zero_hardcode_detection.rs  # Full Batch 012 demo
│   ├── pure_vector_rate.rs         # Challenge 008 port
│   ├── explainable_forensics.rs    # Batch 014 extended primitives
│   ├── attack_variant_detection.rs # Zero-shot detection via analogy
│   ├── improved_detection.rs       # Pattern attribution
│   └── targeted_rate_limiting.rs   # Dimension-level rate limiting
└── benches/
    └── benchmarks.rs    # Criterion benchmarks
```

<div align="center">
<img src="assets/forbidden-binding-spell.gif" alt="Forbidden Binding">

*From mystical runes to mathematical vectors. The magic is in the math.*
</div>

## Parity with Python

| Feature | Python | Rust |
|---------|--------|------|
| Deterministic vectors | ✅ | ✅ |
| Role-filler binding | ✅ | ✅ |
| All VSA primitives | ✅ | ✅ |
| Extended primitives (Batch 014) | ✅ | ✅ |
| Accumulators | ✅ | ✅ |
| Scalar encoding (linear/log/circular) | ✅ | ✅ |
| Sequence encoding | ✅ | ✅ |
| SIMD acceleration | ❌ | ✅ |
| Zero-hardcode detection | ✅ | ✅ (12x faster) |

## See Also

- **[holon](https://github.com/watmin/holon)** — Python implementation with HTTP API, extensive documentation, and 12 challenge batches exploring VSA capabilities. Start here for learning and prototyping.

## License

MIT

---

*"The sorcerer supreme doesn't debug. The sorcerer supreme computes in dimensions where bugs cannot exist."*
