# Holon (Rust)

**VSA for structured data.** JSON becomes geometry — role-filler binding preserves structure, similarity search finds shape, not just content.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python_version-holon-blue.svg)](https://github.com/watmin/holon)

<div align="center">
<img src="https://raw.githubusercontent.com/watmin/holon-rs/main/assets/superposition-incantation.gif" alt="Superposition Incantation">

*The sorcerer computes in high dimensions. Similarity becomes destiny.*
</div>

High-performance Vector Symbolic Architecture library in Rust. Encode structured data as vectors, compose with algebraic primitives, learn patterns from streams, detect anomalies online. Same foundations as Python Holon, but **12x faster**.

## Architecture

Three layers, same as Python Holon:

```
┌─────────────────────────────────────────────┐
│  highlevel::  Holon  (convenience wrapper)  │
├─────────────────────────────────────────────┤
│  memory::  OnlineSubspace, EngramLibrary    │
├─────────────────────────────────────────────┤
│  kernel::  primitives / encoder / scalar    │
│            similarity / accumulator         │
│            walkable / vector / vector_mgr   │
└─────────────────────────────────────────────┘
```

1. **`kernel`** — VSA primitives, encoding, similarity, accumulators, walkable trait
2. **`memory`** — Online subspace learning (CCIPCA), engram storage and recall
3. **`highlevel`** — `Holon` convenience wrapper that owns an `Encoder` and delegates

Import directly from layers for clarity, or use crate-level re-exports for convenience:

```rust
// Explicit layer imports (recommended for library / production code)
use holon::kernel::{Encoder, VectorManager, Primitives, Similarity};
use holon::memory::OnlineSubspace;

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);  // vm uses Arc internally; clone to share

let a = enc.encode_json(r#"{"role": "admin"}"#)?;
let b = enc.encode_json(r#"{"role": "user"}"#)?;
let sim = Similarity::cosine(&a, &b);
let bound = Primitives::bind(&a, &b);

// Crate-level re-exports (less typing, same types)
use holon::{Encoder, VectorManager, OnlineSubspace};

// Holon facade (most ergonomic for quick scripts)
use holon::highlevel::Holon;
let holon = Holon::new(4096);
let vec = holon.encode_json(r#"{"key": "value"}"#)?;
let sim = holon.similarity(&a, &b);
let sub = holon.create_subspace(32);
```

The facade never hides functionality — it delegates to the same public types you can use directly.

## Quick Start

```rust
use holon::kernel::{Encoder, VectorManager, Primitives, Similarity, Accumulator};

fn main() -> holon::Result<()> {
    let vm = VectorManager::new(4096);
    let enc = Encoder::new(vm);

    // Encode structured data as vectors
    let normal = enc.encode_json(r#"{"type": "login", "user": "alice"}"#)?;
    let suspicious = enc.encode_json(r#"{"type": "login", "attempts": 1000}"#)?;

    // Similarity: 0.0 = orthogonal, 1.0 = identical
    let sim = Similarity::cosine(&normal, &suspicious);
    println!("Similarity: {:.3}", sim);  // Different structures → low similarity

    // Learn what's "normal" from a stream
    let mut baseline = Accumulator::new(4096);
    for _ in 0..1000 {
        let event = enc.encode_json(r#"{"type": "normal"}"#)?;
        baseline.add(&event);
    }
    let normal_pattern = baseline.normalize();

    // Detect anomalies
    let anomaly = enc.encode_json(r#"{"type": "attack", "payload": "DROP TABLE"}"#)?;
    let anomaly_score = 1.0 - Similarity::cosine(&anomaly, &normal_pattern);
    println!("Anomaly score: {:.3}", anomaly_score);  // High = anomalous

    Ok(())
}
```

> **Note**: The Quick Start uses explicit kernel imports. For a more concise facade-based
> API, see `holon::highlevel::Holon`.

<div align="center">
<img src="https://raw.githubusercontent.com/watmin/holon-rs/main/assets/time-bending-lattices.gif" alt="Time-Bending Lattices">

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
use holon::kernel::VectorManager;

// Tokyo data center
let tokyo = VectorManager::with_seed(4096, 42);
let vec_tokyo = tokyo.get_vector("suspicious_pattern");

// NYC data center
let nyc = VectorManager::with_seed(4096, 42);
let vec_nyc = nyc.get_vector("suspicious_pattern");

assert_eq!(vec_tokyo, vec_nyc);  // Always true - no sync needed!
```

This enables **distributed consensus without synchronization**. Every node with the same seed generates identical representations.

### Role-Filler Binding

Structure matters. `{"src_port": 53}` is different from `{"dst_port": 53}`:

```rust
use holon::kernel::{Encoder, VectorManager, Similarity};

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);

let src = enc.encode_json(r#"{"src_port": 53}"#).unwrap();
let dst = enc.encode_json(r#"{"dst_port": 53}"#).unwrap();

let sim = Similarity::cosine(&src, &dst);
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

Encode rates, temperatures, angles, timestamps — where similar values have similar vectors:

```rust
use holon::kernel::{ScalarEncoder, ScalarMode};

let scalar = ScalarEncoder::new(4096);

// Log-scale: equal ratios = equal similarity
let r100 = scalar.encode_log(100.0);
let r1000 = scalar.encode_log(1000.0);
let r10000 = scalar.encode_log(10000.0);
// 100→1000 similarity ≈ 1000→10000 (both 10x ratio)

// Circular: hour 23 is similar to hour 0
let h23 = scalar.encode(23.0, ScalarMode::Circular { period: 24.0 });
let h0 = scalar.encode(0.0, ScalarMode::Circular { period: 24.0 });
```

| Mode | Use Case | Similarity Property |
|------|----------|---------------------|
| `$log` | Packet rates, file sizes, frequencies | Equal ratios = equal similarity |
| `$linear` | Temperatures, positions | Equal differences = equal similarity |
| `$time` | Unix timestamps, event times | Circular (hour/dow/month) + positional |

### Temporal Encoding

Encode Unix timestamps with circular periodicity — same hour next week ≈ same vector:

```rust
use holon::kernel::{ScalarValue, TimeResolution};

// Default resolution (Hour)
let morning = ScalarValue::time(1_700_000_000.0);

// Explicit resolution for finer/coarser discrimination
let precise = ScalarValue::time_with_resolution(ts, TimeResolution::Second);
let coarse = ScalarValue::time_with_resolution(ts, TimeResolution::Day);
```

TimeFloat decomposes timestamps into four components bound by role vectors:
- **Hour-of-day** (circular, period=24) — 9am clusters with 9am
- **Day-of-week** (circular, period=7) — Monday clusters with Monday
- **Month** (circular, period=12) — seasonal patterns
- **Position** (transformer sin/cos) — absolute discrimination, resolution-dependent

### Memory Layer: OnlineSubspace + Engrams

Learn what "normal" looks like from a stream, then score new observations:

```rust
use holon::kernel::{Encoder, VectorManager};
use holon::memory::{OnlineSubspace, EngramLibrary};

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);

// Learn the normal traffic manifold online (CCIPCA algorithm)
let mut subspace = OnlineSubspace::new(4096, 32);
for event in training_events {
    let vec = enc.encode_walkable(&event);
    subspace.update(&vec.to_f64());
}

// Score new observations: residual > threshold → anomaly
let probe = enc.encode_walkable(&new_event);
if subspace.residual(&probe.to_f64()) > subspace.threshold() {
    println!("anomaly detected");
}

// Store learned patterns as named engrams
let mut library = EngramLibrary::new(4096);
library.add("normal_traffic", &subspace, None, Default::default());

// Later: recall the best-matching pattern
let matches = library.match_vec(&probe.to_f64(), 3, 10);
// Returns [(name, residual)] sorted ascending — lower = better fit
```

### Walkable Trait: Zero-Serialization Encoding

Skip JSON entirely — encode your structs directly:

```rust
use holon::kernel::{Encoder, VectorManager, Walkable, WalkType, WalkableValue, ScalarValue};

struct Packet {
    protocol: String,
    dst_port: u16,
    rate_pps: f64,
}

impl Walkable for Packet {
    fn walk_type(&self) -> WalkType { WalkType::Map }
    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            ("protocol", WalkableValue::Scalar(ScalarValue::String(self.protocol.clone()))),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            ("rate", WalkableValue::Scalar(ScalarValue::log(self.rate_pps))),
        ]
    }
}

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);
let vec = enc.encode_walkable(&my_packet);
```

### Accumulators: Frequency Matters

The secret weapon for anomaly detection:

```rust
use holon::kernel::{Primitives, Accumulator, Similarity};

// Bundle is idempotent - 100x same = 1x
let bundled = Primitives::bundle(&[&a, &a, &a, /* ...100x... */ &a]);

// Accumulator preserves frequency - 100x normal drowns out 1x rare
let mut acc = Accumulator::new(4096);
for _ in 0..100 {
    acc.add(&normal);
}
acc.add(&rare);

// 99% of the signal is "normal" - rare barely registers
```

## Examples

```bash
# --- Showcases (compositional algebra + memory, cross-domain) ---
cargo run --example compositional_recall --release      # algebraic queries over incident library
cargo run --example streaming_changepoint --release     # unlabeled phase detection: segment + invert
cargo run --example config_drift_remediation --release  # detect drift, attribute fields, remediate

# --- Anomaly detection (network traffic) ---
cargo run --example zero_hardcode_detection --release --features simd
cargo run --example pure_vector_rate --release --features simd
cargo run --example improved_detection --release
cargo run --example payload_anomaly_detection --release

# --- Walkable (zero-serialization) ---
cargo run --example walkable_detection --release
cargo run --example walkable_rate --release

# --- Memory layer (OnlineSubspace + Engrams) ---
cargo run --example online_anomaly_memory --release
cargo run --example temporal_encoding --release

# --- Extended primitives ---
cargo run --example explainable_forensics --release
cargo run --example attack_variant_detection --release
cargo run --example targeted_rate_limiting --release
cargo run --example rate_limit_mitigation --release

# --- Encoding ---
cargo run --example magnitude_aware_encoding --release
cargo run --example byte_match_derivation --release

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
│   ├── lib.rs              # Crate root: module declarations + re-exports
│   ├── error.rs            # Error types (shared across layers)
│   ├── kernel/
│   │   ├── mod.rs          # Kernel re-exports
│   │   ├── vector.rs       # Bipolar vectors {-1, 0, 1}
│   │   ├── vector_manager.rs # Deterministic atom→vector
│   │   ├── primitives.rs   # VSA operations
│   │   ├── encoder.rs      # JSON + Walkable → vector encoding
│   │   ├── walkable.rs     # Zero-serialization Walkable trait
│   │   ├── scalar.rs       # Continuous value encoding
│   │   ├── accumulator.rs  # Streaming primitives
│   │   └── similarity.rs   # Metrics (cosine, dot, etc.)
│   ├── memory/
│   │   ├── mod.rs          # Re-exports
│   │   ├── subspace.rs     # OnlineSubspace (CCIPCA)
│   │   └── engram.rs       # Engram + EngramLibrary
│   └── highlevel/
│       ├── mod.rs          # Re-exports
│       └── client.rs       # Holon convenience wrapper
├── examples/               # 17 runnable demos
└── benches/
    └── benchmarks.rs       # Criterion benchmarks
```

<div align="center">
<img src="https://raw.githubusercontent.com/watmin/holon-rs/main/assets/forbidden-binding-spell.gif" alt="Forbidden Binding">

*From mystical runes to mathematical vectors. The magic is in the math.*
</div>

## Parity with Python

| Feature | Python | Rust |
|---------|--------|------|
| Three-layer architecture (kernel/memory/highlevel) | ✅ | ✅ |
| Deterministic vectors | ✅ | ✅ |
| Role-filler binding | ✅ | ✅ |
| All VSA primitives | ✅ | ✅ |
| Extended primitives (Batch 014) | ✅ | ✅ |
| Accumulators | ✅ | ✅ |
| Scalar encoding (linear/log/circular) | ✅ | ✅ |
| Temporal encoding (TimeFloat) | ✅ | ✅ |
| Inline numeric markers ($log, $linear, $time) | ✅ | ✅ |
| Walkable trait (zero-serialization) | ✅ | ✅ |
| OnlineSubspace (CCIPCA) | ✅ | ✅ |
| Engram / EngramLibrary | ✅ | ✅ |
| Sequence encoding | ✅ | ✅ |
| SIMD acceleration | ❌ | ✅ |
| Persistence (CPU/Qdrant store) | ✅ | — (deferred) |
| Zero-hardcode detection | ✅ | ✅ (12x faster) |

## See Also

- **[holon](https://github.com/watmin/holon)** — Python implementation with HTTP API, extensive documentation, and 12 challenge batches exploring VSA capabilities. Start here for learning and prototyping.

## License

MIT

---

*"The sorcerer supreme doesn't debug. The sorcerer supreme computes in dimensions where bugs cannot exist."*
