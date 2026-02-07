# Holon (Rust)

Programmatic Neural Memory using Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC).

## Overview

Holon is a library for building deterministic, explainable AI systems using high-dimensional vector operations. It provides a clean, unified interface for:

- **Encoding**: Convert structured data (JSON, key-value pairs) into vectors
- **VSA Primitives**: bind, bundle, negate, amplify, prototype, etc.
- **Streaming**: Frequency-preserving accumulators for online learning
- **Similarity**: Multiple metrics for vector comparison
- **Scalar Encoding**: Continuous values with linear, log, or circular encoding

## Quick Start

```rust
use holon::{Holon, ScalarMode};

fn main() -> holon::Result<()> {
    // Create a Holon instance
    let holon = Holon::new(4096);

    // Encode structured data
    let billing = holon.encode_json(r#"{"type": "billing", "amount": 100}"#)?;
    let technical = holon.encode_json(r#"{"type": "technical"}"#)?;

    // Compute similarity
    let sim = holon.similarity(&billing, &technical);
    println!("Similarity: {:.3}", sim);

    // VSA primitives
    let combined = holon.bundle(&[&billing, &technical]);
    let without_billing = holon.negate(&combined, &billing);

    // Continuous scalar encoding
    let rate100 = holon.encode_scalar_log(100.0);
    let rate1000 = holon.encode_scalar_log(1000.0);
    // Equal ratios have equal similarity drops

    // Streaming with accumulators
    let mut baseline = holon.create_accumulator();
    for _ in 0..100 {
        let event = holon.encode_json(r#"{"type": "normal"}"#)?;
        holon.accumulate(&mut baseline, &event);
    }
    let normal_pattern = holon.normalize_accumulator(&baseline);

    // Detect anomalies
    let anomaly = holon.encode_json(r#"{"type": "unusual"}"#)?;
    let anomaly_score = 1.0 - holon.similarity(&anomaly, &normal_pattern);
    println!("Anomaly score: {:.3}", anomaly_score);

    Ok(())
}
```

## Core Concepts

### Deterministic Vectors

The same atomic value always produces the exact same vector:

```rust
let holon1 = Holon::with_seed(4096, 42);
let holon2 = Holon::with_seed(4096, 42);

let v1 = holon1.get_vector("hello");
let v2 = holon2.get_vector("hello");
assert_eq!(v1, v2);  // Always true
```

This enables distributed systems to agree on vector representations without synchronization.

### Role-Filler Binding

Structural encoding preserves key-value relationships:

```rust
let src_53 = holon.encode_json(r#"{"src_port": 53}"#)?;
let dst_53 = holon.encode_json(r#"{"dst_port": 53}"#)?;

// These are DIFFERENT vectors because of role-filler binding
let sim = holon.similarity(&src_53, &dst_53);
assert!(sim < 0.5);
```

### VSA Primitives

| Primitive | Purpose | Property |
|-----------|---------|----------|
| `bind(a, b)` | Create association | `unbind(bind(a, b), a) ≈ b` |
| `bundle([a, b, c])` | Create superposition | Similar to all inputs |
| `negate(abc, b)` | Remove component | Reduces similarity to b |
| `amplify(abc, a, 2.0)` | Strengthen component | Increases similarity to a |
| `prototype([...], 0.5)` | Extract common pattern | Majority voting |

### Accumulators vs Bundle

Critical difference for anomaly detection:

```rust
// Bundle is idempotent
let bundled = holon.bundle(&[&a, &a, &a, &a, &a]);  // Same as just 'a'

// Accumulator preserves frequency
let mut acc = holon.create_accumulator();
for _ in 0..100 {
    holon.accumulate(&mut acc, &common);
}
holon.accumulate(&mut acc, &rare);

// common has 100x more influence than rare
```

### Scalar Encoding

Encode continuous values where similar values have similar vectors:

```rust
// Linear: equal absolute differences → equal similarity drops
let t100 = holon.encode_scalar(100.0, ScalarMode::Linear { scale: 1000.0 });
let t110 = holon.encode_scalar(110.0, ScalarMode::Linear { scale: 1000.0 });

// Logarithmic: equal ratios → equal similarity drops
let r100 = holon.encode_scalar_log(100.0);
let r1000 = holon.encode_scalar_log(1000.0);
let r10000 = holon.encode_scalar_log(10000.0);
// sim(r100, r1000) ≈ sim(r1000, r10000)

// Circular: wraps around (hour 23 similar to hour 0)
let h23 = holon.encode_scalar(23.0, ScalarMode::Circular { period: 24.0 });
let h0 = holon.encode_scalar(0.0, ScalarMode::Circular { period: 24.0 });
```

## Building

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Project Structure

```
holon-rs/
├── Cargo.toml          # Package manifest
├── src/
│   ├── lib.rs          # Main entry point, Holon struct
│   ├── vector.rs       # Vector type (bipolar {-1, 0, 1})
│   ├── vector_manager.rs # Deterministic atom → vector mapping
│   ├── primitives.rs   # VSA operations (bind, bundle, etc.)
│   ├── encoder.rs      # Structured data encoding
│   ├── scalar.rs       # Continuous scalar encoding
│   ├── accumulator.rs  # Streaming operations
│   ├── similarity.rs   # Similarity metrics
│   └── error.rs        # Error types
└── benches/
    └── benchmarks.rs   # Criterion benchmarks
```

## License

MIT
