# Encoding Guide

Holon encodes structured data as high-dimensional bipolar vectors. The encoding method determines what "similarity" means for your data. This guide covers the available modes and when to use each.

## Structured Data Encoding

The primary encoding path converts JSON or Walkable structs into vectors using **role-filler binding**: each value (filler) is bound to its key (role) before bundling.

```rust
use holon::kernel::{Encoder, VectorManager, Similarity};

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);

// Role-filler binding preserves structure
let src = enc.encode_json(r#"{"src_port": 53}"#)?;
let dst = enc.encode_json(r#"{"dst_port": 53}"#)?;

// Same value, different role → different vectors
assert!(Similarity::cosine(&src, &dst) < 0.5);
```

This is why Holon captures *structure*, not just content. `{"src_port": 53}` and `{"dst_port": 53}` share the value "53" but are bound to different roles, producing near-orthogonal vectors.

## Sequence Encoding

For ordered or unordered collections, Holon provides four sequence modes:

| Mode | Order preserved? | Use case | Example |
|------|-----------------|----------|---------|
| `Bundle` | No | Tags, categories, sets | `["admin", "readonly"]` |
| `Positional` | Yes | Event sequences | `["login", "transfer", "logout"]` |
| `Chained` | Yes (pairwise) | Transaction chains | `["A", "B", "C"]` → AB + BC |
| `Ngram { n }` | Yes (windowed) | Fuzzy substring matching | bigrams, trigrams |

```rust
use holon::kernel::{Encoder, VectorManager, SequenceMode, Similarity};

let vm = VectorManager::new(4096);
let enc = Encoder::new(vm);

// Order-independent: same items in any order → same vector
let s1 = enc.encode_sequence(&["A", "B", "C"], SequenceMode::Bundle);
let s2 = enc.encode_sequence(&["C", "A", "B"], SequenceMode::Bundle);
assert!(Similarity::cosine(&s1, &s2) > 0.9);

// Order-preserving: reversed sequence → different vector
let s3 = enc.encode_sequence(&["A", "B", "C"], SequenceMode::Positional);
let s4 = enc.encode_sequence(&["C", "B", "A"], SequenceMode::Positional);
assert!(Similarity::cosine(&s3, &s4) < 0.8);

// N-gram: shared subsequences → partial similarity
let s5 = enc.encode_sequence(&["A", "B", "C", "D"], SequenceMode::Ngram { n: 2 });
let s6 = enc.encode_sequence(&["A", "B", "X", "Y"], SequenceMode::Ngram { n: 2 });
// Shared "AB" bigram gives partial similarity
```

### Choosing a Sequence Mode

- **Bundle** when order doesn't matter (tags, feature sets, permissions)
- **Positional** when exact position matters (event traces, protocol sequences)
- **Chained** when transitions matter more than absolute position (state machines)
- **Ngram** when fuzzy/partial matching matters (text search, DNA sequences)

## Scalar Encoding

By default, numbers encode as strings — `100` and `200` produce orthogonal vectors with no magnitude relationship. For continuous values where similar numbers should produce similar vectors, use scalar encoding.

### Log-Scale (`$log` / `encode_log`)

Equal *ratios* produce equal similarity. 100→1000 has the same similarity drop as 1000→10000 (both 10x).

```rust
use holon::kernel::{ScalarEncoder, Similarity};

let scalar = ScalarEncoder::new(4096);

let r100 = scalar.encode_log(100.0);
let r1k = scalar.encode_log(1_000.0);
let r10k = scalar.encode_log(10_000.0);

// 10x ratios → similar similarity drops
let sim_10x = Similarity::cosine(&r100, &r1k);
let sim_10x_2 = Similarity::cosine(&r1k, &r10k);
// sim_10x ≈ sim_10x_2
```

**Use for**: Packet rates, byte counts, file sizes, prices, request counts — anything where ratios matter more than absolute differences.

### Linear (`ScalarMode::Linear`)

Equal *absolute differences* produce equal similarity. 100→110 has the same similarity drop as 1000→1010 (both +10).

```rust
use holon::kernel::{ScalarEncoder, ScalarMode};

let scalar = ScalarEncoder::new(4096);

let t72 = scalar.encode(72.0, ScalarMode::Linear { scale: 200.0 });
let t75 = scalar.encode(75.0, ScalarMode::Linear { scale: 200.0 });
// 72°F and 75°F are very similar
```

**Use for**: Temperatures, positions, coordinates, latencies — anything where absolute differences matter.

### Circular (`ScalarMode::Circular`)

Values wrap around a period. Hour 23 is close to hour 1, not 22 units away.

```rust
use holon::kernel::{ScalarEncoder, ScalarMode};

let scalar = ScalarEncoder::new(4096);

let h23 = scalar.encode(23.0, ScalarMode::Circular { period: 24.0 });
let h1 = scalar.encode(1.0, ScalarMode::Circular { period: 24.0 });
// Only 2 hours apart on the clock → high similarity
```

**Use for**: Time of day, day of week, compass bearings, angles, phase.

### Temporal (`ScalarValue::time`)

Decomposes Unix timestamps into four circular+positional components for multi-resolution temporal similarity:

```rust
use holon::kernel::{ScalarValue, TimeResolution};

// Default resolution (Hour)
let morning = ScalarValue::time(1_700_000_000.0);

// Finer resolution
let precise = ScalarValue::time_with_resolution(ts, TimeResolution::Second);
```

Components: hour-of-day (period 24), day-of-week (period 7), month (period 12), and absolute position (sin/cos).

**Use for**: Event timestamps where "same time of day" or "same day of week" should be similar.

## Choosing the Right Encoding

| Data type | Encoding | Inline marker | Why |
|-----------|----------|--------------|-----|
| Network rates | Log-scale | `$log` | 100 pps and 100k pps are qualitatively different |
| File sizes | Log-scale | `$log` | 1 KB vs 1 GB: ratio matters |
| Latency | Linear | `$linear` | 10ms vs 20ms: absolute difference matters |
| Temperature | Linear | `$linear` | 72°F vs 100°F: absolute difference matters |
| Hour of day | Circular | — | 23:00 is near 01:00, not 22 hours away |
| Day of week | Circular | — | Sunday is near Monday |
| Unix timestamp | Temporal | `$time` | Multi-resolution: hour + day + month + position |
| Port numbers | String (default) | — | Port 80 and 81 are unrelated protocols |
| User IDs | String (default) | — | Exact match only |
| Status codes | String (default) | — | 200 and 201 are categorical, not numeric |

### Inline Markers in JSON

When encoding JSON, use `$log`, `$linear`, and `$time` markers to control encoding inline:

```rust
let enc = Encoder::new(VectorManager::new(4096));

// Numbers without markers encode as strings (no magnitude relationship)
let v1 = enc.encode_json(r#"{"rate": 100}"#)?;
let v2 = enc.encode_json(r#"{"rate": 200}"#)?;
// similarity ≈ 0.0 (orthogonal — "100" and "200" are just different strings)

// With $log marker: similar magnitudes → similar vectors
let v3 = enc.encode_json(r#"{"rate": {"$log": 100}}"#)?;
let v4 = enc.encode_json(r#"{"rate": {"$log": 200}}"#)?;
// similarity ≈ 0.98 (2x ratio = very similar)
```

### Walkable Scalars

When implementing the `Walkable` trait, use `ScalarValue` variants for typed scalar encoding:

```rust
use holon::kernel::{ScalarValue, WalkableValue};

fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
    vec![
        // Log-scale for rates
        ("rate_pps", WalkableValue::Scalar(ScalarValue::log(self.rate))),
        // Linear for latency
        ("latency_ms", WalkableValue::Scalar(ScalarValue::linear(self.latency))),
        // Temporal for timestamps
        ("timestamp", WalkableValue::Scalar(ScalarValue::time(self.ts))),
        // Default string encoding for categorical values
        ("protocol", self.protocol.to_walkable_value()),
    ]
}
```

## Summary

1. **Structure** (JSON/Walkable) — Role-filler binding preserves keys, nesting, relationships
2. **Sequences** — Choose mode based on whether order and position matter
3. **Scalars** — Choose mode based on what "similar" means for your domain
4. **Default** (string) — Use for categorical/nominal values where similarity is meaningless
