# Architecture

## Overview

Holon follows a **layered architecture** inspired by operating system design: a minimal, stable kernel with increasingly opinionated layers built on top. This design ensures the foundation remains stable while allowing experimentation and customization at higher levels.

The Rust and Python implementations share the same three-layer structure and the same algebraic semantics. Code using `holon::kernel` types maps directly to `holon.kernel` in Python.

## Three-Layer Design

```
┌─────────────────────────────────────────────┐
│   highlevel::  Holon                        │  ← Convenience & Composition
│   - Owns Encoder + VectorManager            │
│   - Delegates to kernel + memory            │
└─────────────────────────────────────────────┘
                    ↓ uses
┌─────────────────────────────────────────────┐
│   memory::  OnlineSubspace, EngramLibrary   │  ← Programmatic Neural Memory
│   - CCIPCA manifold learning                │
│   - Pattern snapshots + recall              │
└─────────────────────────────────────────────┘
                    ↓ uses
┌─────────────────────────────────────────────┐
│   kernel::  Primitives / Encoder / Scalar   │  ← Foundational VSA/HDC
│             Similarity / Accumulator        │
│             Walkable / Vector / VectorMgr   │
└─────────────────────────────────────────────┘
```

### Layer 1: `kernel` — The Foundation

**Purpose**: Minimal, stable VSA/HDC primitives that rarely change.

**Modules**:
- `primitives.rs` — Core VSA operations (~40 associated functions on `Primitives`)
- `encoder.rs` — Structure-preserving data encoding (JSON + Walkable)
- `scalar.rs` — Continuous value encoding (log-scale, linear, circular, temporal)
- `accumulator.rs` — Streaming frequency-weighted composition
- `vector_manager.rs` — Deterministic atom-to-vector mapping
- `walkable.rs` — Zero-serialization encoding trait
- `similarity.rs` — Distance/similarity metrics
- `vector.rs` — Bipolar vector type `{-1, 0, 1}`

**Design Principle**: No dependencies on `memory` or `highlevel`. This layer is the stable core that both higher layers and external crates build on.

### Layer 2: `memory` — The Innovation

**Purpose**: Novel programmatic neural memory system built on kernel primitives.

**Modules**:
- `subspace.rs` — `OnlineSubspace` for CCIPCA-based manifold learning
- `engram.rs` — `Engram` and `EngramLibrary` for pattern snapshots and recall

**Design Principle**: This is Holon's crown jewel. Uses kernel primitives to implement anomaly detection, pattern learning, and single-packet classification. Depends on kernel but not on highlevel.

### Layer 3: `highlevel` — The Convenience Layer

**Purpose**: Ergonomic wrapper for common workflows.

**Modules**:
- `client.rs` — `Holon` struct that owns an `Encoder`, `VectorManager`, and `ScalarEncoder`

**Design Principle**: Thin delegation layer. Every method on `Holon` calls through to the same public types available in `kernel` and `memory`. Users trade explicit control for less boilerplate.

## Import Patterns

### Explicit Layer Imports (Recommended)

```rust
use holon::kernel::{Encoder, VectorManager, Primitives, Similarity};
use holon::memory::{OnlineSubspace, EngramLibrary};
```

**Benefits**:
- Layer boundaries are visible at the call site
- Easier to reason about dependencies
- Better for library code consumed by others

### Crate-Level Re-exports (Quick Scripts)

```rust
use holon::{Encoder, VectorManager, Primitives, OnlineSubspace};
```

**Benefits**:
- Less verbose
- Backward compatible with pre-layered structure
- Good for one-off scripts and examples

### Facade (Maximum Convenience)

```rust
use holon::highlevel::Holon;

let holon = Holon::new(4096);
let vec = holon.encode_json(r#"{"key": "value"}"#)?;
```

**Benefits**:
- Least boilerplate
- Single object carries all state
- Good for getting started quickly

## Dependency Rules

```
highlevel
    ↓ depends on
memory
    ↓ depends on
kernel
    ↓ depends on
(std + serde + sha2 + rand)
```

**No circular dependencies.** Each layer only imports from layers below it. The `error` module sits outside the layers and is shared across all three.

## Why Three Layers?

1. **Stability** — Kernel API is stable; changes are rare. Memory and highlevel can evolve faster without breaking downstream users.
2. **Clarity** — Layer boundaries make it obvious what depends on what. A function importing only from `kernel` cannot accidentally depend on learning behavior.
3. **Testability** — Kernel can be tested independently (177 tests, 0.2s). No memory or highlevel setup needed.
4. **Cross-Language Parity** — Kernel mirrors `holon.kernel` in Python, enabling identical algorithms in both languages.
5. **Flexibility** — Users choose their abstraction level: raw primitives for library authors, facade for quick prototyping.

## Why `accumulator` in kernel, not memory?

Accumulators are primitive composition operations (like `bundle`). They don't depend on learning, subspace geometry, or pattern matching — they just add vectors and track counts. They belong with the other algebraic primitives.

## File Organization

```
src/
├── lib.rs              # Crate root: module declarations + backward-compat re-exports
├── error.rs            # Error types (shared across layers)
├── kernel/
│   ├── mod.rs          # Re-exports
│   ├── vector.rs       # Bipolar vectors {-1, 0, 1}
│   ├── vector_manager.rs
│   ├── primitives.rs   # ~40 VSA operations
│   ├── encoder.rs      # JSON + Walkable → vector
│   ├── walkable.rs     # Zero-serialization trait
│   ├── scalar.rs       # Continuous value encoding
│   ├── accumulator.rs  # Streaming composition
│   └── similarity.rs   # Distance/similarity metrics
├── memory/
│   ├── mod.rs
│   ├── subspace.rs     # OnlineSubspace (CCIPCA)
│   └── engram.rs       # Engram + EngramLibrary
└── highlevel/
    ├── mod.rs
    └── client.rs       # Holon convenience wrapper
```
