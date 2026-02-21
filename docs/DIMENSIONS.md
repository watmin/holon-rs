# Dimension Selection Guide

## Quick Reference

| Use Case | Recommended | Per-vector bytes | Why |
|----------|-------------|-----------------|-----|
| Simple documents (<20 fields) | 1024 | 1 KB | Fast, memory efficient |
| Complex documents (20-100 fields) | 4096 | 4 KB | Best accuracy/memory balance |
| Very complex (100+ fields, time encoding) | 4096-8192 | 4-8 KB | Handles high field counts |
| Maximum headroom (unknown complexity) | 16384 | 16 KB | Rarely needed |

## The Trade-offs

Higher dimensions provide:
- More orthogonal random vectors (less interference between atoms)
- Better discrimination for complex structures
- Higher bundle capacity (more items superposed cleanly)

Lower dimensions provide:
- Faster operations (similarity, encoding scale linearly with `d`)
- Lower memory usage (linear scaling)
- Sufficient accuracy for most use cases

## Empirical Findings

### Where Dimensions Matter

| Test | 512d | 1024d | 4096d | 16384d |
|------|------|-------|-------|--------|
| 100-field doc precision | **40%** | 100% | 100% | 100% |
| Category discrimination | 100% | 100% | 100% | 100% |
| Bundle capacity (1000 items) | 100% | 100% | 100% | 100% |
| Near-duplicate score gap | 0.217 | 0.194 | 0.186 | 0.187 |
| Time encoding score | 0.808 | 0.783 | **0.858** | 0.825 |

**Key insight**: 512 dimensions fail at 100+ field documents. 1024+ handles everything we tested.

## Memory Planning

Holon vectors are bipolar `i8` (`{-1, 0, 1}`), so each vector is exactly `d` bytes.

| Records | 1024d | 4096d | 16384d |
|---------|-------|-------|--------|
| 1M | 1.0 GB | 3.9 GB | 15.6 GB |
| 10M | 10 GB | 39 GB | 156 GB |
| 100M | 98 GB | 390 GB | 1.5 TB |

These are raw vector storage only. `OnlineSubspace` with `k` components adds roughly `k * d * 8` bytes (f64 basis vectors).

## Recommendations by Scenario

### Scenario 1: Network Packet Analysis / DDoS Detection
**Use 4096 dimensions**
- Walkable structs with 5-15 fields
- Scalar encoding for rates and timestamps
- Sufficient headroom for protocol + payload features
- All showcase examples use this

### Scenario 2: High-Volume Log/Event Streams
**Use 1024 dimensions**
- Simpler structures, high throughput matters
- 4x more vectors per GB than 4096
- Sufficient for most anomaly detection

### Scenario 3: Complex Nested Data (configs, policies, schemas)
**Use 4096-8192 dimensions**
- Deep nesting with many fields
- Need reliable discrimination at all levels
- Config drift detection works well at 4096

### Scenario 4: Prototyping / Development
**Use 1024 dimensions**
- Fast iteration, quick feedback
- Can always increase later
- Sufficient for validating approach

## VSA/HDC Theory

The theoretical capacity of hyperdimensional vectors scales approximately with sqrt(d):
- **Bundling capacity**: ~sqrt(d) items can be superposed cleanly
- **Binding capacity**: ~d/log(d) key-value pairs

In practice, Holon's encoding is more robust due to:
- Bipolar thresholding after operations
- Role-based binding (keys bound to values)
- Recursive structure encoding

## Changing Dimensions

Dimensions are set when creating a `VectorManager` or `Holon` and cannot be changed for existing vectors:

```rust
use holon::kernel::VectorManager;

let vm = VectorManager::new(4096);
```

Vectors from different dimensions are incompatible — there is no projection or resize. To change dimensions, re-encode all data.
