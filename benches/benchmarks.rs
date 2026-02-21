//! Benchmarks for Holon operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holon::highlevel::Holon;
use holon::kernel::{ScalarMode, ScalarValue, WalkType, Walkable, WalkableRef, WalkableValue};
use holon::memory::{EngramLibrary, OnlineSubspace};

// =============================================================================
// Walkable Packet for benchmarking (with fast visitor)
// =============================================================================

#[derive(Clone)]
struct BenchPacket {
    protocol: String,
    src_port: u16,
    dst_port: u16,
    flags: String,
    payload_size: u32,
}

impl Walkable for BenchPacket {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    // Slow path (for fallback)
    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String(self.protocol.clone())),
            ),
            ("src_port", (self.src_port as i64).to_walkable_value()),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            (
                "flags",
                WalkableValue::Scalar(ScalarValue::String(self.flags.clone())),
            ),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
        ]
    }

    // Fast path - zero allocation!
    fn walk_map_visitor(&self, visitor: &mut dyn FnMut(&str, WalkableRef<'_>)) {
        visitor("protocol", WalkableRef::string(&self.protocol));
        visitor("src_port", WalkableRef::int(self.src_port as i64));
        visitor("dst_port", WalkableRef::int(self.dst_port as i64));
        visitor("flags", WalkableRef::string(&self.flags));
        visitor("payload_size", WalkableRef::int(self.payload_size as i64));
    }

    fn has_fast_visitor(&self) -> bool {
        true
    }
}

// Slow packet that doesn't implement fast visitor (for comparison)
#[derive(Clone)]
struct SlowPacket {
    protocol: String,
    src_port: u16,
    dst_port: u16,
    flags: String,
    payload_size: u32,
}

impl Walkable for SlowPacket {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String(self.protocol.clone())),
            ),
            ("src_port", (self.src_port as i64).to_walkable_value()),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            (
                "flags",
                WalkableValue::Scalar(ScalarValue::String(self.flags.clone())),
            ),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
        ]
    }
}

fn benchmark_vector_generation(c: &mut Criterion) {
    let holon = Holon::new(4096);

    c.bench_function("get_vector", |b| {
        b.iter(|| holon.get_vector(black_box("test_atom")))
    });
}

fn benchmark_bind(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let vec_a = holon.get_vector("A");
    let vec_b = holon.get_vector("B");

    c.bench_function("bind", |b| b.iter(|| holon.bind(black_box(&vec_a), black_box(&vec_b))));
}

fn benchmark_bundle(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let vectors: Vec<_> = (0..10).map(|i| holon.get_vector(&format!("vec_{}", i))).collect();
    let refs: Vec<_> = vectors.iter().collect();

    c.bench_function("bundle_10", |b| b.iter(|| holon.bundle(black_box(&refs))));
}

fn benchmark_similarity(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let vec_a = holon.get_vector("A");
    let vec_b = holon.get_vector("B");

    c.bench_function("similarity", |b| {
        b.iter(|| holon.similarity(black_box(&vec_a), black_box(&vec_b)))
    });
}

fn benchmark_encode_json(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let json = r#"{"type": "billing", "amount": 100, "customer": "acme"}"#;

    c.bench_function("encode_json", |b| {
        b.iter(|| holon.encode_json(black_box(json)))
    });
}

fn benchmark_scalar_encoding(c: &mut Criterion) {
    let holon = Holon::new(4096);

    c.bench_function("encode_scalar_linear", |b| {
        b.iter(|| holon.encode_scalar(black_box(50.0), ScalarMode::Linear { scale: 100.0 }))
    });

    c.bench_function("encode_scalar_log", |b| {
        b.iter(|| holon.encode_scalar_log(black_box(1000.0)))
    });
}

fn benchmark_accumulator(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let vec = holon.get_vector("example");

    c.bench_function("accumulator_add", |b| {
        b.iter(|| {
            let mut acc = holon.create_accumulator();
            for _ in 0..100 {
                holon.accumulate(&mut acc, black_box(&vec));
            }
            acc
        })
    });
}

// =============================================================================
// Walkable vs JSON Encoding
// =============================================================================

fn benchmark_encode_walkable(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let packet = BenchPacket {
        protocol: "TCP".to_string(),
        src_port: 443,
        dst_port: 8080,
        flags: "PA".to_string(),
        payload_size: 1200,
    };

    c.bench_function("encode_walkable", |b| {
        b.iter(|| holon.encode_walkable(black_box(&packet)))
    });
}

fn benchmark_walkable_vs_json(c: &mut Criterion) {
    let holon = Holon::new(4096);

    // Same data structure, three encoding paths
    let fast_packet = BenchPacket {
        protocol: "TCP".to_string(),
        src_port: 443,
        dst_port: 8080,
        flags: "PA".to_string(),
        payload_size: 1200,
    };

    let slow_packet = SlowPacket {
        protocol: "TCP".to_string(),
        src_port: 443,
        dst_port: 8080,
        flags: "PA".to_string(),
        payload_size: 1200,
    };

    let json = r#"{"protocol":"TCP","src_port":443,"dst_port":8080,"flags":"PA","payload_size":1200}"#;

    let mut group = c.benchmark_group("encoding_comparison");

    group.bench_function("json_path", |b| {
        b.iter(|| holon.encode_json(black_box(json)))
    });

    group.bench_function("walkable_slow", |b| {
        b.iter(|| holon.encode_walkable(black_box(&slow_packet)))
    });

    group.bench_function("walkable_fast", |b| {
        b.iter(|| holon.encode_walkable(black_box(&fast_packet)))
    });

    group.finish();
}

fn benchmark_json_string_building(c: &mut Criterion) {
    let packet = BenchPacket {
        protocol: "TCP".to_string(),
        src_port: 443,
        dst_port: 8080,
        flags: "PA".to_string(),
        payload_size: 1200,
    };

    c.bench_function("json_string_building", |b| {
        b.iter(|| {
            format!(
                r#"{{"protocol":"{}","src_port":{},"dst_port":{},"flags":"{}","payload_size":{}}}"#,
                black_box(&packet.protocol),
                black_box(packet.src_port),
                black_box(packet.dst_port),
                black_box(&packet.flags),
                black_box(packet.payload_size)
            )
        })
    });
}

// =============================================================================
// Memory Layer Benchmarks
// =============================================================================

fn benchmark_subspace_update(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let mut subspace = holon.create_subspace(32);

    // Pre-train so we measure steady-state performance, not initialization
    let warmup_vec: Vec<f64> = (0..4096).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    for _ in 0..50 {
        subspace.update(&warmup_vec);
    }

    // Bench update with a varying vector
    let bench_vec: Vec<f64> = (0..4096).map(|i| (i as f64).sin()).collect();
    c.bench_function("subspace_update", |b| {
        b.iter(|| subspace.update(black_box(&bench_vec)))
    });
}

fn benchmark_subspace_residual(c: &mut Criterion) {
    let holon = Holon::new(4096);
    let mut subspace = holon.create_subspace(32);

    // Train
    for i in 0..200 {
        let v: Vec<f64> = (0..4096).map(|j| ((i * j) as f64).sin()).collect();
        subspace.update(&v);
    }

    let probe: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.1).cos()).collect();
    c.bench_function("subspace_residual", |b| {
        b.iter(|| subspace.residual(black_box(&probe)))
    });
}

fn benchmark_engram_match(c: &mut Criterion) {
    let dim = 4096;

    // Create 10 engrams from different distributions
    let mut library = EngramLibrary::new(dim);
    for pattern in 0..10usize {
        let mut sub = OnlineSubspace::new(dim, 16);
        for i in 0..100usize {
            let v: Vec<f64> = (0..dim)
                .map(|j| ((pattern * 100 + i + j) as f64 * 0.01).sin())
                .collect();
            sub.update(&v);
        }
        library.add(&format!("pattern_{}", pattern), &sub, None, Default::default());
    }

    let probe: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.01).sin()).collect();
    c.bench_function("engram_match_10", |b| {
        b.iter(|| library.match_vec(black_box(&probe), 3, 10))
    });
}

// =============================================================================
// TimeScale Encoding Benchmarks
// =============================================================================

fn benchmark_encode_time(c: &mut Criterion) {
    let holon = Holon::new(4096);

    c.bench_function("encode_time", |b| {
        b.iter(|| {
            holon.encode_walkable_value(&holon::WalkableValue::Scalar(holon::ScalarValue::time(
                black_box(1_700_000_000.0),
            )))
        })
    });
}

criterion_group!(
    benches,
    benchmark_vector_generation,
    benchmark_bind,
    benchmark_bundle,
    benchmark_similarity,
    benchmark_encode_json,
    benchmark_scalar_encoding,
    benchmark_accumulator,
    benchmark_encode_walkable,
    benchmark_walkable_vs_json,
    benchmark_json_string_building,
    benchmark_subspace_update,
    benchmark_subspace_residual,
    benchmark_engram_match,
    benchmark_encode_time,
);

criterion_main!(benches);
