//! Benchmarks for Holon operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holon::{Holon, ScalarMode, ScalarValue, WalkType, Walkable, WalkableValue};

// =============================================================================
// Walkable Packet for benchmarking
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

    // Same data structure, two encoding paths
    let packet = BenchPacket {
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

    group.bench_function("walkable_path", |b| {
        b.iter(|| holon.encode_walkable(black_box(&packet)))
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
);

criterion_main!(benches);
