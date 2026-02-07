//! Benchmarks for Holon operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holon::{Holon, ScalarMode};

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

criterion_group!(
    benches,
    benchmark_vector_generation,
    benchmark_bind,
    benchmark_bundle,
    benchmark_similarity,
    benchmark_encode_json,
    benchmark_scalar_encoding,
    benchmark_accumulator,
);

criterion_main!(benches);
