use criterion::{criterion_group, criterion_main, Criterion};

fn vad_benchmarks(_c: &mut Criterion) {
    // TODO: Add benchmarks once backends are fully implemented
}

criterion_group!(benches, vad_benchmarks);
criterion_main!(benches);
