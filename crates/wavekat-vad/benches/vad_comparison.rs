use criterion::{criterion_group, criterion_main, Criterion};

fn vad_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_process");

    // Benchmark each backend's per-frame inference time.
    // All backends process silence frames to measure pure computation overhead.

    #[cfg(feature = "webrtc")]
    {
        use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
        use wavekat_vad::VoiceActivityDetector;

        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
        let samples = vec![0i16; vad.capabilities().frame_size];

        group.bench_function("webrtc", |b| {
            b.iter(|| vad.process(criterion::black_box(&samples), 16000).unwrap())
        });
    }

    #[cfg(feature = "silero")]
    {
        use wavekat_vad::backends::silero::SileroVad;
        use wavekat_vad::VoiceActivityDetector;

        let mut vad = SileroVad::new(16000).unwrap();
        let samples = vec![0i16; vad.capabilities().frame_size];

        group.bench_function("silero", |b| {
            b.iter(|| vad.process(criterion::black_box(&samples), 16000).unwrap())
        });
    }

    #[cfg(feature = "ten-vad")]
    {
        use wavekat_vad::backends::ten_vad::TenVad;
        use wavekat_vad::VoiceActivityDetector;

        let mut vad = TenVad::new().unwrap();
        let samples = vec![0i16; vad.capabilities().frame_size];

        group.bench_function("ten_vad", |b| {
            b.iter(|| vad.process(criterion::black_box(&samples), 16000).unwrap())
        });
    }

    #[cfg(feature = "firered")]
    {
        use wavekat_vad::backends::firered::FireRedVad;
        use wavekat_vad::VoiceActivityDetector;

        let mut vad = FireRedVad::new().unwrap();
        // FireRedVad needs 3 frames to produce the first result (buffering 400 samples)
        let warmup = vec![0i16; 160];
        let _ = vad.process(&warmup, 16000).unwrap();
        let _ = vad.process(&warmup, 16000).unwrap();
        let samples = vec![0i16; vad.capabilities().frame_size];

        group.bench_function("fireredvad", |b| {
            b.iter(|| vad.process(criterion::black_box(&samples), 16000).unwrap())
        });
    }

    group.finish();
}

criterion_group!(benches, vad_benchmarks);
criterion_main!(benches);
