<p align="center">
  <a href="https://github.com/wavekat/wavekat-vad">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-vad-narrow.svg" alt="WaveKat VAD">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-vad.svg)](https://crates.io/crates/wavekat-vad)
[![docs.rs](https://docs.rs/wavekat-vad/badge.svg)](https://docs.rs/wavekat-vad)
[![CI](https://github.com/wavekat/wavekat-vad/actions/workflows/ci.yml/badge.svg)](https://github.com/wavekat/wavekat-vad/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1184705840.svg)](https://doi.org/10.5281/zenodo.19216274)

Voice Activity Detection library for Rust with multiple backend support.

## Quick Start

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
let samples: Vec<i16> = vec![0; 160]; // 10ms at 16kHz
let probability = vad.process(&samples, 16000).unwrap();
```

## Backends

| Backend | Feature | Sample Rates | Frame Size | Output |
|---------|---------|-------------|------------|--------|
| WebRTC | `webrtc` (default) | 8/16/32/48 kHz | 10, 20, or 30ms | Binary (0.0 or 1.0) |
| Silero | `silero` | 8/16 kHz | 32ms (256 or 512 samples) | Continuous (0.0–1.0) |
| TEN-VAD | `ten-vad` | 16 kHz only | 16ms (256 samples) | Continuous (0.0–1.0) |
| FireRedVAD | `firered` | 16 kHz only | 10ms (160 samples) | Continuous (0.0–1.0) |

```toml
[dependencies]
wavekat-vad = "0.1"                    # WebRTC only (default)
wavekat-vad = { version = "0.1", features = ["silero"] }
wavekat-vad = { version = "0.1", features = ["ten-vad"] }
wavekat-vad = { version = "0.1", features = ["firered"] }
wavekat-vad = { version = "0.1", features = ["webrtc", "silero", "ten-vad", "firered"] }  # all backends
```

### Benchmarks

Performance measured against the [TEN-VAD testset](https://github.com/TEN-framework/ten-vad/tree/main/testset) — 30 audio files from LibriSpeech, GigaSpeech, and DNS Challenge with manual speech/non-speech annotations. Threshold: 0.5.

<!-- benchmark-table-start -->
*v0.1.14*

| Backend | Precision | Recall | F1 Score | Frame Size | Avg Inference | RTF |
|---------|-----------|--------|----------|------------|---------------|-----|
| WebRTC | 0.821 | 0.983 | 0.895 | 480 (30 ms) | 2.7 µs | 0.0001 |
| Silero | 0.938 | 0.938 | 0.938 | 512 (32 ms) | 118.4 µs | 0.0037 |
| TEN-VAD | 0.942 | 0.915 | 0.928 | 256 (16 ms) | 62.0 µs | 0.0039 |
| FireRedVAD | 0.950 | 0.879 | 0.913 | 160 (10 ms) | 542.8 µs | 0.0543 |
<!-- benchmark-table-end -->

> Accuracy metrics are deterministic; inference times are approximate and vary by hardware. Measured with `--release` on GitHub Actions `ubuntu-latest` runners. Run locally: `make accuracy` or `make bench`

### WebRTC

Google's WebRTC VAD. Fast and lightweight, returns binary speech/silence detection. Supports four aggressiveness modes.

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

// Default 30ms frame duration
let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();

// Or specify frame duration (10, 20, or 30ms)
let mut vad = WebRtcVad::with_frame_duration(16000, WebRtcVadMode::Aggressive, 20).unwrap();

let samples = vec![0i16; 320]; // 20ms at 16kHz
let result = vad.process(&samples, 16000).unwrap(); // 0.0 or 1.0
```

### Silero

Neural network (LSTM) via ONNX Runtime. Returns continuous probability, best overall F1 across benchmarks. Only supports 8kHz and 16kHz.

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::silero::SileroVad;

let mut vad = SileroVad::new(16000).unwrap();
let samples = vec![0i16; 512]; // 32ms at 16kHz
let probability = vad.process(&samples, 16000).unwrap(); // 0.0–1.0

// Or load a custom model
let vad = SileroVad::from_file("path/to/model.onnx", 16000).unwrap();
```

### TEN-VAD

Agora's TEN-VAD with pure Rust preprocessing (no C dependency). Returns continuous probability, 16kHz only.

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::ten_vad::TenVad;

let mut vad = TenVad::new().unwrap();
let samples = vec![0i16; 256]; // 16ms at 16kHz
let probability = vad.process(&samples, 16000).unwrap(); // 0.0–1.0
```

### FireRedVAD

Xiaohongshu's FireRedVAD using a DFSMN architecture with pure Rust FBank preprocessing. Returns continuous probability, 16kHz only.

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::firered::FireRedVad;

let mut vad = FireRedVad::new().unwrap();
let samples = vec![0i16; 160]; // 10ms at 16kHz
let probability = vad.process(&samples, 16000).unwrap(); // 0.0–1.0
```

## The `VoiceActivityDetector` Trait

All backends implement a common trait, so you can write code that is generic over backends:

```rust
use wavekat_vad::{VoiceActivityDetector, VadCapabilities};

fn detect_speech(vad: &mut dyn VoiceActivityDetector, audio: &[i16], sample_rate: u32) {
    let caps = vad.capabilities();
    // caps.sample_rate  — required sample rate
    // caps.frame_size   — required frame size in samples
    // caps.frame_duration_ms — frame duration

    for frame in audio.chunks_exact(caps.frame_size) {
        let probability = vad.process(frame, sample_rate).unwrap();
        if probability > 0.5 {
            println!("Speech detected!");
        }
    }
}
```

## `FrameAdapter`

Real-world audio arrives in arbitrary chunk sizes. `FrameAdapter` buffers incoming samples and feeds correctly-sized frames to the backend automatically.

```rust
use wavekat_vad::FrameAdapter;
use wavekat_vad::backends::silero::SileroVad;

let vad = SileroVad::new(16000).unwrap();
let mut adapter = FrameAdapter::new(Box::new(vad));

// Feed arbitrary-sized chunks — adapter handles buffering
let chunk = vec![0i16; 1000]; // not a multiple of 512

// Get all complete frame results at once
let probabilities = adapter.process_all(&chunk, 16000).unwrap();

// Or get just the latest result (convenient for real-time)
let latest = adapter.process_latest(&chunk, 16000).unwrap();

// Or process one frame at a time
let result = adapter.process(&chunk, 16000).unwrap(); // Some(prob) or None
```

## Preprocessing

Optional audio preprocessing to improve VAD accuracy. Available stages: high-pass filter, noise suppression, and amplitude normalization.

```rust
use wavekat_vad::preprocessing::{Preprocessor, PreprocessorConfig};

// Use a preset
let config = PreprocessorConfig::raw_mic();     // 80Hz HP + normalize + denoise
// let config = PreprocessorConfig::telephony(); // 200Hz HP only

// Or configure manually
let config = PreprocessorConfig {
    high_pass_hz: Some(80.0),       // remove low-frequency rumble
    denoise: false,                  // requires "denoise" feature
    normalize_dbfs: Some(-20.0),     // normalize amplitude
};

let mut preprocessor = Preprocessor::new(&config, 16000);
let raw_audio: Vec<i16> = vec![0; 512];
let cleaned = preprocessor.process(&raw_audio);
// feed `cleaned` to your VAD
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `webrtc` | Yes | WebRTC VAD backend |
| `silero` | No | Silero VAD backend (ONNX model downloaded at build time) |
| `ten-vad` | No | TEN-VAD backend (ONNX model downloaded at build time) |
| `firered` | No | FireRedVAD backend (ONNX model downloaded at build time) |
| `denoise` | No | RNNoise-based noise suppression in the preprocessing pipeline |
| `serde` | No | `Serialize`/`Deserialize` for config types |

### ONNX Model Downloads

Silero, TEN-VAD, and FireRedVAD models are downloaded automatically at build time. The Silero backend is pinned to **v6.2.1** by default.

For offline or CI builds, point to a local model file:

```sh
SILERO_MODEL_PATH=/path/to/silero_vad.onnx cargo build --features silero
TEN_VAD_MODEL_PATH=/path/to/ten-vad.onnx cargo build --features ten-vad
FIRERED_MODEL_PATH=/path/to/fireredvad.onnx FIRERED_CMVN_PATH=/path/to/cmvn.ark cargo build --features firered
```

To use a different Silero model version, override the download URL:

```sh
SILERO_MODEL_URL=https://github.com/snakers4/silero-vad/raw/v6.0/src/silero_vad/data/silero_vad.onnx cargo build --features silero
```

## Error Handling

All backends return `Result<f32, VadError>`. The error type covers:

- `VadError::InvalidSampleRate(u32)` — unsupported sample rate for the backend
- `VadError::InvalidFrameSize { got, expected }` — wrong number of samples
- `VadError::BackendError(String)` — backend-specific error (e.g., ONNX failure)

Use `capabilities()` to check a backend's requirements before processing.

## vad-lab

> **vad-lab has moved to [wavekat/wavekat-lab](https://github.com/wavekat/wavekat-lab).**
>
> It is now a standalone repo so it can grow to cover other WaveKat libraries (turn detection, etc.) without being tied to this crate.

See [wavekat/wavekat-lab](https://github.com/wavekat/wavekat-lab) for setup and usage.

## Videos

| Video | Description |
|---|---|
| <a href="https://www.youtube.com/watch?v=j2KkhpFRKaY"><img src="https://img.youtube.com/vi/j2KkhpFRKaY/maxresdefault.jpg" alt="FireRed VAD Showdown" width="400"></a> | **[Adding FireRedVAD as the 4th backend](https://www.youtube.com/watch?v=j2KkhpFRKaY)** <br> Benchmarking Xiaohongshu's FireRedVAD against Silero, TEN VAD, and WebRTC across accuracy and latency. |
| <a href="https://www.youtube.com/watch?v=450O3w9c-e8"><img src="https://img.youtube.com/vi/450O3w9c-e8/maxresdefault.jpg" alt="VAD Lab Demo" width="400"></a> | **[VAD Lab: Real-time multi-backend comparison](https://www.youtube.com/watch?v=450O3w9c-e8)** <br> Live demo of VAD Lab comparing WebRTC, Silero, and TEN VAD side by side with real-time waveform visualization. |

## License

Apache-2.0

### TEN-VAD model notice

The TEN-VAD ONNX model (used by the `ten-vad` feature) is licensed under Apache-2.0 with a non-compete clause by the TEN-framework / Agora. It restricts deployment that competes with Agora's offerings and limits deployment to "solely for your benefit and the benefit of your direct End Users." This is **not standard open-source** despite the Apache-2.0 label. Review the [TEN-VAD license](https://github.com/TEN-framework/ten-vad) before using in production.

### Acknowledgements

This project wraps and builds on several upstream projects:

- [webrtc-vad](https://github.com/kaegi/webrtc-vad) — Rust bindings for Google's WebRTC VAD
- [Silero VAD](https://github.com/snakers4/silero-vad) — neural network VAD by the Silero team
- [TEN-VAD](https://github.com/TEN-framework/ten-vad) — lightweight VAD by TEN-framework / Agora
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) — DFSMN-based VAD by the FireRedTeam
- [ort](https://github.com/pykeio/ort) — ONNX Runtime bindings for Rust
- [nnnoiseless](https://github.com/jneem/nnnoiseless) — Rust port of RNNoise for noise suppression
