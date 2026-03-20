# WaveKat VAD

[![Crates.io](https://img.shields.io/crates/v/wavekat-vad.svg)](https://crates.io/crates/wavekat-vad)
[![docs.rs](https://docs.rs/wavekat-vad/badge.svg)](https://docs.rs/wavekat-vad)
[![CI](https://github.com/wavekat/wavekat-vad/actions/workflows/ci.yml/badge.svg)](https://github.com/wavekat/wavekat-vad/actions/workflows/ci.yml)

Voice Activity Detection library for Rust with multiple backend support.

## Usage

```rust
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
let samples: Vec<i16> = vec![0; 160]; // 10ms at 16kHz
let probability = vad.process(&samples, 16000).unwrap();
```

## Backends

| Backend | Feature | Description |
|---------|---------|-------------|
| WebRTC | `webrtc` (default) | Google's WebRTC VAD - fast, binary output |
| Silero | `silero` | Neural network via ONNX - higher accuracy, continuous probability |
| TEN-VAD | `ten-vad` | Agora's TEN-VAD via ONNX - pure Rust, no C dependency |

```toml
[dependencies]
wavekat-vad = "0.1"                    # WebRTC only
wavekat-vad = { version = "0.1", features = ["silero"] }
wavekat-vad = { version = "0.1", features = ["ten-vad"] }
```

ONNX models (Silero and TEN-VAD) are downloaded automatically at build time. For offline builds, set `SILERO_MODEL_PATH` or `TEN_VAD_MODEL_PATH` to a local `.onnx` file.

## vad-lab

Dev tool for live VAD experimentation. Captures audio server-side and streams results to a web UI.

<p align="center">
  <img src="docs/images/vad-lab-screenshot.png" alt="vad-lab screenshot" width="700">
  <br>
  <em>vad-lab web interface</em>
</p>

### Quick Start

```sh
make setup         # Install dependencies (once)
make dev-backend   # Terminal 1
make dev-frontend  # Terminal 2
```

## License

Apache-2.0

### TEN-VAD model notice

The TEN-VAD ONNX model (used by the `ten-vad` feature) is licensed under Apache-2.0 with a non-compete clause by the TEN-framework / Agora. It restricts deployment that competes with Agora's offerings and limits deployment to "solely for your benefit and the benefit of your direct End Users." This is **not standard open-source** despite the Apache-2.0 label. Review the [TEN-VAD license](https://github.com/TEN-framework/ten-vad) before using in production.

### Third-party notices

This project uses [nnnoiseless](https://github.com/jneem/nnnoiseless) (BSD-3-Clause) for noise suppression via the `denoise` feature.
