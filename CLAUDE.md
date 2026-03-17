# WaveKat VAD — Project Instructions

Voice Activity Detection library for Rust. This crate provides a unified interface for multiple VAD backends, enabling experimentation and benchmarking across different implementations.

## Project Goals

1. **Experimentation First** — Test and compare different VAD technologies before settling on a final API
2. **Unified Interface** — Common trait abstraction over multiple backends
3. **Benchmarking** — Measure accuracy, latency, and resource usage across implementations
4. **Publication** — Eventually publish to crates.io as a standalone crate

## Repository Structure

This is a Cargo workspace with two main pieces: the library crate (the product) and a dev tool for experimentation.

```
wavekat-vad/
├── Cargo.toml                  # workspace root
├── crates/
│   └── wavekat-vad/            # library crate (the product)
│       ├── src/
│       │   ├── lib.rs          # Public API, trait definitions
│       │   ├── frame.rs        # AudioFrame type (i16/f32 samples)
│       │   ├── error.rs        # Error types
│       │   └── backends/
│       │       ├── mod.rs
│       │       ├── webrtc.rs   # webrtc-vad wrapper
│       │       └── silero.rs   # silero-vad (ONNX) wrapper
│       └── Cargo.toml
├── tools/
│   └── vad-lab/                # dev tool for experimentation
│       ├── backend/            # Rust: axum + cpal + websocket
│       │   ├── src/
│       │   │   ├── main.rs
│       │   │   ├── audio_source.rs  # mic capture (cpal) / file playback
│       │   │   ├── pipeline.rs      # fan-out to N VAD configs
│       │   │   ├── session.rs       # config & result persistence
│       │   │   └── ws.rs            # WebSocket streaming
│       │   ├── build.rs        # embed frontend dist/
│       │   └── Cargo.toml
│       └── frontend/           # React app
│           ├── src/
│           │   ├── components/
│           │   │   ├── Waveform.tsx       # canvas waveform display
│           │   │   ├── VadTimeline.tsx    # speech probability overlay
│           │   │   ├── ConfigPanel.tsx    # VAD config editor
│           │   │   └── SessionManager.tsx
│           │   └── lib/
│           │       ├── websocket.ts       # WS client
│           │       └── audio.ts           # audio decoding helpers
│           └── package.json
├── testdata/                   # audio samples for testing
│   ├── speech/
│   └── silence/
├── docs/
│   ├── plan.md                 # implementation plan
│   └── experiments/            # experiment notes and results
├── benches/
│   └── vad_comparison.rs
└── tests/
    └── integration.rs
```

## Core Trait

```rust
pub trait VoiceActivityDetector: Send {
    /// Process audio frame, return probability of speech (0.0 - 1.0)
    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError>;

    /// Reset internal state
    fn reset(&mut self);
}
```

## VAD Backends to Explore

### Phase 1: Initial Implementation
- [ ] **webrtc-vad** — Google's WebRTC VAD (fast, low accuracy)
- [ ] **silero-vad** — Neural network based (higher accuracy, requires ONNX runtime)

### Phase 2: Additional Backends (optional)
- [ ] **rnnoise** — RNN-based noise suppression with VAD
- [ ] **custom threshold** — Simple energy-based detection for baseline

## vad-lab — Experimentation Tool

A web-based dev tool for live VAD experimentation. **Not a product — just a tool to help us understand VAD better.**

### Two Modes

1. **Live recording** — capture mic server-side via `cpal`, stream audio + VAD results to the browser in real-time
2. **File replay** — load a WAV file, run VAD configs on it, display full timeline

### Architecture

- **Server (Rust)**: axum + WebSocket. Handles mic capture, audio processing, VAD pipeline. All audio stays server-side.
- **Frontend (React)**: visualization only. Waveform display, VAD result overlays, config panel, session management.
- **Single binary**: frontend assets embedded in the Rust binary via `rust-embed`. Run one command, opens browser.

### Server-Side Audio Capture

The server handles all audio recording (not the browser). Flow:
1. Client requests device list
2. Server returns available mic devices via `cpal`
3. Client selects device and starts recording
4. Server captures audio, feeds frames to VAD pipeline, streams results via WebSocket
5. Client displays waveform + VAD results in real-time

### WebSocket Protocol

```
Server → Client:
  { type: "devices", devices: [{ id, name, sample_rates }] }
  { type: "audio", timestamp, samples: [...] }
  { type: "vad", timestamp, config_id, probability }
  { type: "done" }

Client → Server:
  { type: "list_devices" }
  { type: "start_recording", device_id, sample_rate }
  { type: "stop_recording" }
  { type: "load_file", path: "..." }
  { type: "set_configs", configs: [...] }
```

### Multi-Config Pipeline

- Each VAD config specifies: backend name, backend-specific params, human label
- Audio frames are fanned out to N `VoiceActivityDetector` instances (one per config)
- Each runs in its own task/thread
- Results are streamed to the frontend and saved to session files

## Testing

- **Every module must have unit tests.** Use `#[cfg(test)] mod tests { ... }`.
- Use `#[tokio::test]` for async test functions if needed.
- Test edge cases: empty frames, invalid sample rates, malformed input.
- Backend tests should verify consistent behavior across implementations.
- Place test audio files in `testdata/` (keep them small, < 1MB each).

## Code Quality

- `cargo fmt --all --check` — No formatting issues
- `cargo build` — Clean build
- `cargo test` — All tests pass
- `cargo clippy --workspace -- -D warnings` — No warnings
- No `unwrap()` in library code — only in tests and examples
- Use `thiserror` for error types
- Add `///` doc comments on all public items

## Dependencies Policy

### Library crate (`crates/wavekat-vad`)

Keep dependencies minimal:
- **Required**: Only what's needed for core functionality
- **Optional**: Backend-specific deps behind feature flags
- **Dev-only**: Benchmarking and testing tools

```toml
[features]
default = ["webrtc"]
webrtc = ["webrtc-vad"]
silero = ["ort"]  # ONNX Runtime

[dependencies]
thiserror = "2"
webrtc-vad = { version = "0.4", optional = true }
ort = { version = "2", optional = true }

[dev-dependencies]
criterion = "0.5"
hound = "3.5"  # WAV file reading
```

### vad-lab tool (`tools/vad-lab`)

Heavier dependencies are fine here — it's a dev tool, not shipped to users:
- `axum`, `tokio`, `tower-http` — web server
- `cpal` — audio capture
- `rust-embed` — embed frontend assets
- `serde`, `serde_json` — serialization
- `hound` — WAV file I/O
- `clap` — CLI args

## Conventions

- Experiment docs go in `docs/experiments/` with format `YYYY-MM-DD-topic.md`
- Each experiment should document: goal, method, results, conclusions
- Feature branches for each backend: `feat/webrtc-vad`, `feat/silero-vad`
- Keep the main branch stable and buildable

## Audio Format Notes

- Primary format: 16-bit signed integers (i16), mono
- Supported sample rates: 8000, 16000, 32000, 48000 Hz
- Frame sizes vary by backend (typically 10-30ms)
- webrtc-vad: requires specific frame sizes (10, 20, or 30ms)
- silero-vad: more flexible, typically 30-96ms chunks

## Publication Checklist (for later)

- [ ] Clean public API with minimal surface area
- [ ] Comprehensive documentation
- [ ] README with usage examples
- [ ] CHANGELOG.md
- [ ] LICENSE (MIT or Apache-2.0)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Version 0.1.0 release
