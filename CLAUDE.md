# WaveKat VAD — Project Instructions

Voice Activity Detection library for Rust. This crate provides a unified interface for multiple VAD backends, enabling experimentation and benchmarking across different implementations.

## Project Goals

1. **Experimentation First** — Test and compare different VAD technologies before settling on a final API
2. **Unified Interface** — Common trait abstraction over multiple backends
3. **Benchmarking** — Measure accuracy, latency, and resource usage across implementations
4. **Publication** — Eventually publish to crates.io as a standalone crate

## Repository Structure

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

## Conventions

- Experiment docs go in `docs/experiments/` with format `YYYY-MM-DD-topic.md`
- Each experiment should document: goal, method, results, conclusions
- Feature branches for each backend: `feat/webrtc-vad`, `feat/silero-vad`
- Keep the main branch stable and buildable
- **Always use squash merge** when merging feature branches into main (keeps history clean)
- **PR titles must use conventional commit format, max 50 characters** — e.g. `feat: add silero backend`, `fix: handle empty frames`. This is used as the squash merge commit message, and release-plz uses it to generate a nicely grouped changelog (Features, Bug Fixes, etc.). Keep details in the PR body, not the title.

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
