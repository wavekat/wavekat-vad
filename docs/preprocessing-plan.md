# Audio Preprocessing Plan

## Problem

Raw microphone audio fed directly into the VAD pipeline produces false positives — ambient noise (HVAC, fans, room rumble) gets classified as speech, even at WebRTC mode 3 (most aggressive). This was observed with MacBook Pro built-in mic in vad-lab.

Telephony audio (G.711, 8kHz) is already bandpass-filtered by the telco network (~300Hz–3.4kHz), but wavekat-vad is a general-purpose crate and must handle noisy raw mic input too.

## Goal

Add a modular, configurable preprocessing pipeline to wavekat-vad that cleans audio before VAD, improving accuracy across diverse input sources.

## Architecture

```
Raw Audio → [Resample] → [High-Pass Filter] → [Noise Suppression] → [Normalize] → VAD
```

Each stage is optional and toggled by configuration. Users pick presets or build custom chains.

## Pipeline Stages

### 1. Resampler

Convert input to a configurable target sample rate. Input sources vary widely (44.1kHz, 48kHz from mics; 8kHz from telephony) and different VAD backends have different requirements.

- Target sample rate is configurable — not hardcoded to any specific rate
- Wrap `rubato` crate (pure Rust, high quality) behind a `resample` feature flag
- Common targets: 16kHz (Silero), 8/16/32/48kHz (WebRTC)

### 2. High-Pass Filter

Remove low-frequency energy (< 80–100Hz) that triggers false positives. Most ambient noise lives here; human speech fundamental starts ~85Hz (male) / ~165Hz (female).

- Implement as a second-order Butterworth biquad filter
- Configurable cutoff frequency (default ~80Hz for raw mic, ~200Hz for telephony)
- Pure Rust, no dependencies — just a struct with 4 state variables

### 3. Noise Suppression

Suppress stationary background noise while preserving speech. This is the main fix for the MacBook mic problem.

- Wrap `nnnoiseless` (pure Rust port of RNNoise) behind a `denoise` feature flag
- RNNoise uses a small RNN — runs well under 1ms per frame
- Optional: the crate also provides a VAD signal we could use as a secondary input

### 4. Normalizer

Normalize amplitude so VAD thresholds work consistently regardless of input gain.

- RMS-based normalization to a target level (e.g., -20 dBFS)
- Optional peak limiting to prevent clipping after gain adjustment
- Important: normalize *after* noise suppression, not before (otherwise we amplify noise)

## API Design

### Pipeline Struct

```rust
pub struct VadPipeline {
    preprocessor: Preprocessor,
    vad: Box<dyn VoiceActivityDetector>,
}

pub struct Preprocessor {
    high_pass: Option<BiquadFilter>,
    denoiser: Option<Denoiser>,
    normalizer: Option<Normalizer>,
}
```

### Configuration

```rust
pub struct PreprocessorConfig {
    pub resample_target_hz: Option<u32>,    // None = disabled, e.g. Some(16000)
    pub high_pass_cutoff_hz: Option<f32>,   // None = disabled
    pub denoise: bool,                       // requires `denoise` feature
    pub normalize_target_dbfs: Option<f32>,  // None = disabled
}
```

### Presets

```rust
impl PreprocessorConfig {
    /// No preprocessing — input is already clean (e.g., telephony)
    pub fn none() -> Self;

    /// Full pipeline for raw mic input
    pub fn raw_mic() -> Self;

    /// Telephony preset — light high-pass only
    pub fn telephony() -> Self;
}
```

### Richer Result Type

```rust
pub struct VadResult {
    pub is_speech: bool,
    pub speech_probability: f32,
    pub energy_dbfs: f32,          // pre-VAD energy, useful for debugging
}
```

## Feature Flags

```toml
[features]
default = ["webrtc"]
webrtc = ["dep:webrtc-vad"]
silero = ["dep:ort"]
denoise = ["dep:nnnoiseless"]     # NEW — noise suppression
resample = ["dep:rubato"]         # NEW — high-quality resampling
```

Keep the crate lightweight by default. Users opt in to heavier preprocessing.

## Implementation Order

### Phase 1: High-Pass Filter
- [ ] Implement `BiquadFilter` (second-order Butterworth high-pass)
- [ ] Add `Preprocessor` struct with high-pass stage
- [ ] Wire into vad-lab for testing with MacBook mic
- [ ] Unit tests: verify frequency response, passthrough when disabled

### Phase 2: Noise Suppression
- [ ] Add `nnnoiseless` dependency behind `denoise` feature
- [ ] Implement `Denoiser` wrapper
- [ ] Integrate into `Preprocessor` chain
- [ ] Test with vad-lab: compare raw vs. denoised VAD results side-by-side

### Phase 3: Normalizer
- [ ] Implement RMS-based normalizer
- [ ] Add to `Preprocessor` chain
- [ ] Test with varying input levels

### Phase 4: VadPipeline + Presets
- [ ] Create `VadPipeline` that owns preprocessor + VAD backend
- [ ] Add `VadResult` return type
- [ ] Implement presets (`none`, `raw_mic`, `telephony`)
- [ ] Update vad-lab to use `VadPipeline`

### Phase 5: Resampler
- [ ] Add `rubato` dependency behind `resample` feature flag
- [ ] Implement configurable target sample rate resampling
- [ ] Unit tests: verify sample rate conversion accuracy

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Denoiser | `nnnoiseless` (RNNoise port) | Pure Rust, no FFI, < 1ms/frame, proven quality |
| High-pass filter | Biquad (in-crate) | Trivial to implement, zero dependencies |
| Resampler | `rubato` (optional) | High quality, pure Rust, but may not be needed |
| Pipeline position | Before VAD, after capture | Clean audio = better VAD accuracy |
| Normalization order | After denoise | Avoid amplifying noise before suppression |

## Open Questions

- Should `VoiceActivityDetector` trait change to return `VadResult` instead of `f32`? This is a breaking API change — may want to add a new trait or method instead.
- Do we need per-frame energy calculation even without normalization? Useful for vad-lab visualization.
- Should the preprocessor be part of the library crate or a separate `wavekat-vad-preprocess` crate?
