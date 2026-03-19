# Audio Preprocessing Plan

## Problem

Raw microphone audio fed directly into the VAD pipeline produces false positives — ambient noise (HVAC, fans, room rumble) gets classified as speech, even at WebRTC mode 3 (most aggressive). This was observed with MacBook Pro built-in mic in vad-lab.

Telephony audio (G.711, 8kHz) is already bandpass-filtered by the telco network (~300Hz–3.4kHz), but wavekat-vad is a general-purpose crate and must handle noisy raw mic input too.

## Goal

Add a modular, configurable preprocessing pipeline to wavekat-vad that cleans audio before VAD, improving accuracy across diverse input sources.

**Key feature**: Each `VadConfig` includes its own preprocessing settings, allowing side-by-side comparison of VAD results with different preprocessing in vad-lab. Preprocessed waveform and spectrum are displayed in the UI for debugging.

## Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         Per-Config Pipeline         │
                                    │                                     │
Raw Audio ─────┬───────────────────►│ Preprocess ──► VAD ──► VadResult    │
               │                    │     │                               │
               │                    │     └──► Preprocessed Samples ──────┼──► Frontend
               │                    │              │                      │    (waveform)
               │                    │              └──► Spectrum ─────────┼──► Frontend
               │                    └─────────────────────────────────────┘    (spectrogram)
               │
               └──► Raw Spectrum ──────────────────────────────────────────► Frontend
                                                                               (original)
```

Each stage is optional and toggled by configuration. Each config gets its own `Preprocessor` instance with independent state (filter history, etc.).

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

### PreprocessorConfig

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// High-pass filter cutoff in Hz. None = disabled.
    pub high_pass_hz: Option<f32>,
    /// Enable RNNoise denoising (requires `denoise` feature).
    pub denoise: bool,
    /// Normalize to target dBFS. None = disabled.
    pub normalize_dbfs: Option<f32>,
}
```

### Preprocessor

```rust
pub struct Preprocessor {
    high_pass: Option<BiquadFilter>,
    // denoiser: Option<Denoiser>,      // Phase 2
    // normalizer: Option<Normalizer>,  // Phase 3
    sample_rate: u32,
}

impl Preprocessor {
    pub fn new(config: &PreprocessorConfig, sample_rate: u32) -> Self;

    /// Process samples and return preprocessed buffer.
    /// Returns a new Vec (does not mutate input).
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16>;
}
```

### VadConfig Integration (vad-lab)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    pub id: String,
    pub label: String,
    pub backend: String,
    pub params: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub preprocessing: PreprocessorConfig,  // NEW
}
```

### Presets

```rust
impl PreprocessorConfig {
    /// No preprocessing — input is already clean (e.g., telephony)
    pub fn none() -> Self {
        Self::default()
    }

    /// Full pipeline for raw mic input
    pub fn raw_mic() -> Self {
        Self {
            high_pass_hz: Some(80.0),
            denoise: true,
            normalize_dbfs: Some(-20.0),
        }
    }

    /// Telephony preset — light high-pass only
    pub fn telephony() -> Self {
        Self {
            high_pass_hz: Some(200.0),
            ..Default::default()
        }
    }
}
```

## vad-lab Integration

### PipelineResult Changes

```rust
pub struct PipelineResult {
    pub config_id: String,
    pub timestamp_ms: f64,
    pub probability: f32,
    pub preprocessed_samples: Vec<i16>,  // NEW - for visualization
}
```

### WebSocket Protocol Additions

```rust
/// Preprocessed audio for a specific config
PreprocessedAudio {
    config_id: String,
    timestamp_ms: f64,
    samples: Vec<i16>,
}

/// Spectrum of preprocessed audio for a specific config
PreprocessedSpectrum {
    config_id: String,
    timestamp_ms: f64,
    magnitudes: Vec<f32>,
}
```

### Frontend UI

- Each config card shows its preprocessed waveform (small inline view)
- Preprocessed spectrum displayed alongside original for comparison
- ConfigPanel shows preprocessing options:
  - High-pass filter toggle + cutoff Hz slider
  - (Future: denoise toggle, normalize toggle + target dBFS)

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

### Phase 1: High-Pass Filter + Library Foundation
- [x] Create `crates/wavekat-vad/src/preprocessing/mod.rs`
- [x] Create `crates/wavekat-vad/src/preprocessing/biquad.rs`
- [x] Implement `BiquadFilter` (second-order Butterworth high-pass)
- [x] Implement `PreprocessorConfig` with serde support
- [x] Implement `Preprocessor` struct
- [x] Export from `lib.rs`
- [x] Unit tests: verify frequency response, passthrough when disabled

### Phase 2: vad-lab Pipeline Integration
- [x] Update `VadConfig` in `session.rs` to include `preprocessing` field
- [x] Update `PipelineResult` to include `preprocessed_samples`
- [x] Update `pipeline.rs` to create `Preprocessor` per config
- [x] Update `pipeline.rs` to apply preprocessing before VAD
- [x] Update `available_backends()` or add `preprocessing_params()` for frontend

### Phase 3: WebSocket + Visualization
- [x] Add `PreprocessedAudio` and `PreprocessedSpectrum` to `ServerMessage`
- [x] Update `ws.rs` to compute spectrum per-config and send to frontend
- [x] Update frontend `websocket.ts` types
- [x] Update `ConfigPanel.tsx` to show preprocessing options
- [ ] Update visualization components to show per-config waveform/spectrum

### Phase 4: Noise Suppression (Optional)
- [ ] Add `nnnoiseless` dependency behind `denoise` feature
- [ ] Create `preprocessing/denoise.rs` wrapper
- [ ] Integrate into `Preprocessor` chain
- [ ] Test with vad-lab: compare raw vs. denoised VAD results side-by-side

### Phase 5: Normalizer (Optional)
- [ ] Implement RMS-based normalizer
- [ ] Add to `Preprocessor` chain
- [ ] Test with varying input levels

### Phase 6: Resampler (Optional)
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
| Config location | Per `VadConfig` | Allows A/B testing different preprocessing settings |
| Preprocessor ownership | Per-config instance | Filter state must be independent per config |
| Sample mutation | Return new Vec | Don't mutate input, allows original to be displayed |

## Testing & Verification

1. **Unit tests**: Verify BiquadFilter frequency response (pass high freqs, attenuate low freqs)
2. **vad-lab test**: Create two configs - one with high-pass at 80Hz, one without
3. **Visual verification**:
   - Record MacBook mic with ambient noise
   - Compare raw vs. preprocessed spectrum - should see low frequencies reduced
   - Compare VAD results - preprocessed should have fewer false positives
4. **Expected result**: High-pass config should show attenuated low-frequency energy in spectrum

## Open Questions (Resolved)

- ~~Should `VoiceActivityDetector` trait change to return `VadResult` instead of `f32`?~~ — No, keep trait simple. Preprocessing is separate.
- ~~Do we need per-frame energy calculation even without normalization?~~ — Yes, useful for vad-lab visualization. Include in preprocessed output.
- ~~Should the preprocessor be part of the library crate or a separate crate?~~ — Part of library crate, simpler for users.

## File Locations

| File | Purpose |
|------|---------|
| `crates/wavekat-vad/src/preprocessing/mod.rs` | Preprocessor struct and config |
| `crates/wavekat-vad/src/preprocessing/biquad.rs` | High-pass filter implementation |
| `crates/wavekat-vad/src/lib.rs` | Export preprocessing module |
| `tools/vad-lab/backend/src/session.rs` | VadConfig with preprocessing |
| `tools/vad-lab/backend/src/pipeline.rs` | Apply preprocessing, return samples |
| `tools/vad-lab/backend/src/ws.rs` | Send preprocessed audio + spectrum |
| `tools/vad-lab/frontend/src/lib/websocket.ts` | New message types |
| `tools/vad-lab/frontend/src/components/ConfigPanel.tsx` | Preprocessing UI |
