# FireRedVAD Implementation Plan

## Overview

Integrate [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) as a new backend for wavekat-vad. FireRedVAD is a state-of-the-art VAD from Xiaohongshu (released March 2026) that outperforms Silero, TEN-VAD, and WebRTC across multiple benchmarks. It uses a DFSMN (Deep Feedforward Sequential Memory Network) architecture with ~0.6M parameters.

## Why FireRedVAD

| Metric (FLEURS-VAD-102) | FireRedVAD | Silero | TEN-VAD | WebRTC |
|--------------------------|-----------|--------|---------|--------|
| AUC-ROC                  | **99.60** | 97.99  | 97.81   | —      |
| F1 Score                 | **97.57** | 95.95  | 95.19   | 52.30  |
| False Alarm Rate         | **2.69**  | 9.41   | 15.47   | 2.83   |
| Miss Rate                | 3.62      | 3.95   | 2.95    | 64.15  |

- Best overall F1 and AUC-ROC
- Lowest false alarm rate
- Apache-2.0 license (no restrictions unlike TEN-VAD)
- Tiny model (~2.2 MB, ~0.6M params)
- Pre-exported ONNX models available — fits our existing `ort` infrastructure

## Key Differences from Existing Backends

| Aspect | WebRTC | Silero | TEN-VAD | FireRedVAD |
|--------|--------|--------|---------|------------|
| Output | Binary (0/1) | Continuous (0.0-1.0) | Continuous (0.0-1.0) | Continuous (0.0-1.0) |
| Sample rate | 8k/16k/32k/48k | 8k/16k | 16k only | 16k only |
| Frame size | 10/20/30ms | 32ms | 16ms | 10ms (160 samples) |
| State | Minimal | LSTM h/c [2,1,128] | 4x hidden [1,64] | DFSMN caches [8,1,128,19] |
| Preprocessing | None | i16→f32 normalize | Pre-emphasis + STFT + mel + pitch | **FBank (80-dim) + CMVN** |
| Model size | N/A (rule-based) | ~2 MB | ~0.5 MB | ~2.2 MB |

## Model Details

### Architecture: DFSMN

- 8 DFSMN blocks, 1 DNN layer
- Hidden size 256, projection size 128
- Input: 80-dim log Mel filterbank (FBank) features
- Streaming model uses causal convolutions (lookback only, no lookahead)

### ONNX Model Variants (pre-exported in repo)

| Model | File | Input | Output | Use Case |
|-------|------|-------|--------|----------|
| Non-streaming VAD | `fireredvad_vad.onnx` | `feat [B,T,80]` | `probs [B,T,1]` | File processing |
| Streaming VAD (with cache) | `fireredvad_stream_vad_with_cache.onnx` | `feat [1,T,80]` + `caches_in [8,1,128,19]` | `probs [1,T,1]` + `caches_out [8,1,128,19]` | Real-time / frame-by-frame |
| Streaming VAD (no cache input) | `fireredvad_stream_vad.onnx` | `feat [B,T,80]` | `probs [B,T,1]` + cache tensors | First-call only variant |
| AED (3-class) | `fireredvad_aed.onnx` | `feat [B,T,80]` | `probs [B,T,3]` | Speech/singing/music classification |

**We will use `fireredvad_stream_vad_with_cache.onnx`** for the streaming backend — it fits our frame-by-frame `VoiceActivityDetector` trait and carries state via explicit cache tensors (similar to Silero's LSTM state).

### Preprocessing Pipeline (must implement in Rust)

1. **FBank feature extraction** (80-dim log Mel filterbank)
   - Window: 25ms (400 samples at 16kHz)
   - Hop: 10ms (160 samples at 16kHz)
   - 80 Mel filters
   - This is the most complex piece — similar to TEN-VAD's mel filterbank but with different parameters

2. **CMVN normalization** (Cepstral Mean-Variance Normalization)
   - Read per-dimension mean/variance from `cmvn.ark` (Kaldi format)
   - Apply: `(feature - mean) / sqrt(variance)`
   - The `cmvn.ark` file is small (~1.3 KB) — embed at compile time

### Frame Mapping

Each call to `process()` with 160 i16 samples (10ms at 16kHz):
1. Buffer into 25ms windows (400 samples) with 10ms hop → produces 1 FBank frame per call
2. Apply CMVN → 1x80 feature vector
3. Run ONNX inference with `feat [1,1,80]` + `caches_in [8,1,128,19]`
4. Return speech probability from `probs [1,1,1]`
5. Store `caches_out` for next call

## Implementation Plan

### Step 1: Add Feature Flag and Dependencies

**File: `crates/wavekat-vad/Cargo.toml`**

```toml
[features]
firered = ["dep:ort", "dep:ndarray", "dep:realfft", "dep:ureq"]

# realfft is already a dependency (used by ten-vad for FFT)
# ndarray is already a dependency (used by silero and ten-vad)
```

No new dependencies needed — `ort`, `ndarray`, and `realfft` are already in the workspace for TEN-VAD.

### Step 2: Download ONNX Model + CMVN in build.rs

**File: `crates/wavekat-vad/build.rs`**

Add `setup_firered_model()` following the existing pattern:

- Download `fireredvad_stream_vad_with_cache.onnx` from the GitHub repo
- Download `cmvn.ark` from the GitHub repo
- Support `FIRERED_MODEL_PATH` and `FIRERED_CMVN_PATH` env vars for offline builds
- Write both files to `OUT_DIR`

Source URLs:
- Model: `https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx`
- CMVN: `https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/cmvn.ark`

### Step 3: Implement CMVN Parser

**File: `crates/wavekat-vad/src/backends/firered/cmvn.rs`**

Parse the Kaldi-format `cmvn.ark` file embedded at compile time:
- Format: text matrix with accumulated counts, means, and variances
- Output: per-dimension mean and inverse-std vectors (80 floats each)
- Apply as `(feature - mean) * inv_std` per dimension

Reference: The Python implementation reads this via `kaldiio.load_mat()` — we need a minimal Kaldi text matrix parser.

### Step 4: Implement FBank Feature Extraction

**File: `crates/wavekat-vad/src/backends/firered/fbank.rs`**

80-dim log Mel filterbank, matching FireRedVAD's `kaldi_native_fbank` configuration:

1. **Windowing**: 25ms Hamming/Hann window (400 samples), 10ms hop (160 samples)
2. **FFT**: 512-point real FFT using `realfft` crate (already a dependency)
3. **Power spectrum**: |FFT|²
4. **Mel filterbank**: 80 triangular filters, 0-8000 Hz, mel-scale spacing
5. **Log compression**: `ln(max(energy, floor))`

This is similar to the TEN-VAD preprocessing in `ten_vad.rs` (which does 40-band mel) but with different parameters:
- 80 bands instead of 40
- 512 FFT instead of 1024
- Different window size (400 vs 768)

We can reuse patterns from the TEN-VAD mel filterbank code but adjust the constants.

### Step 5: Implement FireRedVAD Backend

**File: `crates/wavekat-vad/src/backends/firered/mod.rs`**

```rust
pub struct FireRedVad {
    session: Session,
    /// DFSMN cache state: shape [8, 1, 128, 19]
    caches: Array4<f32>,
    /// FBank feature extractor
    fbank: FbankExtractor,
    /// CMVN mean/inv_std vectors (80-dim each)
    cmvn_mean: Vec<f32>,
    cmvn_inv_std: Vec<f32>,
    /// Overlap buffer for windowed FFT (last 240 samples from previous frame)
    window_buffer: Vec<f32>,
}

impl VoiceActivityDetector for FireRedVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: 16000,
            frame_size: 160,       // 10ms at 16kHz
            frame_duration_ms: 10,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // 1. Validate inputs
        // 2. Convert i16 -> f32, normalize
        // 3. Append to window buffer
        // 4. Extract FBank frame (25ms window, but we only advance 10ms)
        // 5. Apply CMVN
        // 6. Run ONNX: feat [1,1,80] + caches_in [8,1,128,19]
        // 7. Store caches_out, return probability
    }

    fn reset(&mut self) {
        self.caches.fill(0.0);
        self.window_buffer.fill(0.0);
    }
}
```

Constructor pattern matching existing backends:
- `FireRedVad::new()` — uses embedded model + cmvn
- `FireRedVad::from_file(model_path, cmvn_path)` — custom model
- `FireRedVad::from_memory(model_bytes, cmvn_bytes)` — from bytes

### Step 6: Wire Up Module

**File: `crates/wavekat-vad/src/backends/mod.rs`**

```rust
#[cfg(feature = "firered")]
pub mod firered;
```

**File: `crates/wavekat-vad/src/lib.rs`**

Update module docs table to include FireRedVAD.

### Step 7: Tests

Unit tests in `firered/mod.rs` (following Silero's test pattern):
- `create_succeeds` — model loads without error
- `process_silence` — low probability for silence
- `process_wrong_sample_rate` — returns `InvalidSampleRate`
- `process_invalid_frame_size` — returns `InvalidFrameSize`
- `process_returns_continuous_probability` — output in [0.0, 1.0]
- `reset_clears_state` — reset + silence gives low probability
- `state_persists_between_calls` — multiple calls work
- `from_memory_with_embedded_model` — embedded model works
- `from_memory_invalid_bytes` — bad model fails gracefully
- `from_file_nonexistent` — missing file fails gracefully

FBank-specific tests in `firered/fbank.rs`:
- Known FBank output for a simple signal (compare with Python `kaldi_native_fbank`)
- Edge cases: all-zero input, DC signal

CMVN tests in `firered/cmvn.rs`:
- Parse embedded `cmvn.ark` successfully
- Verify dimensions (80)

Integration test with real audio from `testdata/`:
- Process a speech WAV file, verify probability > threshold
- Process a silence WAV file, verify probability < threshold

### Step 8: Update Documentation

- Update `lib.rs` doc table to include FireRedVAD
- Update `backends/mod.rs` doc table
- Update `README.md` feature table

### Step 9: Wire into vad-lab

**File: `tools/vad-lab/backend/Cargo.toml`**

Add `firered` to the wavekat-vad features list:

```toml
wavekat-vad = { path = "../../../crates/wavekat-vad", features = ["webrtc", "silero", "denoise", "ten-vad", "firered", "serde"] }
```

**File: `tools/vad-lab/backend/src/pipeline.rs`**

1. Add FireRedVAD to `create_detector()` match arm:

```rust
"firered-vad" => {
    use wavekat_vad::backends::firered::FireRedVad;
    let vad = FireRedVad::new()
        .map_err(|e| format!("failed to create FireRedVAD: {e}"))?;
    Ok(Box::new(vad))
}
```

2. Add resampling rule in `backend_required_rate()` (FireRedVAD only supports 16kHz):

```rust
"firered-vad" if input_rate != 16000 => Some(16000),
```

3. Register in `available_backends()` with a threshold parameter:

```rust
backends.insert("firered-vad".to_string(), vec![threshold_param.clone()]);
```

## File Summary

### New Files
| File | Purpose |
|------|---------|
| `crates/wavekat-vad/src/backends/firered/mod.rs` | Backend struct, `VoiceActivityDetector` impl, tests |
| `crates/wavekat-vad/src/backends/firered/fbank.rs` | 80-dim FBank feature extraction |
| `crates/wavekat-vad/src/backends/firered/cmvn.rs` | Kaldi CMVN parser |

### Modified Files
| File | Change |
|------|--------|
| `crates/wavekat-vad/Cargo.toml` | Add `firered` feature flag |
| `crates/wavekat-vad/build.rs` | Add model + cmvn download |
| `crates/wavekat-vad/src/backends/mod.rs` | Add `firered` module |
| `crates/wavekat-vad/src/lib.rs` | Update doc comments |
| `tools/vad-lab/backend/Cargo.toml` | Add `firered` feature |
| `tools/vad-lab/backend/src/pipeline.rs` | Add FireRedVAD to `create_detector()`, `backend_required_rate()`, `available_backends()` |

## Python–Rust Parity Validation

Our Rust preprocessing (FBank + CMVN) **must** produce numerically comparable results to the Python `kaldi_native_fbank` + `kaldiio` pipeline. If features diverge, the ONNX model will produce garbage. This is the highest-risk part of the implementation.

### Validation Strategy

#### Step A: Generate Reference Data from Python

Write a Python script (`scripts/firered_reference.py`) that:

1. Loads a test WAV file from `testdata/speech/`
2. Runs the full FireRedVAD Python pipeline step by step, dumping intermediates:
   - Raw i16 samples → `ref_samples.json`
   - FBank features (pre-CMVN) → `ref_fbank.json` (shape `[T, 80]`)
   - CMVN-normalized features → `ref_features.json` (shape `[T, 80]`)
   - CMVN mean/variance vectors → `ref_cmvn.json` (shape `[2, 80]`)
   - Per-frame speech probabilities → `ref_probs.json` (shape `[T]`)
3. Save all reference data to `testdata/firered_reference/`

This script uses the official `fireredvad` Python package to ensure ground truth.

#### Step B: Rust Comparison Tests

For each preprocessing stage, load the reference JSON and compare against our Rust output:

| Stage | Tolerance | What it catches |
|-------|-----------|-----------------|
| CMVN parsing | Exact match (f32 epsilon) | Kaldi format parsing bugs |
| FBank single frame | max abs error < 1e-4 | Window function, FFT, mel scale, log floor differences |
| FBank full file | max abs error < 1e-3 | Accumulated drift from overlap buffering |
| CMVN-normalized features | max abs error < 1e-3 | Mean/variance application order |
| Final probabilities | max abs error < 0.02 | End-to-end parity |

#### Step C: Key Details to Match Exactly

These are the specific numerical choices in `kaldi_native_fbank` that we must replicate:

1. **Window function**: Povey window (Hann-like but `pow(0.85)` modified) vs standard Hann — check which one FireRedVAD uses in its config
2. **Mel scale**: HTK mel scale (`2595 * log10(1 + f/700)`) vs Slaney — must match
3. **Energy floor**: `log(max(energy, 1e-10))` or similar — the floor value matters
4. **Pre-emphasis**: Whether FireRedVAD applies pre-emphasis before FBank (check config `--dither` and `--preemphasis-coefficient`)
5. **FFT normalization**: Some implementations divide by N, others don't
6. **First/last frame padding**: How partial frames at start/end are handled
7. **CMVN application**: Global (from `cmvn.ark`) vs per-utterance — FireRedVAD uses global

### Alternative: `kaldi-native-fbank` Rust Bindings

If achieving numerical parity in pure Rust proves too difficult, we can fall back to **FFI bindings to `kaldi-native-fbank`** (it's a C++ library with a clean C API). This guarantees bit-exact feature extraction at the cost of a C++ build dependency. Evaluate this if Step B tolerances are not met after the pure Rust attempt.

## Other Risks

1. **Window buffering**: The 25ms FBank window with 10ms hop means we need to buffer 15ms of overlap between calls. This is similar to TEN-VAD's approach. **Mitigation**: Follow the established pattern from `ten_vad.rs`.

2. **Model download size**: The streaming ONNX model is small (~2.2 MB), comparable to Silero. No concern here.

3. **`ndarray` dimension**: FireRedVAD cache is 4-dimensional `[8,1,128,19]`. We need `Array4` from ndarray (existing backends only use up to `Array3`). This is supported by ndarray, just haven't used it yet.
