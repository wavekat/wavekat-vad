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

### Step 1: Add Feature Flag and Dependencies ✅

**File: `crates/wavekat-vad/Cargo.toml`**

```toml
[features]
firered = ["dep:ort", "dep:ndarray", "dep:realfft", "dep:ureq"]

# realfft is already a dependency (used by ten-vad for FFT)
# ndarray is already a dependency (used by silero and ten-vad)
```

No new dependencies needed — `ort`, `ndarray`, and `realfft` are already in the workspace for TEN-VAD.

### Step 2: Download ONNX Model + CMVN in build.rs ✅

**File: `crates/wavekat-vad/build.rs`**

Add `setup_firered_model()` following the existing pattern:

- Download `fireredvad_stream_vad_with_cache.onnx` from the GitHub repo
- Download `cmvn.ark` from the GitHub repo
- Support `FIRERED_MODEL_PATH` and `FIRERED_CMVN_PATH` env vars for offline builds
- Write both files to `OUT_DIR`

Source URLs:
- Model: `https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx`
- CMVN: `https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/cmvn.ark`

### Step 3: Implement CMVN Parser ✅

**File: `crates/wavekat-vad/src/backends/firered/cmvn.rs`**

Parse the Kaldi-format `cmvn.ark` file embedded at compile time:
- Format: **Kaldi binary matrix** (`BDM` header = Binary Double Matrix) — not text format
- Row 0: accumulated sums per dimension + count in last column
- Row 1: accumulated sums of squares per dimension
- Computes: `mean[d] = sum[d] / count`, `variance[d] = (sum_sq[d] / count) - mean[d]^2`
- Output: per-dimension mean and inverse-std vectors (80 floats each)
- Apply as `(feature - mean) * inv_std` per dimension
- Variance floor: `1e-20` (matches Python implementation)

Reference: The Python implementation reads this via `kaldiio.load_mat()`. Our Rust parser reads the Kaldi binary format directly (no dependency on kaldiio).

### Step 4: Implement FBank Feature Extraction ✅

**File: `crates/wavekat-vad/src/backends/firered/fbank.rs`**

80-dim log Mel filterbank, matching FireRedVAD's `kaldi_native_fbank` configuration.

The exact `kaldi_native_fbank` defaults used by FireRedVAD (confirmed from Python source):

| Parameter | Value |
|-----------|-------|
| `samp_freq` | 16000 |
| `frame_length_ms` | 25 (400 samples) |
| `frame_shift_ms` | 10 (160 samples) |
| `window_type` | **povey** (Hann^0.85) |
| `preemph_coeff` | 0.97 |
| `remove_dc_offset` | true |
| `dither` | 0 (disabled for inference) |
| `snip_edges` | true |
| `round_to_power_of_two` | true (FFT size = 512) |
| `num_bins` | 80 |
| `low_freq` | 20 Hz |
| `high_freq` | 0 (= Nyquist = 8000 Hz) |
| `htk_mode` | false |
| `use_energy` | false |
| `use_log_fbank` | true |
| `use_power` | true |
| `energy_floor` | f32::EPSILON (~1.19e-7) |

Processing pipeline per frame:
1. **DC offset removal**: subtract mean of frame
2. **Pre-emphasis**: `x[i] -= 0.97 * x[i-1]` (backwards, Kaldi-style; `x[0] *= 0.03`)
3. **Povey window**: `pow(0.5 - 0.5*cos(2π·n/(N-1)), 0.85)`
4. **FFT**: 512-point real FFT (zero-padded from 400), using `realfft` crate
5. **Power spectrum**: |FFT[k]|²
6. **Mel filterbank**: 80 triangular filters, 20–8000 Hz, **mel-domain interpolation** (not Hz-domain)
7. **Log compression**: `ln(max(energy, ε))`

Key difference from TEN-VAD mel filterbank: the triangular filter weights are computed in the **mel domain** — the weight at FFT bin `i` for filter `m` is `(mel(freq_i) - mel_left) / (mel_center - mel_left)`, not `(freq_i - f_left) / (f_center - f_left)`.

**Important**: Input to FBank is **raw i16 values** passed as f32 (not normalized to [-1,1]). This matches `kaldi_native_fbank` which calls `accept_waveform(sample_rate, wav_np.tolist())` with raw int16 values.

### Step 5: Implement FireRedVAD Backend ✅

**File: `crates/wavekat-vad/src/backends/firered/mod.rs`**

```rust
pub struct FireRedVad {
    session: Session,
    fbank: FbankExtractor,
    cmvn: CmvnStats,                  // holds means + inv_stds
    caches: Array4<f32>,               // [8, 1, 128, 19]
    sample_buffer: Vec<f32>,           // accumulates samples for frame building
    frame_count: usize,
}
```

The sample buffering handles the first-frame startup: the FBank needs 400 samples (25ms) for the first frame but `process()` receives 160 samples (10ms) at a time. The first 2 calls return `0.0`, and the 3rd call produces the first real probability.

Constructor pattern matching existing backends:
- `FireRedVad::new()` — uses embedded model + cmvn
- `FireRedVad::from_file(model_path, cmvn_path)` — custom model
- `FireRedVad::from_memory(model_bytes, cmvn_bytes)` — from bytes

### Step 6: Wire Up Module ✅ (partial)

**File: `crates/wavekat-vad/src/backends/mod.rs`** ✅

```rust
#[cfg(feature = "firered")]
pub mod firered;
```

Also updated the `onnx` module gate to include `firered`.

**File: `crates/wavekat-vad/src/lib.rs`**

TODO: Update module docs table to include FireRedVAD.

### Step 7: Tests ✅

**18 tests total** across all three modules, all passing.

Unit tests in `firered/mod.rs` (11 tests):
- ✅ `create_succeeds` — model loads without error
- ✅ `process_silence` — low probability for silence
- ✅ `process_wrong_sample_rate` — returns `InvalidSampleRate`
- ✅ `process_wrong_frame_size` — returns `InvalidFrameSize`
- ✅ `capabilities` — correct sample rate, frame size, duration
- ✅ `reset_works` — reset + process works
- ✅ `multiple_frames` — 10 sequential frames work
- ✅ `from_memory_with_embedded_model` — embedded model works
- ✅ `from_memory_invalid_bytes` — bad model fails gracefully
- ✅ `from_file_nonexistent` — missing file fails gracefully
- ✅ `probabilities_match_python_reference` — **end-to-end parity test** (98 frames, max diff 0.000012)

FBank-specific tests in `firered/fbank.rs` (3 tests):
- ✅ `povey_window_shape` — endpoints zero, symmetric, midpoint > 0.9
- ✅ `mel_filterbank_structure` — 80 filters, all non-empty, ordered
- ✅ `fbank_matches_python_reference` — **98 frames × 80 bins compared** (max diff 0.00068)

CMVN tests in `firered/cmvn.rs` (4 tests):
- ✅ `parse_cmvn_dimensions` — 80-dim
- ✅ `parse_cmvn_values_match_python` — first 5 means/inv_stds match within 1e-4
- ✅ `normalize_applies_correctly` — formula verified
- ✅ `parse_invalid_data` — empty/truncated data errors

### Step 8: Update Documentation

- [ ] Update `lib.rs` doc table to include FireRedVAD
- [ ] Update `backends/mod.rs` doc table to include FireRedVAD
- [ ] Update `README.md` feature table
- [ ] Update `lib.rs` feature flags table
- [ ] Add `FIRERED_MODEL_PATH` / `FIRERED_CMVN_PATH` to env var docs in `lib.rs`

### Step 9: Wire into vad-lab

- [ ] **File: `tools/vad-lab/backend/Cargo.toml`** — Add `firered` to the wavekat-vad features list
- [ ] **File: `tools/vad-lab/backend/src/pipeline.rs`** — Add FireRedVAD to `create_detector()`, `backend_required_rate()`, `available_backends()`

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

## Python–Rust Parity Validation ✅

Our Rust preprocessing (FBank + CMVN) **must** produce numerically comparable results to the Python `kaldi_native_fbank` + `kaldiio` pipeline. If features diverge, the ONNX model will produce garbage. This was the highest-risk part of the implementation.

### Validation Strategy

#### Step A: Generate Reference Data from Python ✅

Python script `scripts/firered/reference.py` generates reference data using `kaldi_native_fbank` and `kaldiio` (installed in `scripts/.venv`). Uses a deterministic 1-second sine wave test signal (200+800+2000+5000 Hz mix) to produce 98 FBank frames.

Reference data saved to `testdata/firered_reference/`:
- `ref_samples.json` — 16000 raw i16 samples
- `ref_fbank.json` — FBank features pre-CMVN [98, 80]
- `ref_features.json` — CMVN-normalized features [98, 80]
- `ref_cmvn.json` — CMVN mean/inv_std vectors [80] each
- `ref_probs.json` — per-frame ONNX probabilities [98]

#### Step B: Rust Comparison Tests ✅ — All Passing

| Stage | Tolerance | Actual Max Diff | Result |
|-------|-----------|-----------------|--------|
| CMVN parsing | < 1e-4 | < 1e-4 | ✅ |
| FBank (98 frames × 80 bins) | < 1e-3 | **0.00068** | ✅ |
| Final probabilities (98 frames) | < 0.02 | **0.000012** | ✅ |

The pure Rust implementation achieves excellent numerical parity — **probability error is 1600× below tolerance**.

#### Step C: Key Details Resolved

These are the specific numerical choices in `kaldi_native_fbank` that we replicated:

1. **Window function**: **Povey window** — `pow(0.5 - 0.5*cos(2π·n/(N-1)), 0.85)` (confirmed via Python introspection)
2. **Mel scale**: **Kaldi mel scale** (non-HTK mode) — `1127 * ln(1 + f/700)`, numerically equivalent to HTK's `2595 * log10(1 + f/700)` but using natural log
3. **Energy floor**: `f32::EPSILON` (~1.19e-7) — matches `std::numeric_limits<float>::epsilon()` in kaldi-native-fbank C++
4. **Pre-emphasis**: 0.97 coefficient, applied **within each frame** (backwards loop), `x[0] *= (1 - 0.97)`. Applied after DC offset removal, before windowing.
5. **FFT normalization**: `realfft` crate does NOT divide by N (matches Kaldi)
6. **First/last frame padding**: `snip_edges=true` — no padding, only full frames are extracted
7. **CMVN application**: Global (from `cmvn.ark`), applied as `(feature - mean) * inv_std`

### FFI Fallback: Not Needed

Pure Rust achieved parity well within tolerances. No need for `kaldi-native-fbank` C++ bindings.

## Risks — Resolved

1. **Window buffering** ✅ — The FbankExtractor stores the last 240 samples as overlap. The FireRedVad struct additionally buffers incoming 160-sample chunks via `sample_buffer` until enough samples are accumulated for a full 400-sample window.

2. **Model download size** ✅ — ~2.2 MB, comparable to Silero. No concern.

3. **`ndarray` dimension** ✅ — `Array4<f32>` works correctly for the `[8,1,128,19]` DFSMN cache.

4. **First-frame startup** — New discovery: since `process()` receives 160 samples but the first FBank frame needs 400, the first 2 calls return `0.0` while samples accumulate. The 3rd call (at 480 samples) produces the first real probability. This is acceptable for streaming use.
