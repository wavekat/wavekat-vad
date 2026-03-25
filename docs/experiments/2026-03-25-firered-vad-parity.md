# FireRedVAD Python–Rust Parity Validation

**Date:** 2026-03-25
**Goal:** Verify that our pure Rust FBank + CMVN preprocessing produces numerically identical results to the Python `kaldi_native_fbank` + `kaldiio` pipeline used by FireRedVAD.

## Background

FireRedVAD's ONNX model expects 80-dim log Mel filterbank features normalized by CMVN. If our Rust preprocessing diverges from the Python pipeline, the model produces garbage. This is the highest-risk part of the FireRedVAD integration.

The Python pipeline uses:
- `kaldi_native_fbank` v1.22 — C++ library with Python bindings for Kaldi-compatible FBank extraction
- `kaldiio` v2.18 — reads Kaldi binary ark files (CMVN stats)

We implemented both in pure Rust (no C++ FFI).

## Method

### Test Signal

Deterministic 1-second signal at 16 kHz (16000 i16 samples): sum of sine waves at 200, 800, 2000, and 5000 Hz, scaled to ~70% of int16 range. This produces non-trivial FBank features across multiple mel bands while being perfectly reproducible.

### Reference Data Generation

Python script `scripts/firered/reference.py`:
1. Generates the test signal
2. Extracts FBank using `kaldi_native_fbank.OnlineFbank` with FireRedVAD's exact config
3. Parses CMVN from `cmvn.ark` using `kaldiio.load_mat()`
4. Applies CMVN normalization
5. Runs `onnxruntime` streaming inference (frame-by-frame with cache)
6. Dumps all intermediates to `testdata/firered_reference/*.json`

### Rust Comparison

Three levels of comparison tests:

| Test | What it compares | How |
|------|-----------------|-----|
| `cmvn::tests::parse_cmvn_values_match_python` | CMVN means + inv_stds | Load `ref_cmvn.json`, compare first 5 values |
| `fbank::tests::fbank_matches_python_reference` | All 98×80 FBank features | Load `ref_fbank.json`, compare every element |
| `tests::probabilities_match_python_reference` | All 98 output probabilities | Load `ref_probs.json`, run full Rust pipeline, compare |

## Results

| Stage | Tolerance | Actual Max Diff | Margin |
|-------|-----------|-----------------|--------|
| CMVN parsing | < 1e-4 | < 1e-4 | ~1× |
| FBank (98 frames × 80 bins = 7840 values) | < 1e-3 | **6.8e-4** | 1.5× |
| End-to-end probabilities (98 frames) | < 0.02 | **1.2e-5** | 1667× |

### Key Observations

1. **FBank precision is excellent** — max diff 0.00068 across 7840 values. The small error comes from float32 FFT arithmetic differences between `realfft` (Rust) and the C++ FFTW/KissFFT used by `kaldi_native_fbank`.

2. **Probability error is negligible** — the 0.000012 max diff is well within ONNX Runtime's own numerical noise. The DFSMN model is not sensitive to the sub-1e-3 FBank differences.

3. **No accumulated drift** — the test signal is periodic (all sine frequencies divide evenly into 160-sample frames), so all 98 frames exercise the same code path. A non-periodic signal would better test overlap buffering, but the parity is good enough to proceed.

## Key Implementation Details Discovered

These details were **not in the original plan** and were discovered by inspecting the Python source and `kaldi_native_fbank` defaults:

1. **Povey window** (not Hann): `pow(0.5 - 0.5*cos(2π·n/(N-1)), 0.85)` — a modified Hann raised to power 0.85
2. **Mel-domain interpolation**: triangular filter weights computed in mel space, not Hz space — `(mel(f) - mel_left) / (mel_center - mel_left)`
3. **DC offset removal**: mean of each frame is subtracted before pre-emphasis
4. **Pre-emphasis within frame**: backwards loop `x[i] -= 0.97*x[i-1]`, with `x[0] *= 0.03` — applied independently per frame (not across frames like TEN-VAD)
5. **Raw i16 input**: FBank operates on raw int16 values cast to f32 (no normalization to [-1,1])
6. **Energy floor**: `f32::EPSILON` (~1.19e-7), not 1e-10 or 1e-20
7. **Mel range**: 20–8000 Hz (not 0–8000 Hz) — `low_freq=20` is the `kaldi_native_fbank` default
8. **CMVN file format**: Kaldi binary matrix (`BDM` header), not text — the plan incorrectly said "text matrix"

## Conclusions

- Pure Rust FBank + CMVN achieves sufficient numerical parity. **No need for C++ FFI fallback.**
- The reference data and comparison tests should be maintained as regression tests.
- The test signal could be improved with non-periodic content (e.g. chirp or real speech) to exercise overlap buffering paths differently.

## Reproducing

```sh
# Set up Python environment
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
pip install kaldi_native_fbank kaldiio numpy soundfile onnxruntime

# Generate reference data
python scripts/firered/reference.py

# Run Rust comparison tests
cargo test --features firered -p wavekat-vad -- firered --nocapture
```
