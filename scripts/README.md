# Scripts

Scripts for generating reference data used by Rust integration tests, organized by backend.

## Why These Exist

Some VAD backends (e.g. FireRedVAD) require complex preprocessing (FBank, CMVN) that must produce numerically identical output to the original Python implementation. To validate this, we:

1. Run the **original Python pipeline** to generate reference data (intermediate features, probabilities)
2. Run the **Rust implementation** on the same input
3. Compare at each stage — Rust tests load the reference JSON and assert tolerances

The reference data lives in `testdata/<backend>_reference/` and is checked into git so the Rust tests work without Python installed.

## Setup

```sh
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate

# Core deps (reference data generation)
pip install kaldi_native_fbank kaldiio numpy soundfile onnxruntime

# Additional deps for upstream validation (validate_upstream.py)
pip install fireredvad torch huggingface_hub
```

### External model files

`reference.py` needs the ONNX model and CMVN file. These are automatically downloaded during `cargo build --features firered` to `/tmp/`:

- `/tmp/fireredvad_stream_vad_with_cache.onnx`
- `/tmp/firered_cmvn.ark`

If you haven't built the Rust crate yet, download them manually:

```sh
curl -sSL -o /tmp/fireredvad_stream_vad_with_cache.onnx \
  https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx
curl -sSL -o /tmp/firered_cmvn.ark \
  https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/cmvn.ark
```

`validate_upstream.py` additionally needs the PyTorch model from HuggingFace:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('FireRedTeam/FireRedVAD', local_dir='/tmp/FireRedVAD')"
```

## `firered/`

Reference data generation for the FireRedVAD backend. See `docs/experiments/2026-03-25-firered-vad-parity.md` for the full experiment log.

### `firered/reference.py`

Generates end-to-end reference data for FireRedVAD:

- Synthesizes a deterministic 1-second test signal (mixed sine waves)
- Extracts FBank features using `kaldi_native_fbank` (the same library FireRedVAD uses)
- Parses CMVN stats using `kaldiio`
- Runs ONNX streaming inference frame-by-frame

**Output** (`testdata/firered_reference/`):

| File | Contents |
|------|----------|
| `ref_samples.json` | Raw i16 samples (16000 samples, 1 second) |
| `ref_cmvn.json` | CMVN means and inverse std vectors (80-dim each) |
| `ref_fbank.json` | FBank features pre-CMVN [98, 80] |
| `ref_features.json` | CMVN-normalized features [98, 80] |
| `ref_probs.json` | Per-frame speech probabilities [98] |

**To regenerate:**

```sh
source scripts/.venv/bin/activate
python scripts/firered/reference.py
```

### `firered/validate_upstream.py`

End-to-end validation against FireRedVAD's official pip package. Runs the same audio through both our ONNX pipeline and the upstream `fireredvad` package (PyTorch), then compares per-frame probabilities. This proves our FBank config and CMVN match upstream without needing to inspect their source code.

```sh
# With synthetic test signal
python scripts/firered/validate_upstream.py

# With a real WAV file (more convincing)
python scripts/firered/validate_upstream.py --wav target/testset/testset-audio-01.wav
```

Small numerical differences (< 1%) are expected due to PyTorch vs ONNX inference.

### `firered/fbank_details.py`

Dumps detailed FBank intermediates (Povey window, mel filterbank, per-frame DC offset / pre-emphasis / power spectrum) for debugging. Useful when the FBank output drifts from the reference.

```sh
python scripts/firered/fbank_details.py
```

## When to Regenerate

Re-run the reference scripts if:

- The test signal generation changes
- You want to validate against a different audio file
- The upstream `kaldi_native_fbank` or FireRedVAD model changes
- You're debugging a numerical parity issue

The Rust tests will fail if the reference data is out of sync.
