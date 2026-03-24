#!/usr/bin/env python3
"""Generate reference data for FireRedVAD Rust implementation validation.

Dumps intermediate values at each preprocessing stage so the Rust
implementation can be validated step by step.

Output files (in testdata/firered_reference/):
  - ref_samples.json : Raw i16 samples for the test signal
  - ref_cmvn.json    : CMVN mean and inv_std vectors (80-dim each)
  - ref_fbank.json   : FBank features pre-CMVN [T, 80]
  - ref_probs.json   : Per-frame speech probabilities [T]
"""

import json
import math
import os
import sys

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REF_DIR = os.path.join(PROJECT_DIR, "testdata", "firered_reference")
CMVN_PATH = "/tmp/firered_cmvn.ark"
MODEL_PATH = "/tmp/fireredvad_stream_vad_with_cache.onnx"


def generate_test_signal():
    """Generate a deterministic test signal: 1 second of mixed sine waves at 16kHz.

    Uses multiple frequencies to produce non-trivial FBank features.
    Returns int16 samples.
    """
    sample_rate = 16000
    duration = 1.0  # 1 second = 100 frames of 10ms
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # Mix of frequencies: 200Hz, 800Hz, 2000Hz, 5000Hz
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.25 * np.sin(2 * np.pi * 800 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.15 * np.sin(2 * np.pi * 5000 * t)
    )

    # Scale to int16 range (use ~70% of range to avoid clipping)
    signal = signal / np.max(np.abs(signal)) * 0.7 * 32767
    samples_i16 = signal.astype(np.int16)
    return samples_i16


def parse_cmvn(cmvn_path):
    """Parse CMVN ark file and return (dim, means, inv_stds)."""
    stats = kaldiio.load_mat(cmvn_path)
    assert stats.shape[0] == 2, f"Expected 2 rows, got {stats.shape[0]}"
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    assert count >= 1

    floor = 1e-20
    means = []
    inverse_std_variances = []
    for d in range(dim):
        mean = stats[0, d] / count
        means.append(float(mean))
        variance = (stats[1, d] / count) - mean * mean
        if variance < floor:
            variance = floor
        istd = 1.0 / math.sqrt(variance)
        inverse_std_variances.append(float(istd))

    return dim, np.array(means), np.array(inverse_std_variances)


def extract_fbank(samples_i16, sample_rate=16000):
    """Extract 80-dim FBank features using kaldi_native_fbank.

    Matches FireRedVAD's exact configuration.
    """
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, samples_i16.tolist())

    frames = []
    for i in range(fbank.num_frames_ready):
        frames.append(fbank.get_frame(i))

    if len(frames) == 0:
        return np.zeros((0, 80))
    return np.vstack(frames)


def apply_cmvn(features, means, inv_stds):
    """Apply CMVN normalization."""
    return (features - means) * inv_stds


def run_onnx_streaming(features, model_path):
    """Run streaming ONNX inference frame by frame, returning per-frame probs."""
    session = ort.InferenceSession(model_path)

    # Print input/output info
    print("ONNX Model Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
    print("ONNX Model Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")

    # Initialize caches: [8, 1, 128, 19]
    caches = np.zeros((8, 1, 128, 19), dtype=np.float32)
    probs = []

    for t in range(features.shape[0]):
        feat_frame = features[t:t+1, :].reshape(1, 1, 80).astype(np.float32)

        outputs = session.run(
            None,
            {
                "feat": feat_frame,
                "caches_in": caches,
            },
        )

        prob = outputs[0]  # probs [1, 1, 1]
        caches = outputs[1]  # caches_out [8, 1, 128, 19]

        probs.append(float(prob.flatten()[0]))

    return probs


def main():
    os.makedirs(REF_DIR, exist_ok=True)

    print("=== Step 0: Generate test signal ===")
    samples = generate_test_signal()
    print(f"  Samples: {len(samples)} ({len(samples)/16000:.3f}s)")
    print(f"  Range: [{samples.min()}, {samples.max()}]")
    print(f"  First 10: {samples[:10].tolist()}")

    with open(os.path.join(REF_DIR, "ref_samples.json"), "w") as f:
        json.dump({"samples": samples.tolist(), "sample_rate": 16000}, f)

    print("\n=== Step 1: Parse CMVN ===")
    dim, means, inv_stds = parse_cmvn(CMVN_PATH)
    print(f"  Dimension: {dim}")
    print(f"  Means (first 5): {means[:5].tolist()}")
    print(f"  InvStds (first 5): {inv_stds[:5].tolist()}")

    with open(os.path.join(REF_DIR, "ref_cmvn.json"), "w") as f:
        json.dump({"dim": dim, "means": means.tolist(), "inv_stds": inv_stds.tolist()}, f)

    print("\n=== Step 2: Extract FBank features ===")
    fbank = extract_fbank(samples)
    print(f"  Shape: {fbank.shape}")
    print(f"  Frame 0 (first 5 bins): {fbank[0, :5].tolist()}")
    print(f"  Frame 0 (last 5 bins): {fbank[0, -5:].tolist()}")
    print(f"  Min: {fbank.min():.6f}, Max: {fbank.max():.6f}")

    with open(os.path.join(REF_DIR, "ref_fbank.json"), "w") as f:
        json.dump({"shape": list(fbank.shape), "data": fbank.tolist()}, f)

    print("\n=== Step 3: Apply CMVN ===")
    features = apply_cmvn(fbank, means, inv_stds)
    print(f"  Shape: {features.shape}")
    print(f"  Frame 0 (first 5): {features[0, :5].tolist()}")
    print(f"  Min: {features.min():.6f}, Max: {features.max():.6f}")

    print("\n=== Step 4: Run ONNX streaming inference ===")
    probs = run_onnx_streaming(features, MODEL_PATH)
    print(f"  Num frames: {len(probs)}")
    print(f"  First 10 probs: {probs[:10]}")
    print(f"  Min: {min(probs):.6f}, Max: {max(probs):.6f}")

    with open(os.path.join(REF_DIR, "ref_probs.json"), "w") as f:
        json.dump({"probs": probs}, f)

    print("\n=== Done! Reference data saved to", REF_DIR, "===")


if __name__ == "__main__":
    main()
