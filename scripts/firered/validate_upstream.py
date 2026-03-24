#!/usr/bin/env python3
"""Validate our reference pipeline against FireRedVAD's official pip package.

Runs the same audio through both pipelines and compares per-frame probabilities
end-to-end. If they match, our FBank config, CMVN parsing, and ONNX inference
are correct by definition — no need to inspect upstream internals.

Requirements:
    pip install fireredvad torch soundfile numpy kaldi_native_fbank kaldiio onnxruntime

Usage:
    # With synthetic test signal (default)
    python scripts/firered/validate_upstream.py

    # With a real WAV file
    python scripts/firered/validate_upstream.py --wav path/to/audio.wav
"""

import json
import math
import os
import tempfile

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ONNX_MODEL_PATH = "/tmp/fireredvad_stream_vad_with_cache.onnx"
CMVN_PATH = "/tmp/firered_cmvn.ark"


# ---------------------------------------------------------------------------
# Our pipeline (same logic as reference.py, but on arbitrary audio)
# ---------------------------------------------------------------------------


def parse_cmvn(cmvn_path):
    """Parse CMVN ark file and return (means, inv_stds)."""
    stats = kaldiio.load_mat(cmvn_path)
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    floor = 1e-20
    means = []
    inv_stds = []
    for d in range(dim):
        mean = stats[0, d] / count
        means.append(float(mean))
        variance = (stats[1, d] / count) - mean * mean
        if variance < floor:
            variance = floor
        inv_stds.append(float(1.0 / math.sqrt(variance)))
    return np.array(means), np.array(inv_stds)


def extract_fbank(samples_i16, sample_rate=16000):
    """Extract 80-dim FBank features using kaldi_native_fbank."""
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


def run_our_pipeline(samples_i16):
    """Run our ONNX pipeline on raw i16 samples. Returns per-frame probs."""
    means, inv_stds = parse_cmvn(CMVN_PATH)
    fbank = extract_fbank(samples_i16)
    features = (fbank - means) * inv_stds

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    caches = np.zeros((8, 1, 128, 19), dtype=np.float32)
    probs = []

    for t in range(features.shape[0]):
        feat_frame = features[t : t + 1, :].reshape(1, 1, 80).astype(np.float32)
        outputs = session.run(
            None, {"feat": feat_frame, "caches_in": caches}
        )
        probs.append(float(outputs[0].flatten()[0]))
        caches = outputs[1]

    return probs


# ---------------------------------------------------------------------------
# Upstream pipeline (fireredvad pip package)
# ---------------------------------------------------------------------------


def run_upstream(wav_path, model_dir):
    """Run FireRedVAD's official pip package on the WAV file."""
    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    config = FireRedStreamVadConfig(use_gpu=False)
    vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
    frame_results, _ = vad.detect_full(wav_path)
    return [r.raw_prob for r in frame_results]


# ---------------------------------------------------------------------------
# Test signal generation
# ---------------------------------------------------------------------------


def generate_test_signal():
    """Same deterministic test signal as reference.py."""
    sample_rate = 16000
    duration = 1.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.25 * np.sin(2 * np.pi * 800 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.15 * np.sin(2 * np.pi * 5000 * t)
    )
    signal = signal / np.max(np.abs(signal)) * 0.7 * 32767
    return signal.astype(np.int16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compare_probs(upstream_probs, our_probs, tolerance=0.01):
    """Compare two probability lists and print results."""
    min_len = min(len(upstream_probs), len(our_probs))
    if len(upstream_probs) != len(our_probs):
        print(
            f"  WARNING: frame count differs: upstream={len(upstream_probs)}, ours={len(our_probs)}"
        )
        print(f"  Comparing first {min_len} frames")

    upstream_arr = np.array(upstream_probs[:min_len])
    our_arr = np.array(our_probs[:min_len])
    diffs = np.abs(upstream_arr - our_arr)

    max_diff = diffs.max()
    mean_diff = diffs.mean()
    max_diff_idx = diffs.argmax()

    print(f"  Max diff:  {max_diff:.8f} (frame {max_diff_idx})")
    print(f"  Mean diff: {mean_diff:.8f}")

    if max_diff < tolerance:
        print(f"\n  PASS: max diff {max_diff:.8f} < tolerance {tolerance}")
    else:
        print(f"\n  FAIL: max diff {max_diff:.8f} >= tolerance {tolerance}")
        worst_indices = np.argsort(diffs)[-5:][::-1]
        print("\n  Worst frames:")
        for idx in worst_indices:
            print(
                f"    frame {idx}: upstream={upstream_arr[idx]:.8f}, "
                f"ours={our_arr[idx]:.8f}, diff={diffs[idx]:.8f}"
            )

    return max_diff < tolerance


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate our ONNX pipeline against FireRedVAD upstream (PyTorch)."
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="Path to a WAV file (16kHz mono). If omitted, uses a synthetic test signal.",
    )
    parser.add_argument(
        "--model-dir",
        default="/tmp/FireRedVAD/Stream-VAD",
        help="Path to Stream-VAD model directory (contains model.pth.tar + cmvn.ark).",
    )
    args = parser.parse_args()

    print("=== Validating our pipeline against FireRedVAD upstream ===\n")

    # Check prerequisites
    model_pth = os.path.join(args.model_dir, "model.pth.tar")
    if not os.path.exists(model_pth):
        print(f"ERROR: {model_pth} not found.")
        print("Download models first:")
        print("  pip install huggingface_hub")
        print(
            '  python -c "from huggingface_hub import snapshot_download; '
            "snapshot_download('FireRedTeam/FireRedVAD', local_dir='/tmp/FireRedVAD')\""
        )
        return

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ERROR: {ONNX_MODEL_PATH} not found.")
        print("The ONNX model is downloaded during `cargo build --features firered`.")
        return

    # Prepare audio
    cleanup_wav = False
    if args.wav:
        wav_path = args.wav
        data, sr = sf.read(wav_path, dtype="int16")
        samples_i16 = data
        duration = len(data) / sr
        print(f"Input: {wav_path} ({duration:.2f}s, {sr} Hz, {len(data)} samples)")
    else:
        samples_i16 = generate_test_signal()
        wav_path = os.path.join(tempfile.gettempdir(), "firered_validate_test.wav")
        sf.write(wav_path, samples_i16, 16000, subtype="PCM_16")
        cleanup_wav = True
        print(f"Input: synthetic test signal (1.00s, 16000 Hz, {len(samples_i16)} samples)")

    # Run upstream (PyTorch)
    print("\n--- Upstream (fireredvad pip, PyTorch) ---")
    try:
        upstream_probs = run_upstream(wav_path, args.model_dir)
    except ImportError:
        print("ERROR: fireredvad not installed. Run: pip install fireredvad torch")
        return
    print(f"  Frames: {len(upstream_probs)}")
    print(f"  First 5: {[f'{p:.6f}' for p in upstream_probs[:5]]}")
    print(f"  Range: [{min(upstream_probs):.6f}, {max(upstream_probs):.6f}]")

    # Run our pipeline (ONNX)
    print("\n--- Our pipeline (kaldi_native_fbank + ONNX) ---")
    our_probs = run_our_pipeline(samples_i16)
    print(f"  Frames: {len(our_probs)}")
    print(f"  First 5: {[f'{p:.6f}' for p in our_probs[:5]]}")
    print(f"  Range: [{min(our_probs):.6f}, {max(our_probs):.6f}]")

    # Compare
    print("\n--- Comparison ---")
    passed = compare_probs(upstream_probs, our_probs)

    if passed:
        print("  Our pipeline matches FireRedVAD upstream end-to-end.")

    if cleanup_wav:
        os.remove(wav_path)


if __name__ == "__main__":
    main()
