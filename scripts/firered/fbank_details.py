#!/usr/bin/env python3
"""Dump detailed FBank intermediates for step-by-step Rust validation.

Manually reimplements the kaldi_native_fbank pipeline to dump every
intermediate buffer. Then verifies our manual pipeline matches the
library's output.
"""

import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REF_DIR = os.path.join(PROJECT_DIR, "testdata", "firered_reference")

# Kaldi FBank parameters (matching FireRedVAD config)
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
FRAME_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)  # 400
FRAME_SHIFT = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)    # 160
FFT_SIZE = 512  # next power of 2 >= 400
N_BINS = FFT_SIZE // 2 + 1  # 257
N_MEL = 80
LOW_FREQ = 20.0
HIGH_FREQ = 8000.0  # 0 means Nyquist
PREEMPH = 0.97


def mel_scale(freq):
    """Kaldi mel scale (non-HTK): 1127 * ln(1 + f/700)."""
    return 1127.0 * math.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel):
    """Inverse Kaldi mel scale."""
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def povey_window(length):
    """Povey window: pow(0.5 - 0.5*cos(2*pi*n/(N-1)), 0.85)."""
    window = np.zeros(length)
    for i in range(length):
        window[i] = (0.5 - 0.5 * math.cos(2.0 * math.pi * i / (length - 1))) ** 0.85
    return window


def compute_mel_filterbank():
    """Compute Kaldi-style mel filterbank weights.

    Returns (n_mel, n_bins) matrix of filterbank weights.
    """
    low_mel = mel_scale(LOW_FREQ)
    high_mel = mel_scale(HIGH_FREQ)

    # n_mel + 2 equally spaced points in mel domain
    mel_points = np.linspace(low_mel, high_mel, N_MEL + 2)
    hz_points = np.array([inverse_mel_scale(m) for m in mel_points])

    # Convert Hz to FFT bin indices (real-valued, for interpolation)
    bin_freqs = np.array([i * SAMPLE_RATE / FFT_SIZE for i in range(N_BINS)])

    weights = np.zeros((N_MEL, N_BINS))
    for m in range(N_MEL):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        for k in range(N_BINS):
            freq = bin_freqs[k]
            if freq < f_left or freq > f_right:
                continue
            if freq <= f_center:
                if f_center > f_left:
                    weights[m, k] = (freq - f_left) / (f_center - f_left)
            else:
                if f_right > f_center:
                    weights[m, k] = (f_right - freq) / (f_right - f_center)

    return weights


def process_frame(samples_f32, window):
    """Process a single frame through the Kaldi FBank pipeline.

    Args:
        samples_f32: float64 samples for this frame (400 samples)
        window: Povey window coefficients

    Returns dict with all intermediates.
    """
    frame = samples_f32.copy()

    # 1. Remove DC offset
    dc_offset = np.mean(frame)
    frame -= dc_offset

    # 2. Pre-emphasis (in-place, backwards)
    # Kaldi does: for i in N-1..1: x[i] -= 0.97*x[i-1]; x[0] -= 0.97*x[0]
    preemph_frame = frame.copy()
    for i in range(len(preemph_frame) - 1, 0, -1):
        preemph_frame[i] -= PREEMPH * preemph_frame[i - 1]
    preemph_frame[0] -= PREEMPH * preemph_frame[0]

    # 3. Apply window
    windowed = preemph_frame * window

    # 4. FFT (zero-pad to FFT_SIZE)
    padded = np.zeros(FFT_SIZE)
    padded[:FRAME_LENGTH] = windowed
    spectrum = np.fft.rfft(padded)

    # 5. Power spectrum
    power = np.abs(spectrum) ** 2

    # 6. Mel filterbank
    mel_weights = compute_mel_filterbank()
    mel_energies = mel_weights @ power

    # 7. Log (with floor)
    epsilon = np.finfo(np.float32).eps  # ~1.19e-7
    log_mel = np.log(np.maximum(mel_energies, epsilon))

    return {
        "dc_offset": float(dc_offset),
        "after_dc_removal": frame[:10].tolist(),
        "after_preemph": preemph_frame[:10].tolist(),
        "after_window": windowed[:10].tolist(),
        "power_spectrum_first10": power[:10].tolist(),
        "power_spectrum_last5": power[-5:].tolist(),
        "mel_energies": mel_energies.tolist(),
        "log_mel": log_mel.tolist(),
    }


def main():
    # Load reference samples
    with open(os.path.join(REF_DIR, "ref_samples.json")) as f:
        data = json.load(f)
    samples_i16 = np.array(data["samples"], dtype=np.int16)

    # Load reference FBank for comparison
    with open(os.path.join(REF_DIR, "ref_fbank.json")) as f:
        ref_data = json.load(f)
    ref_fbank = np.array(ref_data["data"])

    print("=== Povey Window (first 10, last 5) ===")
    window = povey_window(FRAME_LENGTH)
    print(f"  First 10: {window[:10].tolist()}")
    print(f"  Last 5: {window[-5:].tolist()}")
    print(f"  Window sum: {window.sum():.6f}")

    print("\n=== Mel Filterbank ===")
    mel_weights = compute_mel_filterbank()
    print(f"  Shape: {mel_weights.shape}")
    # Find first non-zero bin for each filter
    for m in [0, 1, 39, 79]:
        nonzero = np.nonzero(mel_weights[m])[0]
        if len(nonzero) > 0:
            print(f"  Filter {m}: bins {nonzero[0]}-{nonzero[-1]}, "
                  f"peak={mel_weights[m].max():.6f}")

    # Process first 5 frames
    print("\n=== Frame-by-Frame Processing ===")
    details = []
    for frame_idx in range(min(5, ref_fbank.shape[0])):
        start = frame_idx * FRAME_SHIFT
        end = start + FRAME_LENGTH
        # Kaldi passes raw int16 values as float
        frame_samples = samples_i16[start:end].astype(np.float64)

        result = process_frame(frame_samples, window)

        # Compare with kaldi_native_fbank reference
        max_diff = np.max(np.abs(np.array(result["log_mel"]) - ref_fbank[frame_idx]))
        print(f"\n  Frame {frame_idx}:")
        print(f"    DC offset: {result['dc_offset']:.6f}")
        print(f"    After preemph (first 5): {result['after_preemph'][:5]}")
        print(f"    Log mel (first 5): {result['log_mel'][:5]}")
        print(f"    Ref fbank (first 5): {ref_fbank[frame_idx, :5].tolist()}")
        print(f"    Max diff vs reference: {max_diff:.8f}")

        details.append(result)

    # Save all intermediates
    output = {
        "povey_window": window.tolist(),
        "mel_filterbank_shape": list(mel_weights.shape),
        "frame_details": details,
    }

    with open(os.path.join(REF_DIR, "ref_fbank_intermediates.json"), "w") as f:
        json.dump(output, f)

    print(f"\n=== Saved intermediates to {REF_DIR}/ref_fbank_intermediates.json ===")


if __name__ == "__main__":
    main()
