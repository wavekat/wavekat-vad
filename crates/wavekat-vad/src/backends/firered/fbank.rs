//! 80-dim log Mel filterbank (FBank) feature extractor.
//!
//! Matches the `kaldi_native_fbank` configuration used by FireRedVAD:
//!
//! - Sample rate: 16 kHz
//! - Frame length: 25 ms (400 samples)
//! - Frame shift: 10 ms (160 samples)
//! - FFT size: 512 (next power of 2 >= 400)
//! - 80 Mel filter banks, 20–8000 Hz (Kaldi mel scale)
//! - Povey window
//! - Pre-emphasis: 0.97
//! - DC offset removal
//! - No dither
//! - `snip_edges = true`

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Sample rate (16 kHz only).
const SAMPLE_RATE: u32 = 16000;

/// Frame length in samples (25 ms at 16 kHz).
const FRAME_LENGTH: usize = 400;

/// Frame shift in samples (10 ms at 16 kHz).
pub(crate) const FRAME_SHIFT: usize = 160;

/// FFT size (next power of 2 >= FRAME_LENGTH).
const FFT_SIZE: usize = 512;

/// Number of FFT bins (FFT_SIZE / 2 + 1).
const N_BINS: usize = FFT_SIZE / 2 + 1; // 257

/// Number of Mel filter banks.
const N_MEL: usize = 80;

/// Low frequency for Mel filterbank (Hz).
const LOW_FREQ: f32 = 20.0;

/// High frequency for Mel filterbank (Hz). 0 means Nyquist.
const HIGH_FREQ: f32 = 8000.0; // sample_rate / 2

/// Pre-emphasis coefficient.
const PREEMPH_COEFF: f32 = 0.97;

/// Kaldi mel scale: 1127 * ln(1 + f/700).
#[inline]
fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Inverse Kaldi mel scale.
#[allow(dead_code)]
#[inline]
fn inverse_mel_scale(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Sparse Mel filter (only stores non-zero bins).
struct MelFilter {
    /// First FFT bin with a non-zero coefficient.
    start_bin: usize,
    /// Non-zero filter coefficients.
    weights: Vec<f32>,
}

/// 80-dim FBank feature extractor matching kaldi_native_fbank.
pub(crate) struct FbankExtractor {
    /// FFT plan.
    fft: Arc<dyn RealToComplex<f32>>,
    /// Povey window coefficients (FRAME_LENGTH).
    window: Vec<f32>,
    /// Mel filterbank (sparse).
    mel_filters: Vec<MelFilter>,
    /// Overlap buffer: last (FRAME_LENGTH - FRAME_SHIFT) = 240 samples
    /// from the previous frame, used for windowing.
    overlap_buffer: Vec<f32>,
    /// Whether this is the first frame (no overlap yet).
    first_frame: bool,
    /// Reusable FFT input buffer (FFT_SIZE).
    fft_input: Vec<f32>,
    /// Reusable FFT scratch buffer.
    fft_scratch: Vec<realfft::num_complex::Complex<f32>>,
    /// Reusable FFT output buffer (N_BINS complex values).
    fft_output: Vec<realfft::num_complex::Complex<f32>>,
    /// Reusable power spectrum buffer (N_BINS).
    power_spectrum: Vec<f32>,
    /// Total frames processed.
    frame_count: usize,
}

impl FbankExtractor {
    /// Create a new FBank extractor.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_scratch_len();

        let window = Self::povey_window();
        let mel_filters = Self::compute_mel_filterbank();

        Self {
            fft,
            window,
            mel_filters,
            overlap_buffer: vec![0.0; FRAME_LENGTH - FRAME_SHIFT], // 240 samples
            first_frame: true,
            fft_input: vec![0.0; FFT_SIZE],
            fft_scratch: vec![realfft::num_complex::Complex::new(0.0, 0.0); scratch_len],
            fft_output: vec![realfft::num_complex::Complex::new(0.0, 0.0); N_BINS],
            power_spectrum: vec![0.0; N_BINS],
            frame_count: 0,
        }
    }

    /// Generate Povey window: pow(0.5 - 0.5*cos(2*PI*n/(N-1)), 0.85).
    fn povey_window() -> Vec<f32> {
        (0..FRAME_LENGTH)
            .map(|i| {
                let hann = 0.5
                    - 0.5
                        * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos();
                hann.powf(0.85) as f32
            })
            .collect()
    }

    /// Compute Kaldi-style Mel filterbank using mel-domain interpolation.
    ///
    /// For each filter m, the center frequencies (left, center, right) are
    /// equally spaced in the mel domain. Filter weights for each FFT bin
    /// are computed by mapping the bin frequency to mel scale and
    /// interpolating within the triangle.
    fn compute_mel_filterbank() -> Vec<MelFilter> {
        let mel_low = mel_scale(LOW_FREQ);
        let mel_high = mel_scale(HIGH_FREQ);
        let mel_delta = (mel_high - mel_low) / (N_MEL as f32 + 1.0);
        let fft_bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;

        let mut filters = Vec::with_capacity(N_MEL);

        for m in 0..N_MEL {
            let left_mel = mel_low + m as f32 * mel_delta;
            let center_mel = mel_low + (m + 1) as f32 * mel_delta;
            let right_mel = mel_low + (m + 2) as f32 * mel_delta;

            let mut start_bin = N_BINS; // will be set to first non-zero bin
            let mut weights = Vec::new();

            for i in 0..N_BINS {
                let freq = fft_bin_width * i as f32;
                let mel = mel_scale(freq);

                // Strict inequality: mel > left_mel && mel < right_mel
                if mel > left_mel && mel < right_mel {
                    let weight = if mel <= center_mel {
                        (mel - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - mel) / (right_mel - center_mel)
                    };

                    if start_bin == N_BINS {
                        start_bin = i;
                    }
                    // Fill any gap between last weight and this bin
                    let expected_idx = i - start_bin;
                    while weights.len() < expected_idx {
                        weights.push(0.0);
                    }
                    weights.push(weight);
                }
            }

            if start_bin == N_BINS {
                start_bin = 0;
            }

            filters.push(MelFilter { start_bin, weights });
        }

        filters
    }

    /// Extract one FBank frame from new f32 samples.
    ///
    /// Input: `FRAME_SHIFT` (160) f32 samples at 16 kHz.
    /// Output: 80-dim log Mel filterbank feature vector.
    ///
    /// Internally buffers the overlap from previous frames to form
    /// the full 400-sample analysis window. Must call
    /// [`extract_frame_full`](Self::extract_frame_full) for the first frame.
    pub fn extract_frame(&mut self, samples: &[f32], output: &mut [f32; N_MEL]) {
        debug_assert_eq!(samples.len(), FRAME_SHIFT);
        debug_assert!(
            !self.first_frame,
            "must call extract_frame_full for the first frame"
        );
        let overlap_len = FRAME_LENGTH - FRAME_SHIFT; // 240

        let mut frame = [0.0f32; FRAME_LENGTH];
        frame[..overlap_len].copy_from_slice(&self.overlap_buffer);
        frame[overlap_len..].copy_from_slice(samples);

        self.overlap_buffer.copy_from_slice(&frame[FRAME_SHIFT..]);
        self.process_frame(&frame, output);
        self.frame_count += 1;
    }

    /// Extract a FBank frame from a complete 400-sample window.
    ///
    /// Used for the first frame or when the caller provides full windows directly.
    pub fn extract_frame_full(
        &mut self,
        frame_samples: &[f32; FRAME_LENGTH],
        output: &mut [f32; N_MEL],
    ) {
        // Store overlap for next frame
        self.overlap_buffer
            .copy_from_slice(&frame_samples[FRAME_SHIFT..]);
        self.first_frame = false;

        self.process_frame(frame_samples, output);
        self.frame_count += 1;
    }

    /// Process a complete 400-sample frame through the FBank pipeline.
    ///
    /// Steps:
    /// 1. Remove DC offset (subtract mean)
    /// 2. Pre-emphasis: x[i] -= 0.97 * x[i-1]; x[0] *= (1 - 0.97)
    /// 3. Apply Povey window
    /// 4. Zero-pad to FFT_SIZE (512)
    /// 5. Compute FFT
    /// 6. Compute power spectrum |X[k]|^2
    /// 7. Apply Mel filterbank
    /// 8. Log compress: ln(max(energy, epsilon))
    fn process_frame(&mut self, frame: &[f32], output: &mut [f32; N_MEL]) {
        let mut work = [0.0f32; FRAME_LENGTH];
        work.copy_from_slice(&frame[..FRAME_LENGTH]);

        // 1. Remove DC offset
        let mean = work.iter().sum::<f32>() / FRAME_LENGTH as f32;
        for s in work.iter_mut() {
            *s -= mean;
        }

        // 2. Pre-emphasis (backwards, Kaldi-style)
        for i in (1..FRAME_LENGTH).rev() {
            work[i] -= PREEMPH_COEFF * work[i - 1];
        }
        work[0] -= PREEMPH_COEFF * work[0]; // = work[0] * (1 - PREEMPH_COEFF)

        // 3. Apply Povey window
        for (s, &w) in work.iter_mut().zip(self.window.iter()) {
            *s *= w;
        }

        // 4. Zero-pad to FFT_SIZE and compute FFT
        self.fft_input[..FRAME_LENGTH].copy_from_slice(&work);
        self.fft_input[FRAME_LENGTH..].fill(0.0);

        self.fft
            .process_with_scratch(
                &mut self.fft_input,
                &mut self.fft_output,
                &mut self.fft_scratch,
            )
            .expect("FFT failed");

        // 5. Power spectrum |X[k]|^2
        for (pow, c) in self.power_spectrum.iter_mut().zip(self.fft_output.iter()) {
            *pow = c.re * c.re + c.im * c.im;
        }

        // 6. Apply Mel filterbank + 7. Log compress
        let epsilon = f32::EPSILON; // ~1.19e-7, matches Kaldi's std::numeric_limits<float>::epsilon()
        for (m, filter) in self.mel_filters.iter().enumerate() {
            let mut energy = 0.0f32;
            let spectrum_slice = &self.power_spectrum[filter.start_bin..];
            for (w, &p) in filter.weights.iter().zip(spectrum_slice.iter()) {
                energy += w * p;
            }
            output[m] = energy.max(epsilon).ln();
        }
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.overlap_buffer.fill(0.0);
        self.first_frame = true;
        self.fft_input.fill(0.0);
        self.power_spectrum.fill(0.0);
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn povey_window_shape() {
        let window = FbankExtractor::povey_window();
        assert_eq!(window.len(), FRAME_LENGTH);

        // Endpoints should be 0
        assert!((window[0]).abs() < 1e-10);
        assert!((window[FRAME_LENGTH - 1]).abs() < 1e-10);

        // Middle should be close to 1
        let mid = window[FRAME_LENGTH / 2];
        assert!(mid > 0.9, "window midpoint = {mid}, expected > 0.9");

        // Should be symmetric
        for i in 0..FRAME_LENGTH / 2 {
            let diff = (window[i] - window[FRAME_LENGTH - 1 - i]).abs();
            assert!(diff < 1e-6, "asymmetry at {i}: {diff}");
        }
    }

    #[test]
    fn mel_filterbank_structure() {
        let filters = FbankExtractor::compute_mel_filterbank();
        assert_eq!(filters.len(), N_MEL);

        // All filters should have some non-zero weights
        for (i, f) in filters.iter().enumerate() {
            assert!(!f.weights.is_empty(), "filter {i} has no weights");
        }

        // Filters should be ordered (start bins increase)
        for i in 1..N_MEL {
            assert!(
                filters[i].start_bin >= filters[i - 1].start_bin,
                "filter {i} start_bin {} < filter {} start_bin {}",
                filters[i].start_bin,
                i - 1,
                filters[i - 1].start_bin
            );
        }
    }

    #[test]
    fn fbank_matches_python_reference() {
        // Load reference data
        let ref_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_fbank.json"
        ));
        let ref_data: serde_json::Value = serde_json::from_str(ref_json).unwrap();
        let ref_fbank: Vec<Vec<f64>> = serde_json::from_value(ref_data["data"].clone()).unwrap();
        let ref_shape: Vec<usize> = serde_json::from_value(ref_data["shape"].clone()).unwrap();
        assert_eq!(ref_shape[1], N_MEL);

        // Load reference samples
        let samples_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_samples.json"
        ));
        let samples_data: serde_json::Value = serde_json::from_str(samples_json).unwrap();
        let samples: Vec<i16> = serde_json::from_value(samples_data["samples"].clone()).unwrap();

        // Extract FBank features frame by frame (matching snip_edges=true)
        let mut extractor = FbankExtractor::new();
        let num_frames = (samples.len() - FRAME_LENGTH) / FRAME_SHIFT + 1;
        assert_eq!(num_frames, ref_shape[0]);

        let mut max_diff: f64 = 0.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * FRAME_SHIFT;
            let end = start + FRAME_LENGTH;
            let frame_samples: Vec<f32> = samples[start..end].iter().map(|&s| s as f32).collect();
            let frame_arr: &[f32; FRAME_LENGTH] = frame_samples.as_slice().try_into().unwrap();

            let mut output = [0.0f32; N_MEL];
            extractor.extract_frame_full(frame_arr, &mut output);

            // Compare with Python reference
            for bin in 0..N_MEL {
                let diff = (output[bin] as f64 - ref_fbank[frame_idx][bin]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Tolerance: 2e-3 for FBank (accounts for float32 FFT differences)
        assert!(
            max_diff < 2e-3,
            "FBank max diff vs Python reference: {max_diff:.8} (tolerance: 2e-3)"
        );
        eprintln!("FBank max diff vs Python reference: {max_diff:.8}");
    }
}
