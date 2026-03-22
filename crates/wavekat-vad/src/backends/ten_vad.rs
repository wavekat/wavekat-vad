//! TEN-VAD backend using pure Rust ONNX inference.
//!
//! This backend wraps [Agora's TEN-VAD](https://github.com/TEN-framework/ten-vad)
//! model, a lightweight neural network for voice activity detection. The full
//! preprocessing pipeline (pre-emphasis, STFT, mel filterbank, pitch estimation)
//! is implemented in pure Rust — only ONNX Runtime (through the
//! [`ort`](https://crates.io/crates/ort) crate) is needed for inference. Returns
//! continuous speech probability scores between 0.0 and 1.0.
//!
//! # Audio Requirements
//!
//! - **Sample rate:** 16000 Hz only
//! - **Frame size:** 256 samples (16 ms)
//! - **Format:** 16-bit signed integers (i16)
//!
//! # Internal State
//!
//! The model maintains 4 hidden-state tensors and the preprocessor keeps
//! several buffers (time-domain window, pre-emphasis state, feature stack)
//! across calls. This means:
//! - Frames **must** be fed sequentially — skipping or reordering frames
//!   will produce inaccurate results.
//! - Call [`reset()`](crate::VoiceActivityDetector::reset) when starting
//!   a new audio stream or after a gap in input.
//!
//! # Preprocessing Pipeline
//!
//! 1. Pre-emphasis filter: `y[n] = x[n] - 0.97 * x[n-1]`
//! 2. STFT: FFT size 1024, hop size 256, window size 768 (Hann)
//! 3. 40-band mel filterbank (0-8000 Hz)
//! 4. Log compression and mean/variance normalization
//! 5. Autocorrelation-based pitch estimation
//! 6. Feature stacking: 3 frames x 41 features = \[1, 3, 41\]
//!
//! # Model Loading
//!
//! The default ONNX model is embedded in the binary at compile time — no
//! external files are needed at runtime. For custom models, use
//! [`TenVad::from_file`] or [`TenVad::from_memory`].
//!
//! # Model
//!
//! Uses the `ten-vad.onnx` model (downloaded at build time):
//! - Inputs: features \[1,3,41\] + 4 hidden states \[1,64\]
//! - Outputs: probability + 4 updated hidden states
//!
//! # Example
//!
//! ```no_run
//! use wavekat_vad::backends::ten_vad::TenVad;
//! use wavekat_vad::VoiceActivityDetector;
//!
//! let mut vad = TenVad::new().unwrap();
//! let samples = vec![0i16; 256]; // 16ms at 16kHz
//! let probability = vad.process(&samples, 16000).unwrap();
//! println!("Speech probability: {probability:.3}");
//! ```

use super::onnx;
use crate::error::VadError;
use crate::{VadCapabilities, VoiceActivityDetector};
use ndarray::{Array2, Array3};
use ort::{inputs, session::Session, value::Tensor};
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Embedded TEN-VAD ONNX model.
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ten-vad.onnx"));

// ============================================================================
// Constants from TEN-VAD C++ implementation
// ============================================================================

/// Sample rate (16kHz only).
const SAMPLE_RATE: u32 = 16000;

/// FFT size.
const FFT_SIZE: usize = 1024;

/// Hop size (frame size).
const HOP_SIZE: usize = 256;

/// Analysis window size.
const WINDOW_SIZE: usize = 768;

/// Number of FFT bins.
const N_BINS: usize = (FFT_SIZE / 2) + 1; // 513

/// Number of mel filter banks.
const N_MEL_BANDS: usize = 40;

/// Feature length (mel bands + pitch).
const FEATURE_LEN: usize = N_MEL_BANDS + 1; // 41

/// Context window length (stacked frames).
const CONTEXT_LEN: usize = 3;

/// Hidden state dimension.
const HIDDEN_DIM: usize = 64;

/// Pre-emphasis coefficient.
const PRE_EMPHASIS: f32 = 0.97;

/// Epsilon for log compression.
const EPS: f32 = 1e-20;

/// Feature means for normalization (40 mel bands + 1 pitch).
/// Values copied from TEN-VAD C++ implementation (coeff.h).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const FEATURE_MEANS: [f32; FEATURE_LEN] = [
    -8.198236465454e+00, -6.265716552734e+00, -5.483818531036e+00,
    -4.758691310883e+00, -4.417088985443e+00, -4.142892837524e+00,
    -3.912850379944e+00, -3.845927953720e+00, -3.657090425491e+00,
    -3.723418712616e+00, -3.876134157181e+00, -3.843890905380e+00,
    -3.690405130386e+00, -3.756065845490e+00, -3.698696136475e+00,
    -3.650463104248e+00, -3.700468778610e+00, -3.567321300507e+00,
    -3.498900175095e+00, -3.477807044983e+00, -3.458816051483e+00,
    -3.444923877716e+00, -3.401328563690e+00, -3.306261301041e+00,
    -3.278556823730e+00, -3.233250856400e+00, -3.198616027832e+00,
    -3.204526424408e+00, -3.208798646927e+00, -3.257838010788e+00,
    -3.381376743317e+00, -3.534021377563e+00, -3.640867948532e+00,
    -3.726858854294e+00, -3.773730993271e+00, -3.804667234421e+00,
    -3.832901000977e+00, -3.871120452881e+00, -3.990592956543e+00,
    -4.480289459229e+00, 9.235690307617e+01,
];

/// Pre-computed reciprocals of `(FEATURE_STDS[i] + EPS)` for fast normalization.
/// Replaces per-frame division with multiplication.
const FEATURE_INV_STDS: [f32; FEATURE_LEN] = {
    let mut inv = [0.0f32; FEATURE_LEN];
    let mut i = 0;
    while i < FEATURE_LEN {
        inv[i] = 1.0 / (FEATURE_STDS[i] + EPS);
        i += 1;
    }
    inv
};

/// Feature standard deviations for normalization (40 mel bands + 1 pitch).
/// Values copied from TEN-VAD C++ implementation (coeff.h).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const FEATURE_STDS: [f32; FEATURE_LEN] = [
    5.166063785553e+00, 4.977209568024e+00, 4.698895931244e+00,
    4.630621433258e+00, 4.634347915649e+00, 4.641156196594e+00,
    4.640676498413e+00, 4.666367053986e+00, 4.650534629822e+00,
    4.640020847321e+00, 4.637400150299e+00, 4.620099067688e+00,
    4.596316337585e+00, 4.562654972076e+00, 4.554360389709e+00,
    4.566910743713e+00, 4.562489986420e+00, 4.562412738800e+00,
    4.585299491882e+00, 4.600179672241e+00, 4.592845916748e+00,
    4.585922718048e+00, 4.583496570587e+00, 4.626092910767e+00,
    4.626957893372e+00, 4.626289367676e+00, 4.637005805969e+00,
    4.683015823364e+00, 4.726813793182e+00, 4.734289646149e+00,
    4.753227233887e+00, 4.849722862244e+00, 4.869434833527e+00,
    4.884482860565e+00, 4.921327114105e+00, 4.959212303162e+00,
    4.996619224548e+00, 5.044823646545e+00, 5.072216987610e+00,
    5.096439361572e+00, 1.152136917114e+02,
];

// ============================================================================
// Preprocessing Components
// ============================================================================

/// Pre-emphasis filter: y\[n\] = x\[n\] - α * x\[n-1\]
struct PreEmphasis {
    prev_sample: f32,
}

impl PreEmphasis {
    fn new() -> Self {
        Self { prev_sample: 0.0 }
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            output[i] = sample - PRE_EMPHASIS * self.prev_sample;
            self.prev_sample = sample;
        }
    }

    fn reset(&mut self) {
        self.prev_sample = 0.0;
    }
}

/// STFT analyzer with Hann window.
struct StftAnalyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    input_buffer: Vec<f32>,
    scratch: Vec<realfft::num_complex::Complex<f32>>,
    /// Reusable FFT output buffer to avoid per-frame allocation.
    spectrum: Vec<realfft::num_complex::Complex<f32>>,
}

impl StftAnalyzer {
    fn new() -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_scratch_len();

        // Generate Hann window (768 samples, zero-padded to 1024)
        let window = Self::generate_hann_window();

        Self {
            fft,
            window,
            input_buffer: vec![0.0; FFT_SIZE],
            scratch: vec![realfft::num_complex::Complex::new(0.0, 0.0); scratch_len],
            spectrum: vec![realfft::num_complex::Complex::new(0.0, 0.0); N_BINS],
        }
    }

    fn generate_hann_window() -> Vec<f32> {
        // Hann window of size 768
        (0..WINDOW_SIZE)
            .map(|i| {
                let phase = std::f32::consts::PI * i as f32 / (WINDOW_SIZE - 1) as f32;
                phase.sin().powi(2)
            })
            .collect()
    }

    /// Compute power spectrum from pre-emphasized audio frame.
    /// Input: WINDOW_SIZE (768) samples
    /// Output: N_BINS (513) power values
    fn compute_power_spectrum(&mut self, input: &[f32], output: &mut [f32]) {
        // Apply Hann window to first WINDOW_SIZE samples, zero-pad the rest
        for (buf, (&inp, &win)) in self.input_buffer[..WINDOW_SIZE]
            .iter_mut()
            .zip(input.iter().zip(self.window.iter()))
        {
            *buf = inp * win;
        }
        self.input_buffer[WINDOW_SIZE..].fill(0.0);

        // Compute FFT (reuses self.spectrum buffer)
        self.fft
            .process_with_scratch(
                &mut self.input_buffer,
                &mut self.spectrum,
                &mut self.scratch,
            )
            .expect("FFT failed");

        // Compute power spectrum (magnitude squared)
        for (out, c) in output.iter_mut().zip(self.spectrum.iter()) {
            *out = c.re * c.re + c.im * c.im;
        }
    }

    fn reset(&mut self) {
        self.input_buffer.fill(0.0);
    }
}

/// Sparse representation of a single mel triangular filter.
/// Only stores the non-zero coefficient range, cutting iteration by ~90%.
struct MelFilter {
    /// First FFT bin with a non-zero coefficient.
    start_bin: usize,
    /// Non-zero filter coefficients (covers `start_bin..start_bin + coefficients.len()`).
    coefficients: Vec<f32>,
}

/// Mel filterbank for converting power spectrum to mel-scale energies.
struct MelFilterbank {
    /// Sparse triangular filters, one per mel band.
    filters: Vec<MelFilter>,
}

impl MelFilterbank {
    fn new() -> Self {
        let filters = Self::compute_filterbank();
        Self { filters }
    }

    fn compute_filterbank() -> Vec<MelFilter> {
        // Mel scale conversion
        fn hz_to_mel(hz: f32) -> f32 {
            2595.0 * (1.0 + hz / 700.0).log10()
        }

        fn mel_to_hz(mel: f32) -> f32 {
            700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
        }

        let low_mel = hz_to_mel(0.0);
        let high_mel = hz_to_mel(8000.0);

        // Compute mel points
        let mut bin_indices = Vec::with_capacity(N_MEL_BANDS + 2);
        for i in 0..=(N_MEL_BANDS + 1) {
            let mel = i as f32 * (high_mel - low_mel) / (N_MEL_BANDS as f32 + 1.0) + low_mel;
            let hz = mel_to_hz(mel);
            let bin = ((FFT_SIZE as f32 + 1.0) * hz / SAMPLE_RATE as f32) as usize;
            bin_indices.push(bin);
        }

        // Build sparse triangular filters
        let mut filters = Vec::with_capacity(N_MEL_BANDS);
        for j in 0..N_MEL_BANDS {
            let left = bin_indices[j];
            let right = bin_indices[j + 2];
            let center = bin_indices[j + 1];
            let span = right - left;
            let mut coefficients = vec![0.0; span];

            // Rising edge
            if center > left {
                for i in left..center {
                    coefficients[i - left] = (i - left) as f32 / (center - left) as f32;
                }
            }

            // Falling edge
            if right > center {
                for i in center..right {
                    coefficients[i - left] = (right - i) as f32 / (right - center) as f32;
                }
            }

            filters.push(MelFilter {
                start_bin: left,
                coefficients,
            });
        }

        filters
    }

    /// Apply mel filterbank to power spectrum.
    fn apply(&self, power_spectrum: &[f32], output: &mut [f32]) {
        // Power normalizer (input is assumed to be in [-32768, 32767] scale)
        let power_normal = 32768.0 * 32768.0;

        for (band, filter) in self.filters.iter().enumerate() {
            let mut energy = 0.0;
            let spectrum_slice = &power_spectrum[filter.start_bin..];
            for (coef, &power) in filter.coefficients.iter().zip(spectrum_slice.iter()) {
                energy += power * coef;
            }
            // Normalize and log compress
            energy /= power_normal;
            output[band] = (energy + EPS).ln();
        }
    }
}

/// Simple pitch estimator using autocorrelation.
///
/// This is a simplified implementation that estimates pitch frequency
/// from the autocorrelation of the input signal.
struct PitchEstimator {
    /// Buffer for autocorrelation computation
    buffer: Vec<f32>,
    /// Previous pitch estimate for smoothing
    prev_pitch: f32,
}

impl PitchEstimator {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(HOP_SIZE * 2),
            prev_pitch: 0.0,
        }
    }

    /// Estimate pitch frequency from audio frame.
    /// Returns pitch in Hz, or 0.0 if no pitch detected.
    ///
    /// Uses a coarse-to-fine two-stage autocorrelation search:
    /// 1. Coarse pass scans periods at stride 4 (~72 iterations vs ~289)
    /// 2. Fine pass refines ±4 around the best coarse candidate (~8 iterations)
    fn estimate(&mut self, samples: &[f32]) -> f32 {
        // Pitch range: 50 Hz - 500 Hz at 16kHz = period 32-320 samples
        let min_period = SAMPLE_RATE as usize / 500; // ~32 samples
        let max_period = SAMPLE_RATE as usize / 50; // ~320 samples

        let len = samples.len().min(max_period + 64);
        if len < min_period + 32 {
            return 0.0;
        }

        // Compute energy
        let energy: f32 = samples.iter().take(len).map(|&x| x * x).sum();
        if energy < 1e-6 {
            self.prev_pitch = 0.0;
            return 0.0;
        }

        let upper = max_period.min(len - 32);

        // Helper: compute normalized autocorrelation for a given period
        let autocorrelation = |period: usize| -> f32 {
            let mut corr = 0.0;
            let mut energy1 = 0.0;
            let mut energy2 = 0.0;
            for i in 0..64.min(len - period) {
                corr += samples[i] * samples[i + period];
                energy1 += samples[i] * samples[i];
                energy2 += samples[i + period] * samples[i + period];
            }
            let norm = (energy1 * energy2).sqrt();
            if norm > 1e-10 {
                corr / norm
            } else {
                0.0
            }
        };

        // Coarse pass: scan at stride 4
        const COARSE_STRIDE: usize = 4;
        let mut best_period = 0;
        let mut best_corr = 0.0;

        let mut period = min_period;
        while period <= upper {
            let corr = autocorrelation(period);
            if corr > best_corr {
                best_corr = corr;
                best_period = period;
            }
            period += COARSE_STRIDE;
        }

        // Fine pass: search ±COARSE_STRIDE around the best coarse candidate
        if best_period > 0 {
            let fine_start = best_period.saturating_sub(COARSE_STRIDE).max(min_period);
            let fine_end = (best_period + COARSE_STRIDE).min(upper);
            for period in fine_start..=fine_end {
                let corr = autocorrelation(period);
                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }
        }

        // Require minimum correlation for voiced detection
        let pitch = if best_corr > 0.4 && best_period > 0 {
            SAMPLE_RATE as f32 / best_period as f32
        } else {
            0.0
        };

        // Simple smoothing
        self.prev_pitch = pitch * 0.7 + self.prev_pitch * 0.3;
        self.prev_pitch
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_pitch = 0.0;
    }
}

/// Complete TEN-VAD preprocessor.
struct TenVadPreprocessor {
    pre_emphasis: PreEmphasis,
    stft: StftAnalyzer,
    mel_filterbank: MelFilterbank,
    pitch_estimator: PitchEstimator,
    /// Time-domain input buffer (for windowing)
    time_buffer: Vec<f32>,
    /// Pre-emphasized buffer
    emph_buffer: Vec<f32>,
    /// Feature stack: [CONTEXT_LEN][FEATURE_LEN]
    feature_stack: Vec<f32>,
    /// Number of frames processed (for warmup)
    frame_count: usize,
    /// Reusable buffer for i16→f32 conversion (avoids per-frame allocation).
    samples_f32: Vec<f32>,
    /// Reusable power spectrum buffer (avoids per-frame allocation).
    power_spectrum: Vec<f32>,
}

impl TenVadPreprocessor {
    fn new() -> Self {
        Self {
            pre_emphasis: PreEmphasis::new(),
            stft: StftAnalyzer::new(),
            mel_filterbank: MelFilterbank::new(),
            pitch_estimator: PitchEstimator::new(),
            time_buffer: vec![0.0; WINDOW_SIZE],
            emph_buffer: vec![0.0; WINDOW_SIZE],
            feature_stack: vec![0.0; CONTEXT_LEN * FEATURE_LEN],
            frame_count: 0,
            samples_f32: vec![0.0; HOP_SIZE],
            power_spectrum: vec![0.0; N_BINS],
        }
    }

    /// Process one frame of audio and return features.
    /// Input: HOP_SIZE (256) i16 samples
    /// Output: [CONTEXT_LEN, FEATURE_LEN] = [3, 41] features (flattened)
    fn process(&mut self, samples: &[i16]) -> &[f32] {
        // Convert i16 to f32 (reuse buffer, no allocation)
        for (dst, &src) in self.samples_f32.iter_mut().zip(samples.iter()) {
            *dst = src as f32;
        }

        // Shift time buffer and append new samples
        self.time_buffer.copy_within(HOP_SIZE.., 0);
        self.time_buffer[WINDOW_SIZE - HOP_SIZE..].copy_from_slice(&self.samples_f32);

        // Shift emph_buffer and only pre-emphasize the new samples.
        // Set prev_sample to the last old sample for correct filter continuity.
        self.emph_buffer.copy_within(HOP_SIZE.., 0);
        self.pre_emphasis.prev_sample = self.time_buffer[WINDOW_SIZE - HOP_SIZE - 1];
        self.pre_emphasis.process(
            &self.time_buffer[WINDOW_SIZE - HOP_SIZE..],
            &mut self.emph_buffer[WINDOW_SIZE - HOP_SIZE..],
        );

        // Compute power spectrum (reuse buffer, no allocation)
        self.stft
            .compute_power_spectrum(&self.emph_buffer, &mut self.power_spectrum);

        // Compute mel features
        let mut mel_features = [0.0f32; N_MEL_BANDS];
        self.mel_filterbank
            .apply(&self.power_spectrum, &mut mel_features);

        // Estimate pitch
        let pitch_freq = self.pitch_estimator.estimate(&self.samples_f32);

        // Shift feature stack and add new features
        self.feature_stack.copy_within(FEATURE_LEN.., 0);

        // Add normalized features to the end of the stack (multiply by reciprocal)
        let dst_offset = (CONTEXT_LEN - 1) * FEATURE_LEN;
        for (i, &mel) in mel_features.iter().enumerate() {
            self.feature_stack[dst_offset + i] = (mel - FEATURE_MEANS[i]) * FEATURE_INV_STDS[i];
        }
        // Pitch feature (last feature)
        self.feature_stack[dst_offset + N_MEL_BANDS] =
            (pitch_freq - FEATURE_MEANS[N_MEL_BANDS]) * FEATURE_INV_STDS[N_MEL_BANDS];

        self.frame_count += 1;

        &self.feature_stack
    }

    fn reset(&mut self) {
        self.pre_emphasis.reset();
        self.stft.reset();
        self.pitch_estimator.reset();
        self.time_buffer.fill(0.0);
        self.emph_buffer.fill(0.0);
        self.feature_stack.fill(0.0);
        self.frame_count = 0;
        self.samples_f32.fill(0.0);
        self.power_spectrum.fill(0.0);
    }
}

// ============================================================================
// Main VAD Implementation
// ============================================================================

/// Voice activity detector using Agora's TEN-VAD ONNX model with pure Rust
/// preprocessing.
///
/// Accepts 16 kHz / 256-sample (16 ms) frames and returns a continuous
/// speech probability (0.0–1.0). The full preprocessing pipeline (pre-emphasis,
/// STFT, mel filterbank, pitch estimation) runs in Rust — no external
/// libraries beyond ONNX Runtime are required.
///
/// Internal state (hidden states + preprocessor buffers) persists across
/// calls. Call [`reset()`](VoiceActivityDetector::reset) when switching
/// to a new audio stream. See the [module-level docs](self) for the
/// full preprocessing pipeline description.
pub struct TenVad {
    /// ONNX Runtime session.
    session: Session,
    /// Preprocessor for feature extraction.
    preprocessor: TenVadPreprocessor,
    /// Hidden states: 4 tensors of shape [1, 64].
    hidden_states: [Array2<f32>; 4],
}

// SAFETY: ort::Session is Send in ort 2.x, and all other fields are owned Send types.
unsafe impl Send for TenVad {}

impl TenVad {
    /// Create a new TEN-VAD instance using the embedded model.
    ///
    /// The ONNX model is embedded in the binary at compile time — no
    /// external files are needed at runtime.
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if the ONNX session fails to initialize.
    pub fn new() -> Result<Self, VadError> {
        Self::from_memory(MODEL_BYTES)
    }

    /// Create a new TEN-VAD instance from a custom ONNX model file.
    ///
    /// Use this to load a different model version or a custom-trained model.
    /// The model must be compatible with the TEN-VAD input/output format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if the model file cannot be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_vad::backends::ten_vad::TenVad;
    ///
    /// let vad = TenVad::from_file("path/to/custom_model.onnx").unwrap();
    /// ```
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VadError> {
        Self::from_session(onnx::session_from_file(path)?)
    }

    /// Create a new TEN-VAD instance from model bytes in memory.
    ///
    /// Use this to load a custom or alternative ONNX model. The model
    /// must be compatible with the TEN-VAD input/output format.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Raw ONNX model data
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if the ONNX session fails to initialize.
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, VadError> {
        Self::from_session(onnx::session_from_memory(model_bytes)?)
    }

    fn from_session(session: Session) -> Result<Self, VadError> {
        let hidden_states = [
            Array2::<f32>::zeros((1, HIDDEN_DIM)),
            Array2::<f32>::zeros((1, HIDDEN_DIM)),
            Array2::<f32>::zeros((1, HIDDEN_DIM)),
            Array2::<f32>::zeros((1, HIDDEN_DIM)),
        ];

        Ok(Self {
            session,
            preprocessor: TenVadPreprocessor::new(),
            hidden_states,
        })
    }
}

impl VoiceActivityDetector for TenVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: SAMPLE_RATE,
            frame_size: HOP_SIZE,
            frame_duration_ms: (HOP_SIZE as u32 * 1000) / SAMPLE_RATE,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // Validate sample rate
        if sample_rate != SAMPLE_RATE {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        // Validate frame size
        if samples.len() != HOP_SIZE {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: HOP_SIZE,
            });
        }

        // Run preprocessing
        let features = self.preprocessor.process(samples);

        // Create feature tensor: shape [1, 3, 41]
        let feature_array =
            Array3::from_shape_vec((1, CONTEXT_LEN, FEATURE_LEN), features.to_vec()).map_err(
                |e| VadError::BackendError(format!("failed to create feature array: {e}")),
            )?;
        let feature_tensor = Tensor::from_array(feature_array)
            .map_err(|e| VadError::BackendError(format!("failed to create feature tensor: {e}")))?;

        // Create hidden state tensors
        let h0_tensor = Tensor::from_array(self.hidden_states[0].clone())
            .map_err(|e| VadError::BackendError(format!("failed to create h0 tensor: {e}")))?;
        let h1_tensor = Tensor::from_array(self.hidden_states[1].clone())
            .map_err(|e| VadError::BackendError(format!("failed to create h1 tensor: {e}")))?;
        let h2_tensor = Tensor::from_array(self.hidden_states[2].clone())
            .map_err(|e| VadError::BackendError(format!("failed to create h2 tensor: {e}")))?;
        let h3_tensor = Tensor::from_array(self.hidden_states[3].clone())
            .map_err(|e| VadError::BackendError(format!("failed to create h3 tensor: {e}")))?;

        // Run inference
        // Model inputs: input_1 (features), input_2/3/6/7 (hidden states)
        // Model outputs: output_1 (probability), output_2/3/6/7 (updated hidden states)
        let outputs = self
            .session
            .run(inputs![
                "input_1" => feature_tensor,
                "input_2" => h0_tensor,
                "input_3" => h1_tensor,
                "input_6" => h2_tensor,
                "input_7" => h3_tensor,
            ])
            .map_err(|e| VadError::BackendError(format!("inference failed: {e}")))?;

        // Extract output probability
        let output = outputs
            .get("output_1")
            .ok_or_else(|| VadError::BackendError("missing 'output_1' tensor".into()))?;
        let (_, output_data): (_, &[f32]) = output
            .try_extract_tensor()
            .map_err(|e| VadError::BackendError(format!("failed to extract output: {e}")))?;
        let probability = *output_data
            .first()
            .ok_or_else(|| VadError::BackendError("empty output tensor".into()))?;

        // Update hidden states
        for (i, name) in ["output_2", "output_3", "output_6", "output_7"]
            .iter()
            .enumerate()
        {
            let h_out = outputs
                .get(*name)
                .ok_or_else(|| VadError::BackendError(format!("missing '{name}' tensor")))?;
            let (_, h_data): (_, &[f32]) = h_out
                .try_extract_tensor()
                .map_err(|e| VadError::BackendError(format!("failed to extract {name}: {e}")))?;

            if h_data.len() == HIDDEN_DIM {
                self.hidden_states[i]
                    .as_slice_mut()
                    .ok_or_else(|| {
                        VadError::BackendError("hidden state buffer not contiguous".into())
                    })?
                    .copy_from_slice(h_data);
            }
        }

        Ok(probability.clamp(0.0, 1.0))
    }

    fn reset(&mut self) {
        self.preprocessor.reset();
        for h in &mut self.hidden_states {
            h.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_ten_vad_onnx() {
        let vad = TenVad::new();
        assert!(vad.is_ok(), "Failed to create TenVad: {:?}", vad.err());
    }

    #[test]
    fn capabilities() {
        let vad = TenVad::new().unwrap();
        let caps = vad.capabilities();
        assert_eq!(caps.sample_rate, 16000);
        assert_eq!(caps.frame_size, 256);
        assert_eq!(caps.frame_duration_ms, 16);
    }

    #[test]
    fn process_silence() {
        let mut vad = TenVad::new().unwrap();
        let silence = vec![0i16; 256];
        let result = vad.process(&silence, 16000);
        assert!(
            result.is_ok(),
            "Failed to process silence: {:?}",
            result.err()
        );
        let prob = result.unwrap();
        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Probability out of range: {prob}"
        );
    }

    #[test]
    fn process_wrong_sample_rate() {
        let mut vad = TenVad::new().unwrap();
        let samples = vec![0i16; 256];
        let result = vad.process(&samples, 8000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(8000))));
    }

    #[test]
    fn process_wrong_frame_size() {
        let mut vad = TenVad::new().unwrap();
        let samples = vec![0i16; 100];
        let result = vad.process(&samples, 16000);
        assert!(matches!(
            result,
            Err(VadError::InvalidFrameSize {
                got: 100,
                expected: 256
            })
        ));
    }

    #[test]
    fn probability_in_range() {
        let mut vad = TenVad::new().unwrap();
        // Generate some test signal (low amplitude noise)
        let samples: Vec<i16> = (0..256).map(|i| (i % 100) as i16 * 50).collect();
        let result = vad.process(&samples, 16000).unwrap();
        assert!(
            result >= 0.0 && result <= 1.0,
            "Probability out of range: {result}"
        );
    }

    #[test]
    fn reset_works() {
        let mut vad = TenVad::new().unwrap();
        let samples: Vec<i16> = (0..256).map(|i| i as i16 * 10).collect();

        // Process some audio
        let _ = vad.process(&samples, 16000).unwrap();

        // Reset
        vad.reset();

        // Process silence - should work and give low probability
        let silence = vec![0i16; 256];
        let result = vad.process(&silence, 16000);
        assert!(result.is_ok());
    }

    #[test]
    fn multiple_frames() {
        let mut vad = TenVad::new().unwrap();
        let silence = vec![0i16; 256];

        // Process multiple frames
        for _ in 0..10 {
            let result = vad.process(&silence, 16000);
            assert!(result.is_ok());
            let prob = result.unwrap();
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn mel_filterbank_initialization() {
        let fb = MelFilterbank::new();
        assert_eq!(fb.filters.len(), N_MEL_BANDS);
        // Each sparse filter should have non-zero coefficients
        for filter in &fb.filters {
            assert!(
                !filter.coefficients.is_empty(),
                "filter at bin {} has no coefficients",
                filter.start_bin
            );
            assert!(
                filter.start_bin + filter.coefficients.len() <= N_BINS,
                "filter exceeds spectrum bounds"
            );
        }
    }

    #[test]
    fn preprocessor_reset() {
        let mut prep = TenVadPreprocessor::new();
        let samples = vec![1000i16; 256];

        // Process some audio
        let _ = prep.process(&samples);

        // Reset
        prep.reset();

        assert_eq!(prep.frame_count, 0);
        assert!(prep.feature_stack.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_memory_with_embedded_model() {
        // Verify from_memory works with the same bytes that new() uses
        let vad = TenVad::from_memory(MODEL_BYTES);
        assert!(vad.is_ok(), "from_memory failed: {:?}", vad.err());
    }

    #[test]
    fn from_memory_invalid_bytes() {
        let result = TenVad::from_memory(b"not a valid onnx model");
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn from_file_nonexistent() {
        let result = TenVad::from_file("/nonexistent/model.onnx");
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn from_file_with_temp_model() {
        // Write the embedded model to a temp file, then load it via from_file
        let dir = std::env::temp_dir().join("wavekat_vad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("ten-vad-test.onnx");
        std::fs::write(&model_path, MODEL_BYTES).unwrap();

        let result = TenVad::from_file(&model_path);
        assert!(result.is_ok(), "from_file failed: {:?}", result.err());

        // Verify the loaded model works
        let mut vad = result.unwrap();
        let silence = vec![0i16; 256];
        let prob = vad.process(&silence, 16000).unwrap();
        assert!(prob >= 0.0 && prob <= 1.0);

        // Cleanup
        let _ = std::fs::remove_file(&model_path);
    }
}
