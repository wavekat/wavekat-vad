//! FireRedVAD backend using pure Rust preprocessing + ONNX inference.
//!
//! This backend wraps [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD),
//! a state-of-the-art voice activity detector from Xiaohongshu using a DFSMN
//! (Deep Feedforward Sequential Memory Network) architecture. The full
//! preprocessing pipeline (FBank feature extraction + CMVN normalization)
//! is implemented in pure Rust — only ONNX Runtime (through the
//! [`ort`](https://crates.io/crates/ort) crate) is needed for inference.
//! Returns continuous speech probability scores between 0.0 and 1.0.
//!
//! # Audio Requirements
//!
//! - **Sample rate:** 16000 Hz only
//! - **Frame size:** 160 samples (10 ms)
//! - **Format:** 16-bit signed integers (i16)
//!
//! # Internal State
//!
//! The model maintains 8 DFSMN cache tensors (shape `[8, 1, 128, 19]`) and
//! the preprocessor keeps an overlap buffer across calls. This means:
//! - Frames **must** be fed sequentially — skipping or reordering frames
//!   will produce inaccurate results.
//! - Call [`reset()`](crate::VoiceActivityDetector::reset) when starting
//!   a new audio stream or after a gap in input.
//!
//! # Preprocessing Pipeline
//!
//! 1. Buffer 160 samples into 400-sample windows (25 ms, 10 ms hop)
//! 2. Remove DC offset
//! 3. Pre-emphasis: 0.97 coefficient
//! 4. Povey window
//! 5. 512-point FFT → power spectrum
//! 6. 80-band Mel filterbank (20–8000 Hz, Kaldi mel scale)
//! 7. Log compression
//! 8. CMVN normalization (global mean/variance from `cmvn.ark`)
//!
//! # Model Loading
//!
//! The default ONNX model and CMVN file are embedded in the binary at
//! compile time — no external files are needed at runtime. For custom
//! models, use [`FireRedVad::from_file`] or [`FireRedVad::from_memory`].
//!
//! # Example
//!
//! ```no_run
//! use wavekat_vad::backends::firered::FireRedVad;
//! use wavekat_vad::VoiceActivityDetector;
//!
//! let mut vad = FireRedVad::new().unwrap();
//! let samples = vec![0i16; 160]; // 10ms at 16kHz
//! let probability = vad.process(&samples, 16000).unwrap();
//! println!("Speech probability: {probability:.3}");
//! ```

pub(crate) mod cmvn;
pub(crate) mod fbank;

use super::onnx;
use crate::error::VadError;
use crate::{VadCapabilities, VoiceActivityDetector};
use cmvn::CmvnStats;
use fbank::FbankExtractor;
use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

/// Embedded FireRedVAD ONNX model (streaming with cache).
const MODEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/fireredvad_stream_vad_with_cache.onnx"
));

/// Embedded CMVN statistics file (Kaldi binary format).
const CMVN_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/firered_cmvn.ark"));

/// Sample rate (16 kHz only).
const SAMPLE_RATE: u32 = 16000;

/// Frame shift (10 ms at 16 kHz).
const FRAME_SHIFT: usize = fbank::FRAME_SHIFT; // 160

/// Frame length for windowed analysis (25 ms at 16 kHz).
const FRAME_LENGTH: usize = 400;

/// Number of Mel filter banks.
const N_MEL: usize = 80;

/// DFSMN cache dimensions.
const CACHE_LAYERS: usize = 8;
const CACHE_BATCH: usize = 1;
const CACHE_PROJ: usize = 128;
const CACHE_LOOKBACK: usize = 19;

/// Voice activity detector backed by the FireRedVAD ONNX model with pure
/// Rust preprocessing.
///
/// Accepts 16 kHz / 160-sample (10 ms) frames and returns a continuous
/// speech probability (0.0–1.0). The full preprocessing pipeline
/// (FBank + CMVN) runs in Rust — no external libraries beyond ONNX
/// Runtime are required.
///
/// Internal state (DFSMN caches + preprocessor buffers) persists across
/// calls. Call [`reset()`](VoiceActivityDetector::reset) when switching
/// to a new audio stream. See the [module-level docs](self) for the
/// full preprocessing pipeline description.
pub struct FireRedVad {
    /// ONNX Runtime session.
    session: Session,
    /// FBank feature extractor.
    fbank: FbankExtractor,
    /// CMVN statistics for normalization.
    cmvn: CmvnStats,
    /// DFSMN cache state: shape [8, 1, 128, 19].
    caches: Array4<f32>,
    /// Sample accumulation buffer for building full frames.
    /// Collects samples until we have FRAME_LENGTH (400) for the first frame,
    /// then FRAME_SHIFT (160) for subsequent frames.
    sample_buffer: Vec<f32>,
    /// Total frames produced so far.
    frame_count: usize,
}

// SAFETY: ort::Session is Send in ort 2.x, and all other fields are owned Send types.
unsafe impl Send for FireRedVad {}

impl FireRedVad {
    /// Create a new FireRedVAD instance using the embedded model and CMVN.
    ///
    /// The ONNX model and CMVN statistics are embedded in the binary at
    /// compile time — no external files are needed at runtime.
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if the ONNX session or CMVN parsing fails.
    pub fn new() -> Result<Self, VadError> {
        let cmvn = CmvnStats::from_kaldi_binary(CMVN_BYTES)?;
        Self::from_session(onnx::session_from_memory(MODEL_BYTES)?, cmvn)
    }

    /// Create a new FireRedVAD instance from custom model and CMVN files.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `cmvn_path` - Path to the Kaldi-format CMVN ark file
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if files cannot be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_vad::backends::firered::FireRedVad;
    ///
    /// let vad = FireRedVad::from_file("model.onnx", "cmvn.ark").unwrap();
    /// ```
    pub fn from_file<P: AsRef<std::path::Path>>(
        model_path: P,
        cmvn_path: P,
    ) -> Result<Self, VadError> {
        let cmvn_data = std::fs::read(cmvn_path.as_ref()).map_err(|e| {
            VadError::BackendError(format!(
                "failed to read CMVN file '{}': {e}",
                cmvn_path.as_ref().display()
            ))
        })?;
        let cmvn = CmvnStats::from_kaldi_binary(&cmvn_data)?;
        Self::from_session(onnx::session_from_file(model_path)?, cmvn)
    }

    /// Create a new FireRedVAD instance from model and CMVN bytes in memory.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Raw ONNX model data
    /// * `cmvn_bytes` - Raw Kaldi-format CMVN data
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if parsing fails.
    pub fn from_memory(model_bytes: &[u8], cmvn_bytes: &[u8]) -> Result<Self, VadError> {
        let cmvn = CmvnStats::from_kaldi_binary(cmvn_bytes)?;
        Self::from_session(onnx::session_from_memory(model_bytes)?, cmvn)
    }

    fn from_session(session: Session, cmvn: CmvnStats) -> Result<Self, VadError> {
        Ok(Self {
            session,
            fbank: FbankExtractor::new(),
            cmvn,
            caches: Array4::<f32>::zeros((CACHE_LAYERS, CACHE_BATCH, CACHE_PROJ, CACHE_LOOKBACK)),
            sample_buffer: Vec::with_capacity(FRAME_LENGTH),
            frame_count: 0,
        })
    }

    /// Run ONNX inference on a single normalized feature frame.
    fn run_inference(&mut self, features: &[f32; N_MEL]) -> Result<f32, VadError> {
        // Create feature tensor: shape [1, 1, 80] — zero-copy view over the slice
        let feat_tensor = TensorRef::from_array_view(([1i64, 1, N_MEL as i64], &features[..]))
            .map_err(|e| VadError::BackendError(format!("failed to create feature tensor: {e}")))?;

        // Create cache tensor: zero-copy view over the existing array (no clone)
        let cache_tensor = TensorRef::from_array_view(self.caches.view())
            .map_err(|e| VadError::BackendError(format!("failed to create cache tensor: {e}")))?;

        // Run inference
        let outputs = self
            .session
            .run(inputs![
                "feat" => feat_tensor,
                "caches_in" => cache_tensor,
            ])
            .map_err(|e| VadError::BackendError(format!("inference failed: {e}")))?;

        // Extract probability
        let probs = outputs
            .get("probs")
            .ok_or_else(|| VadError::BackendError("missing 'probs' tensor".into()))?;
        let (_, probs_data): (_, &[f32]) = probs
            .try_extract_tensor()
            .map_err(|e| VadError::BackendError(format!("failed to extract probs: {e}")))?;
        let probability = *probs_data
            .first()
            .ok_or_else(|| VadError::BackendError("empty probs tensor".into()))?;

        // Update caches
        let new_caches = outputs
            .get("caches_out")
            .ok_or_else(|| VadError::BackendError("missing 'caches_out' tensor".into()))?;
        let (_, cache_data): (_, &[f32]) = new_caches
            .try_extract_tensor()
            .map_err(|e| VadError::BackendError(format!("failed to extract caches: {e}")))?;

        let expected_cache_size = CACHE_LAYERS * CACHE_BATCH * CACHE_PROJ * CACHE_LOOKBACK;
        if cache_data.len() == expected_cache_size {
            self.caches
                .as_slice_mut()
                .ok_or_else(|| VadError::BackendError("cache buffer not contiguous".into()))?
                .copy_from_slice(cache_data);
        } else {
            return Err(VadError::BackendError(format!(
                "unexpected cache size: expected {expected_cache_size}, got {}",
                cache_data.len()
            )));
        }

        Ok(probability.clamp(0.0, 1.0))
    }
}

impl VoiceActivityDetector for FireRedVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: SAMPLE_RATE,
            frame_size: FRAME_SHIFT,
            frame_duration_ms: (FRAME_SHIFT as u32 * 1000) / SAMPLE_RATE,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // Validate sample rate
        if sample_rate != SAMPLE_RATE {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        // Validate frame size
        if samples.len() != FRAME_SHIFT {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: FRAME_SHIFT,
            });
        }

        // Add samples to buffer
        for &s in samples {
            self.sample_buffer.push(s as f32);
        }

        // Check if we have enough samples for a frame
        let needed = if self.frame_count == 0 {
            FRAME_LENGTH // First frame needs 400 samples
        } else {
            FRAME_SHIFT // Subsequent frames need 160 new samples
        };

        if self.sample_buffer.len() < needed {
            // Not enough samples yet — return 0 probability
            // This only happens for the first 2 calls (need 400 samples = 2.5 × 160)
            return Ok(0.0);
        }

        // Extract FBank features
        let mut fbank_features = [0.0f32; N_MEL];

        if self.frame_count == 0 {
            // First frame: use first FRAME_LENGTH samples
            let frame: &[f32; FRAME_LENGTH] = self.sample_buffer[..FRAME_LENGTH]
                .try_into()
                .map_err(|_| VadError::BackendError("buffer size mismatch".into()))?;
            self.fbank.extract_frame_full(frame, &mut fbank_features);

            // Keep the overlap portion for next frame
            let drain_len = FRAME_SHIFT;
            self.sample_buffer.drain(..drain_len);
        } else {
            // Subsequent frames: overlap is already stored in FbankExtractor
            self.fbank
                .extract_frame(&self.sample_buffer[..FRAME_SHIFT], &mut fbank_features);
            self.sample_buffer.drain(..FRAME_SHIFT);
        }

        self.frame_count += 1;

        // Apply CMVN normalization
        self.cmvn.normalize(&mut fbank_features);

        // Run inference
        self.run_inference(&fbank_features)
    }

    fn reset(&mut self) {
        self.fbank.reset();
        self.caches.fill(0.0);
        self.sample_buffer.clear();
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_succeeds() {
        let vad = FireRedVad::new();
        assert!(vad.is_ok(), "Failed to create FireRedVad: {:?}", vad.err());
    }

    #[test]
    fn capabilities() {
        let vad = FireRedVad::new().unwrap();
        let caps = vad.capabilities();
        assert_eq!(caps.sample_rate, 16000);
        assert_eq!(caps.frame_size, 160);
        assert_eq!(caps.frame_duration_ms, 10);
    }

    #[test]
    fn process_silence() {
        let mut vad = FireRedVad::new().unwrap();
        let silence = vec![0i16; 160];

        // Feed enough frames for initial buffering (need 400 samples = 3 frames of 160)
        let _ = vad.process(&silence, 16000).unwrap(); // 160 samples, not enough
        let _ = vad.process(&silence, 16000).unwrap(); // 320 samples, not enough
        let prob = vad.process(&silence, 16000).unwrap(); // 480 samples, first frame produced

        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Probability out of range: {prob}"
        );
    }

    #[test]
    fn process_wrong_sample_rate() {
        let mut vad = FireRedVad::new().unwrap();
        let samples = vec![0i16; 160];
        let result = vad.process(&samples, 8000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(8000))));
    }

    #[test]
    fn process_wrong_frame_size() {
        let mut vad = FireRedVad::new().unwrap();
        let samples = vec![0i16; 100];
        let result = vad.process(&samples, 16000);
        assert!(matches!(
            result,
            Err(VadError::InvalidFrameSize {
                got: 100,
                expected: 160
            })
        ));
    }

    #[test]
    fn reset_works() {
        let mut vad = FireRedVad::new().unwrap();
        let samples: Vec<i16> = (0..160).map(|i| (i * 10) as i16).collect();

        // Process some audio
        let _ = vad.process(&samples, 16000).unwrap();
        let _ = vad.process(&samples, 16000).unwrap();
        let _ = vad.process(&samples, 16000).unwrap();

        // Reset
        vad.reset();

        // Should work again
        let silence = vec![0i16; 160];
        let result = vad.process(&silence, 16000);
        assert!(result.is_ok());
    }

    #[test]
    fn multiple_frames() {
        let mut vad = FireRedVad::new().unwrap();
        let silence = vec![0i16; 160];

        for _ in 0..10 {
            let result = vad.process(&silence, 16000);
            assert!(result.is_ok());
            let prob = result.unwrap();
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn from_memory_with_embedded_model() {
        let vad = FireRedVad::from_memory(MODEL_BYTES, CMVN_BYTES);
        assert!(vad.is_ok(), "from_memory failed: {:?}", vad.err());
    }

    #[test]
    fn from_memory_invalid_bytes() {
        let result = FireRedVad::from_memory(b"not a valid onnx model", CMVN_BYTES);
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn from_file_nonexistent() {
        let result = FireRedVad::from_file("/nonexistent/model.onnx", "/nonexistent/cmvn.ark");
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn probabilities_match_python_reference() {
        // Load reference data
        let samples_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_samples.json"
        ));
        let samples_data: serde_json::Value = serde_json::from_str(samples_json).unwrap();
        let samples: Vec<i16> = serde_json::from_value(samples_data["samples"].clone()).unwrap();

        let probs_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_probs.json"
        ));
        let probs_data: serde_json::Value = serde_json::from_str(probs_json).unwrap();
        let ref_probs: Vec<f64> = serde_json::from_value(probs_data["probs"].clone()).unwrap();

        // Process samples through our Rust pipeline
        // The Python reference processes full frames from a file (snip_edges=true),
        // so we need to match that behavior. The Python gets 98 frames from 16000 samples:
        // num_frames = (16000 - 400) / 160 + 1 = 98
        //
        // Our streaming API buffers samples differently, so let's use
        // extract_frame_full directly to match the Python pipeline exactly.
        let cmvn = CmvnStats::from_kaldi_binary(CMVN_BYTES).unwrap();
        let mut session = onnx::session_from_memory(MODEL_BYTES).unwrap();
        let mut fbank = FbankExtractor::new();
        let mut caches =
            Array4::<f32>::zeros((CACHE_LAYERS, CACHE_BATCH, CACHE_PROJ, CACHE_LOOKBACK));

        let num_frames = (samples.len() - 400) / 160 + 1;
        assert_eq!(num_frames, ref_probs.len());

        let mut max_diff: f64 = 0.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * 160;
            let end = start + 400;
            let frame_samples: Vec<f32> = samples[start..end].iter().map(|&s| s as f32).collect();
            let frame_arr: &[f32; 400] = frame_samples.as_slice().try_into().unwrap();

            // Extract FBank
            let mut features = [0.0f32; 80];
            fbank.extract_frame_full(frame_arr, &mut features);

            // Apply CMVN
            cmvn.normalize(&mut features);

            // Run inference (zero-copy tensor views)
            let feat_tensor =
                TensorRef::from_array_view(([1i64, 1, 80], &features[..])).unwrap();
            let cache_tensor = TensorRef::from_array_view(caches.view()).unwrap();

            let outputs = session
                .run(inputs![
                    "feat" => feat_tensor,
                    "caches_in" => cache_tensor,
                ])
                .unwrap();

            let probs = outputs.get("probs").unwrap();
            let (_, probs_data): (_, &[f32]) = probs.try_extract_tensor().unwrap();
            let probability = probs_data[0];

            let new_caches = outputs.get("caches_out").unwrap();
            let (_, cache_data): (_, &[f32]) = new_caches.try_extract_tensor().unwrap();
            caches.as_slice_mut().unwrap().copy_from_slice(cache_data);

            let diff = (probability as f64 - ref_probs[frame_idx]).abs();
            if diff > max_diff {
                max_diff = diff;
            }

            // Print first few for debugging
            if frame_idx < 5 {
                eprintln!(
                    "  frame {frame_idx}: rust={probability:.6}, python={:.6}, diff={diff:.8}",
                    ref_probs[frame_idx]
                );
            }
        }

        eprintln!("Max probability diff vs Python: {max_diff:.8}");

        // Tolerance: 0.02 for end-to-end probabilities
        assert!(
            max_diff < 0.02,
            "Probability max diff vs Python: {max_diff:.8} (tolerance: 0.02)"
        );
    }

    /// Compare Rust output directly against FireRedVAD's official pip package
    /// (PyTorch) output. This closes the validation chain:
    ///
    /// ```text
    /// FireRedVAD upstream (PyTorch)  ← ref_upstream_probs.json
    ///         ↕ this test
    /// Rust implementation
    /// ```
    ///
    /// The upstream probs are generated by `scripts/firered/validate_upstream.py --save-upstream`
    /// using the same synthetic test signal.
    #[test]
    fn probabilities_match_upstream_fireredvad() {
        let samples_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_samples.json"
        ));
        let samples_data: serde_json::Value = serde_json::from_str(samples_json).unwrap();
        let samples: Vec<i16> = serde_json::from_value(samples_data["samples"].clone()).unwrap();

        let upstream_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/firered_reference/ref_upstream_probs.json"
        ));
        let upstream_data: serde_json::Value = serde_json::from_str(upstream_json).unwrap();
        let upstream_probs: Vec<f64> =
            serde_json::from_value(upstream_data["probs"].clone()).unwrap();

        // Run our full Rust pipeline (same as probabilities_match_python_reference)
        let cmvn = CmvnStats::from_kaldi_binary(CMVN_BYTES).unwrap();
        let mut session = onnx::session_from_memory(MODEL_BYTES).unwrap();
        let mut fbank = FbankExtractor::new();
        let mut caches =
            Array4::<f32>::zeros((CACHE_LAYERS, CACHE_BATCH, CACHE_PROJ, CACHE_LOOKBACK));

        let num_frames = (samples.len() - 400) / 160 + 1;
        assert_eq!(num_frames, upstream_probs.len());

        let mut max_diff: f64 = 0.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * 160;
            let end = start + 400;
            let frame_samples: Vec<f32> = samples[start..end].iter().map(|&s| s as f32).collect();
            let frame_arr: &[f32; 400] = frame_samples.as_slice().try_into().unwrap();

            let mut features = [0.0f32; 80];
            fbank.extract_frame_full(frame_arr, &mut features);
            cmvn.normalize(&mut features);

            let feat_tensor =
                TensorRef::from_array_view(([1i64, 1, 80], &features[..])).unwrap();
            let cache_tensor = TensorRef::from_array_view(caches.view()).unwrap();

            let outputs = session
                .run(inputs![
                    "feat" => feat_tensor,
                    "caches_in" => cache_tensor,
                ])
                .unwrap();

            let probs = outputs.get("probs").unwrap();
            let (_, probs_data): (_, &[f32]) = probs.try_extract_tensor().unwrap();
            let probability = probs_data[0];

            let new_caches = outputs.get("caches_out").unwrap();
            let (_, cache_data): (_, &[f32]) = new_caches.try_extract_tensor().unwrap();
            caches.as_slice_mut().unwrap().copy_from_slice(cache_data);

            let diff = (probability as f64 - upstream_probs[frame_idx]).abs();
            if diff > max_diff {
                max_diff = diff;
            }

            if frame_idx < 5 {
                eprintln!(
                    "  frame {frame_idx}: rust={probability:.6}, upstream={:.6}, diff={diff:.8}",
                    upstream_probs[frame_idx]
                );
            }
        }

        eprintln!("Max probability diff vs upstream FireRedVAD: {max_diff:.8}");

        // Tolerance: 0.02 for Rust vs upstream (PyTorch→ONNX numerical gap + FBank diff)
        assert!(
            max_diff < 0.02,
            "Probability max diff vs upstream: {max_diff:.8} (tolerance: 0.02)"
        );
    }
}
