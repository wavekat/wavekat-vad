//! Silero VAD backend using ONNX Runtime.
//!
//! This backend wraps the [Silero VAD](https://github.com/snakers4/silero-vad)
//! v6 model, a pre-trained LSTM neural network for voice activity detection.
//! It runs inference via ONNX Runtime (through the [`ort`](https://crates.io/crates/ort)
//! crate) and returns continuous speech probability scores between 0.0 and 1.0.
//!
//! # Audio Requirements
//!
//! - **Sample rates:** 8000 or 16000 Hz only
//! - **Frame size:** fixed per sample rate:
//!   - 8 kHz: 256 samples (~32 ms)
//!   - 16 kHz: 512 samples (~32 ms)
//! - **Format:** 16-bit signed integers (i16)
//!
//! # Internal State
//!
//! The model maintains LSTM hidden states and a 64-sample context buffer
//! across calls. This means:
//! - Frames **must** be fed sequentially — skipping or reordering frames
//!   will produce inaccurate results.
//! - Call [`reset()`](crate::VoiceActivityDetector::reset) when starting
//!   a new audio stream or after a gap in input.
//!
//! # Model Loading
//!
//! The default ONNX model (Silero VAD v6) is embedded in the binary at
//! compile time — no external files are needed at runtime. For custom
//! models, use [`SileroVad::from_file`] or [`SileroVad::from_memory`].
//!
//! # Example
//!
//! ```no_run
//! use wavekat_vad::backends::silero::SileroVad;
//! use wavekat_vad::VoiceActivityDetector;
//!
//! let mut vad = SileroVad::new(16000).unwrap();
//! let samples = vec![0i16; 512]; // 32ms at 16kHz
//! let probability = vad.process(&samples, 16000).unwrap();
//! println!("Speech probability: {probability:.3}");
//! ```

use super::onnx;
use crate::error::VadError;
use crate::{ProcessTimings, VadCapabilities, VoiceActivityDetector};
use ndarray::{Array1, Array2, Array3};
use ort::{inputs, session::Session, value::Tensor};
use std::time::{Duration, Instant};

/// Embedded Silero VAD ONNX model (v6).
/// Downloaded automatically at build time by build.rs.
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silero_vad.onnx"));

/// Context size in samples (prepended to each input chunk).
const CONTEXT_SIZE: usize = 64;

/// LSTM hidden state shape: [2, 1, 128] (h and c states).
const STATE_DIM: usize = 128;

/// Voice activity detector backed by the Silero VAD v6 ONNX model.
///
/// Uses an LSTM neural network to produce continuous speech probability
/// scores (0.0–1.0). Internal hidden state and a context buffer persist
/// across calls — see the [module-level docs](self) for details on
/// state management and audio requirements.
pub struct SileroVad {
    /// ONNX Runtime session.
    session: Session,
    /// Sample rate (8000 or 16000).
    sample_rate: u32,
    /// Required chunk size for this sample rate.
    chunk_size: usize,
    /// LSTM hidden state: shape [2, 1, 128].
    state: Array3<f32>,
    /// Context buffer: last 64 samples from previous chunk.
    context: Vec<f32>,
    /// Accumulated time for i16→f32 normalization + context building.
    normalize_time: Duration,
    /// Accumulated time for tensor creation + ONNX run + state update.
    onnx_time: Duration,
    /// Number of frames that produced a result.
    timing_frames: u64,
}

// SAFETY: ort::Session is Send in ort 2.x, and all other fields are owned Send types.
unsafe impl Send for SileroVad {}

impl SileroVad {
    /// Create a new Silero VAD instance using the embedded model.
    ///
    /// The model is automatically downloaded at build time.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (must be 8000 or 16000)
    ///
    /// # Errors
    ///
    /// Returns `VadError::InvalidSampleRate` if the sample rate is not 8000 or 16000.
    /// Returns `VadError::BackendError` if the ONNX session fails to initialize.
    pub fn new(sample_rate: u32) -> Result<Self, VadError> {
        Self::from_memory(MODEL_BYTES, sample_rate)
    }

    /// Create a new Silero VAD instance from a custom ONNX model file.
    ///
    /// Use this to load a different model version or a custom-trained model.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    /// * `sample_rate` - Sample rate in Hz (must be 8000 or 16000)
    ///
    /// # Errors
    ///
    /// Returns `VadError::InvalidSampleRate` if the sample rate is not 8000 or 16000.
    /// Returns `VadError::BackendError` if the model file cannot be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_vad::backends::silero::SileroVad;
    ///
    /// let vad = SileroVad::from_file("path/to/custom_model.onnx", 16000).unwrap();
    /// ```
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        sample_rate: u32,
    ) -> Result<Self, VadError> {
        Self::validate_sample_rate(sample_rate)?;

        let chunk_size = Self::chunk_size_for_rate(sample_rate);
        let session = onnx::session_from_file(path)?;

        let state = Array3::<f32>::zeros((2, 1, STATE_DIM));
        let context = vec![0.0f32; CONTEXT_SIZE];

        Ok(Self {
            session,
            sample_rate,
            chunk_size,
            state,
            context,
            normalize_time: Duration::ZERO,
            onnx_time: Duration::ZERO,
            timing_frames: 0,
        })
    }

    /// Create a new Silero VAD instance from model bytes in memory.
    ///
    /// Use this to load a model from bytes (e.g., from a custom embedding).
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - The ONNX model data
    /// * `sample_rate` - Sample rate in Hz (must be 8000 or 16000)
    pub fn from_memory(model_bytes: &[u8], sample_rate: u32) -> Result<Self, VadError> {
        Self::validate_sample_rate(sample_rate)?;

        let chunk_size = Self::chunk_size_for_rate(sample_rate);
        let session = onnx::session_from_memory(model_bytes)?;

        let state = Array3::<f32>::zeros((2, 1, STATE_DIM));
        let context = vec![0.0f32; CONTEXT_SIZE];

        Ok(Self {
            session,
            sample_rate,
            chunk_size,
            state,
            context,
            normalize_time: Duration::ZERO,
            onnx_time: Duration::ZERO,
            timing_frames: 0,
        })
    }

    fn validate_sample_rate(sample_rate: u32) -> Result<(), VadError> {
        match sample_rate {
            8000 | 16000 => Ok(()),
            _ => Err(VadError::InvalidSampleRate(sample_rate)),
        }
    }

    fn chunk_size_for_rate(sample_rate: u32) -> usize {
        match sample_rate {
            8000 => 256,
            16000 => 512,
            _ => unreachable!("sample rate validated before calling chunk_size_for_rate"),
        }
    }
}

impl VoiceActivityDetector for SileroVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: self.sample_rate,
            frame_size: self.chunk_size,
            frame_duration_ms: (self.chunk_size as u32 * 1000) / self.sample_rate,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // Validate sample rate matches
        if sample_rate != self.sample_rate {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        // Validate frame size
        if samples.len() != self.chunk_size {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: self.chunk_size,
            });
        }

        // --- Preprocessing: normalize + build input ---
        let t_preprocess = Instant::now();

        // Convert i16 samples to f32 and normalize to [-1.0, 1.0]
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // Build input: context + current chunk
        let input_size = CONTEXT_SIZE + self.chunk_size;
        let mut input_data = Vec::with_capacity(input_size);
        input_data.extend_from_slice(&self.context);
        input_data.extend_from_slice(&samples_f32);

        self.normalize_time += t_preprocess.elapsed();

        // --- Inference: tensor creation + ONNX run + state update ---
        let t_inference = Instant::now();

        // Create input tensor: shape [1, context_size + chunk_size]
        let input_array = Array2::from_shape_vec((1, input_size), input_data)
            .map_err(|e| VadError::BackendError(format!("failed to create input array: {e}")))?;
        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| VadError::BackendError(format!("failed to create input tensor: {e}")))?;

        // Create state tensor
        let state_tensor = Tensor::from_array(self.state.clone())
            .map_err(|e| VadError::BackendError(format!("failed to create state tensor: {e}")))?;

        // Create sample rate tensor: shape [1]
        let sr_array = Array1::from_vec(vec![self.sample_rate as i64]);
        let sr_tensor = Tensor::from_array(sr_array)
            .map_err(|e| VadError::BackendError(format!("failed to create sr tensor: {e}")))?;

        // Run inference
        let outputs = self
            .session
            .run(inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(|e| VadError::BackendError(format!("inference failed: {e}")))?;

        // Extract output probability
        let output = outputs
            .get("output")
            .ok_or_else(|| VadError::BackendError("missing 'output' tensor".into()))?;
        let (_, output_data): (_, &[f32]) = output
            .try_extract_tensor()
            .map_err(|e| VadError::BackendError(format!("failed to extract output: {e}")))?;
        let probability = *output_data
            .first()
            .ok_or_else(|| VadError::BackendError("empty output tensor".into()))?;

        // Update hidden state for next call
        let new_state = outputs
            .get("stateN")
            .ok_or_else(|| VadError::BackendError("missing 'stateN' tensor".into()))?;
        let (_, new_state_data): (_, &[f32]) = new_state
            .try_extract_tensor()
            .map_err(|e| VadError::BackendError(format!("failed to extract state: {e}")))?;

        // Copy new state to our buffer (shape is [2, 1, 128] = 256 elements)
        if new_state_data.len() == 2 * STATE_DIM {
            self.state
                .as_slice_mut()
                .ok_or_else(|| VadError::BackendError("state buffer not contiguous".into()))?
                .copy_from_slice(new_state_data);
        } else {
            return Err(VadError::BackendError(format!(
                "unexpected state size: expected {expected}, got {got}",
                expected = 2 * STATE_DIM,
                got = new_state_data.len()
            )));
        }

        // Update context buffer with last CONTEXT_SIZE samples
        let start = samples_f32.len().saturating_sub(CONTEXT_SIZE);
        self.context.copy_from_slice(&samples_f32[start..]);

        self.onnx_time += t_inference.elapsed();
        self.timing_frames += 1;

        // Clamp probability to valid range
        Ok(probability.clamp(0.0, 1.0))
    }

    fn reset(&mut self) {
        // Reset hidden state to zeros
        self.state.fill(0.0);

        // Reset context buffer to zeros
        self.context.fill(0.0);
    }

    fn timings(&self) -> ProcessTimings {
        ProcessTimings {
            stages: vec![("normalize", self.normalize_time), ("onnx", self.onnx_time)],
            frames: self.timing_frames,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_with_valid_rates() {
        let vad_16k = SileroVad::new(16000);
        assert!(vad_16k.is_ok());

        let vad_8k = SileroVad::new(8000);
        assert!(vad_8k.is_ok());
    }

    #[test]
    fn create_with_invalid_rate() {
        // Silero only supports 8kHz and 16kHz
        let vad = SileroVad::new(44100);
        assert!(matches!(vad, Err(VadError::InvalidSampleRate(44100))));

        let vad = SileroVad::new(32000);
        assert!(matches!(vad, Err(VadError::InvalidSampleRate(32000))));

        let vad = SileroVad::new(48000);
        assert!(matches!(vad, Err(VadError::InvalidSampleRate(48000))));
    }

    #[test]
    fn process_silence_16k() {
        let mut vad = SileroVad::new(16000).unwrap();
        let silence = vec![0i16; 512]; // 32ms at 16kHz
        let result = vad.process(&silence, 16000).unwrap();
        // Silence should have low probability
        assert!(
            result < 0.5,
            "Expected low probability for silence, got {result}"
        );
    }

    #[test]
    fn process_silence_8k() {
        let mut vad = SileroVad::new(8000).unwrap();
        let silence = vec![0i16; 256]; // 32ms at 8kHz
        let result = vad.process(&silence, 8000).unwrap();
        assert!(
            result < 0.5,
            "Expected low probability for silence, got {result}"
        );
    }

    #[test]
    fn process_wrong_sample_rate() {
        let mut vad = SileroVad::new(16000).unwrap();
        let samples = vec![0i16; 512];
        let result = vad.process(&samples, 8000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(8000))));
    }

    #[test]
    fn process_invalid_frame_size() {
        let mut vad = SileroVad::new(16000).unwrap();
        let samples = vec![0i16; 100]; // not 512
        let result = vad.process(&samples, 16000);
        assert!(matches!(
            result,
            Err(VadError::InvalidFrameSize {
                got: 100,
                expected: 512
            })
        ));
    }

    #[test]
    fn process_returns_continuous_probability() {
        let mut vad = SileroVad::new(16000).unwrap();
        // Generate some test signal (low amplitude noise)
        let samples: Vec<i16> = (0..512).map(|i| (i % 100) as i16 * 50).collect();
        let result = vad.process(&samples, 16000).unwrap();
        // Result should be in valid range
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut vad = SileroVad::new(16000).unwrap();

        // Process some audio to populate state
        let samples: Vec<i16> = (0..512).map(|i| i as i16 * 10).collect();
        let _ = vad.process(&samples, 16000).unwrap();

        // Reset
        vad.reset();

        // Process silence - should work and give low probability
        let silence = vec![0i16; 512];
        let result = vad.process(&silence, 16000).unwrap();
        assert!(result < 0.5);
    }

    #[test]
    fn state_persists_between_calls() {
        let mut vad = SileroVad::new(16000).unwrap();
        let silence = vec![0i16; 512];

        // Process multiple frames - the model should maintain state
        let prob1 = vad.process(&silence, 16000).unwrap();
        let prob2 = vad.process(&silence, 16000).unwrap();
        let prob3 = vad.process(&silence, 16000).unwrap();

        // All should be low for silence
        assert!(prob1 < 0.5);
        assert!(prob2 < 0.5);
        assert!(prob3 < 0.5);
    }

    #[test]
    fn from_memory_with_embedded_model() {
        let vad = SileroVad::from_memory(MODEL_BYTES, 16000);
        assert!(vad.is_ok(), "from_memory failed: {:?}", vad.err());
    }

    #[test]
    fn from_memory_invalid_bytes() {
        let result = SileroVad::from_memory(b"not a valid onnx model", 16000);
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn from_memory_invalid_sample_rate() {
        // Sample rate validation should happen before ONNX loading
        let result = SileroVad::from_memory(MODEL_BYTES, 44100);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(44100))));
    }

    #[test]
    fn from_file_nonexistent() {
        let result = SileroVad::from_file("/nonexistent/model.onnx", 16000);
        assert!(result.is_err());
        assert!(matches!(result, Err(VadError::BackendError(_))));
    }

    #[test]
    fn from_file_with_temp_model() {
        // Write the embedded model to a temp file, then load it via from_file
        let dir = std::env::temp_dir().join("wavekat_vad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("silero-vad-test.onnx");
        std::fs::write(&model_path, MODEL_BYTES).unwrap();

        let result = SileroVad::from_file(&model_path, 16000);
        assert!(result.is_ok(), "from_file failed: {:?}", result.err());

        // Verify the loaded model works
        let mut vad = result.unwrap();
        let silence = vec![0i16; 512];
        let prob = vad.process(&silence, 16000).unwrap();
        assert!(prob >= 0.0 && prob <= 1.0);

        // Cleanup
        let _ = std::fs::remove_file(&model_path);
    }
}
