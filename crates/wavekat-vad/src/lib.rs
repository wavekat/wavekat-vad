//! WaveKat VAD — Unified voice activity detection with multiple backends.
//!
//! This crate provides a common [`VoiceActivityDetector`] trait with
//! implementations for different VAD backends, enabling experimentation
//! and benchmarking across technologies.
//!
//! # Backends
//!
//! | Backend | Feature | Description |
//! |---------|---------|-------------|
//! | WebRTC | `webrtc` (default) | Google's WebRTC VAD — fast, binary output |
//! | Silero | `silero` | Neural network via ONNX Runtime — higher accuracy, continuous probability |
//! | TEN-VAD | `ten-vad` | Agora's TEN-VAD via ONNX — pure Rust, no C dependency |
//!
//! # Feature flags
//!
//! - **`webrtc`** *(default)* — WebRTC VAD backend
//! - **`silero`** — Silero VAD backend (downloads ONNX model at build time)
//! - **`ten-vad`** — TEN-VAD backend (downloads ONNX model at build time)
//! - **`denoise`** — RNNoise-based noise suppression in the preprocessing pipeline
//! - **`serde`** — `Serialize`/`Deserialize` impls for config types
//!
//! # TEN-VAD model license
//!
//! The TEN-VAD ONNX model is licensed under Apache-2.0 with a non-compete clause
//! by the TEN-framework / Agora. It restricts deployment that competes with Agora's
//! offerings. Review the [TEN-VAD license](https://github.com/TEN-framework/ten-vad)
//! before using in production.
//!
//! # Example
//!
//! ```no_run
//! use wavekat_vad::VoiceActivityDetector;
//! use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
//!
//! let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
//! let samples = vec![0i16; 160]; // 10ms at 16kHz
//! let probability = vad.process(&samples, 16000).unwrap();
//! println!("Speech probability: {probability}");
//! ```

pub mod adapter;
pub mod backends;
pub mod error;
pub mod frame;
pub mod preprocessing;

pub use adapter::FrameAdapter;

pub use error::VadError;

/// Describes the audio requirements of a VAD backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VadCapabilities {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Required frame size in samples.
    pub frame_size: usize,
    /// Frame duration in milliseconds (derived from sample_rate and frame_size).
    pub frame_duration_ms: u32,
}

/// Common interface for voice activity detection backends.
///
/// Each backend implements this trait, allowing callers to swap
/// implementations without changing their processing logic.
pub trait VoiceActivityDetector: Send {
    /// Returns the audio requirements of this detector.
    ///
    /// Use this to determine the expected sample rate and frame size
    /// before calling [`process`](Self::process).
    fn capabilities(&self) -> VadCapabilities;

    /// Process an audio frame and return the probability of speech.
    ///
    /// Returns a value between `0.0` (silence) and `1.0` (speech).
    /// Some backends (e.g. WebRTC) return only binary values (`0.0` or `1.0`),
    /// while others (e.g. Silero) return continuous probabilities.
    ///
    /// # Arguments
    ///
    /// * `samples` — Audio samples as 16-bit signed integers, mono channel.
    ///   Must match the `frame_size` from [`capabilities`](Self::capabilities).
    /// * `sample_rate` — Sample rate in Hz (must match the rate the detector was created with).
    ///
    /// # Errors
    ///
    /// Returns [`VadError`] if the sample rate or frame size is invalid,
    /// or if the backend encounters a processing error.
    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError>;

    /// Reset the detector's internal state.
    ///
    /// Call this when starting a new audio stream or after a long pause.
    fn reset(&mut self);
}
