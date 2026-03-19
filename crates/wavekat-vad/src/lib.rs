//! WaveKat VAD — Unified voice activity detection with multiple backends.
//!
//! This crate provides a common [`VoiceActivityDetector`] trait with
//! implementations for different VAD backends, enabling experimentation
//! and benchmarking across technologies.
//!
//! # Backends
//!
//! - **webrtc** (default) — Google's WebRTC VAD, fast binary detection
//! - **silero** — Neural network based via ONNX Runtime (coming soon)
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

pub mod backends;
pub mod error;
pub mod frame;
pub mod preprocessing;

pub use error::VadError;

/// Common interface for voice activity detection backends.
///
/// Each backend implements this trait, allowing callers to swap
/// implementations without changing their processing logic.
pub trait VoiceActivityDetector: Send {
    /// Process an audio frame and return the probability of speech.
    ///
    /// Returns a value between `0.0` (silence) and `1.0` (speech).
    /// Some backends (e.g. WebRTC) return only binary values (`0.0` or `1.0`),
    /// while others (e.g. Silero) return continuous probabilities.
    ///
    /// # Arguments
    ///
    /// * `samples` — Audio samples as 16-bit signed integers, mono channel.
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
