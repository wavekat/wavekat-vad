//! WaveKat VAD — Unified voice activity detection with multiple backends.
//!
//! This crate provides a common [`VoiceActivityDetector`] trait with
//! implementations for different VAD backends, enabling experimentation
//! and benchmarking across technologies.
//!
//! # Backends
//!
//! | Backend | Feature | Sample Rates | Frame Size | Output |
//! |---------|---------|-------------|------------|--------|
//! | [WebRTC](`backends::webrtc`) | `webrtc` (default) | 8/16/32/48 kHz | 10, 20, or 30ms | Binary (0.0 or 1.0) |
//! | [Silero](`backends::silero`) | `silero` | 8/16 kHz | 32ms | Continuous (0.0–1.0) |
//! | [TEN-VAD](`backends::ten_vad`) | `ten-vad` | 16 kHz only | 16ms | Continuous (0.0–1.0) |
//! | [FireRedVAD](`backends::firered`) | `firered` | 16 kHz only | 10ms | Continuous (0.0–1.0) |
//!
//! # Quick start
//!
//! Add the crate with the backend you need:
//!
//! ```toml
//! [dependencies]
//! wavekat-vad = "0.1"                                  # WebRTC only (default)
//! wavekat-vad = { version = "0.1", features = ["silero"] }  # Silero
//! wavekat-vad = { version = "0.1", features = ["ten-vad"] } # TEN-VAD
//! wavekat-vad = { version = "0.1", features = ["firered"] } # FireRedVAD
//! ```
//!
//! Then create a detector and process audio frames:
//!
//! ```no_run
//! # #[cfg(feature = "webrtc")]
//! # {
//! use wavekat_vad::VoiceActivityDetector;
//! use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
//!
//! let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
//! let samples = vec![0i16; 480]; // 30ms at 16kHz
//! let probability = vad.process(&samples, 16000).unwrap();
//! println!("Speech probability: {probability}");
//! # }
//! ```
//!
//! # Writing backend-generic code
//!
//! All backends implement [`VoiceActivityDetector`], so you can write code
//! that works with any backend:
//!
//! ```no_run
//! use wavekat_vad::VoiceActivityDetector;
//!
//! fn detect_speech(vad: &mut dyn VoiceActivityDetector, audio: &[i16], sample_rate: u32) {
//!     let caps = vad.capabilities();
//!     for frame in audio.chunks_exact(caps.frame_size) {
//!         let prob = vad.process(frame, sample_rate).unwrap();
//!         if prob > 0.5 {
//!             println!("Speech detected!");
//!         }
//!     }
//! }
//! ```
//!
//! # Handling arbitrary chunk sizes
//!
//! Real-world audio often arrives in chunks that don't match the backend's
//! required frame size. Use [`FrameAdapter`] to buffer and split automatically:
//!
//! ```no_run
//! # #[cfg(feature = "webrtc")]
//! # {
//! use wavekat_vad::FrameAdapter;
//! use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
//!
//! let vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
//! let mut adapter = FrameAdapter::new(Box::new(vad));
//!
//! let chunk = vec![0i16; 1000]; // arbitrary size
//! let results = adapter.process_all(&chunk, 16000).unwrap();
//! for prob in &results {
//!     println!("{prob:.3}");
//! }
//! # }
//! ```
//!
//! # Audio preprocessing
//!
//! Optional preprocessing stages can improve accuracy with noisy input.
//! See the [`preprocessing`] module for details.
//!
//! ```
//! use wavekat_vad::preprocessing::{Preprocessor, PreprocessorConfig};
//!
//! let config = PreprocessorConfig::raw_mic(); // 80Hz HP + normalization
//! let mut preprocessor = Preprocessor::new(&config, 16000);
//! let raw: Vec<i16> = vec![0; 512];
//! let cleaned = preprocessor.process(&raw);
//! // feed `cleaned` to your VAD
//! ```
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `webrtc` | Yes | WebRTC VAD backend |
//! | `silero` | No | Silero VAD backend (ONNX model downloaded at build time) |
//! | `ten-vad` | No | TEN-VAD backend (ONNX model downloaded at build time) |
//! | `firered` | No | FireRedVAD backend (ONNX model + CMVN downloaded at build time) |
//! | `denoise` | No | RNNoise-based noise suppression in [`preprocessing`] |
//! | `serde` | No | `Serialize`/`Deserialize` for config types |
//!
//! ## ONNX model downloads
//!
//! The Silero, TEN-VAD, and FireRedVAD backends download their ONNX models
//! automatically at build time. For offline or CI builds, set environment
//! variables to point to local model files:
//!
//! ```sh
//! SILERO_MODEL_PATH=/path/to/silero_vad.onnx cargo build --features silero
//! TEN_VAD_MODEL_PATH=/path/to/ten-vad.onnx cargo build --features ten-vad
//! FIRERED_MODEL_PATH=/path/to/model.onnx FIRERED_CMVN_PATH=/path/to/cmvn.ark cargo build --features firered
//! ```
//!
//! # Error handling
//!
//! All backends return [`Result<f32, VadError>`]. Check a backend's
//! requirements with [`VoiceActivityDetector::capabilities()`] before processing:
//!
//! - [`VadError::InvalidSampleRate`] — unsupported sample rate
//! - [`VadError::InvalidFrameSize`] — wrong number of samples
//! - [`VadError::BackendError`] — backend-specific error (e.g. ONNX failure)
//!
//! # Examples
//!
//! Runnable examples are in the
//! [`examples/`](https://github.com/wavekat/wavekat-vad/tree/main/crates/wavekat-vad/examples)
//! directory:
//!
//! - **[`detect_speech`](https://github.com/wavekat/wavekat-vad/blob/main/crates/wavekat-vad/examples/detect_speech.rs)** —
//!   Detect speech in a WAV file using any backend
//! - **[`ten_vad_file`](https://github.com/wavekat/wavekat-vad/blob/main/crates/wavekat-vad/examples/ten_vad_file.rs)** —
//!   Process a WAV file with TEN-VAD directly
//!
//! ```sh
//! cargo run --example detect_speech -- audio.wav
//! cargo run --example detect_speech --features silero -- -b silero audio.wav
//! cargo run --example ten_vad_file --features ten-vad -- audio.wav
//! ```
//!
//! # TEN-VAD model license
//!
//! The TEN-VAD ONNX model is licensed under Apache-2.0 with a non-compete clause
//! by the TEN-framework / Agora. It restricts deployment that competes with Agora's
//! offerings. Review the [TEN-VAD license](https://github.com/TEN-framework/ten-vad)
//! before using in production.

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
