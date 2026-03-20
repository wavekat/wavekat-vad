//! VAD backend implementations.
//!
//! Each backend is gated behind a feature flag. Enable the feature to make the
//! module available:
//!
//! | Module | Feature | Backend |
//! |--------|---------|---------|
//! | [`webrtc`] | `webrtc` (default) | Google's WebRTC VAD |
//! | [`silero`] | `silero` | Silero VAD v5 (ONNX) |
//! | [`ten_vad`] | `ten-vad` | Agora's TEN-VAD (ONNX) |
//!
//! All backends implement the [`VoiceActivityDetector`](crate::VoiceActivityDetector) trait.

#[cfg(any(feature = "silero", feature = "ten-vad"))]
pub(crate) mod onnx;

#[cfg(feature = "webrtc")]
pub mod webrtc;

#[cfg(feature = "silero")]
pub mod silero;

#[cfg(feature = "ten-vad")]
pub mod ten_vad;
