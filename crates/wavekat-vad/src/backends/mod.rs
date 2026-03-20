#[cfg(feature = "webrtc")]
pub mod webrtc;

#[cfg(feature = "silero")]
pub mod silero;

#[cfg(feature = "ten-vad")]
pub mod ten_vad;

#[cfg(feature = "ten-vad-onnx")]
pub mod ten_vad_onnx;
