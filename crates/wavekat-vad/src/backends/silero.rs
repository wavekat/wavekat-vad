// Silero VAD backend — to be implemented in Step 5.
// This is a stub to keep the module structure in place.

use crate::error::VadError;
use crate::VoiceActivityDetector;

/// Voice activity detector backed by the Silero VAD ONNX model.
pub struct SileroVad {
    _private: (),
}

impl VoiceActivityDetector for SileroVad {
    fn process(&mut self, _samples: &[i16], _sample_rate: u32) -> Result<f32, VadError> {
        Err(VadError::BackendError(
            "silero backend not yet implemented".into(),
        ))
    }

    fn reset(&mut self) {}
}
