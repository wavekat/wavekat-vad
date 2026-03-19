use crate::error::VadError;
use crate::frame::{frame_samples, validate_sample_rate};
use crate::{VadCapabilities, VoiceActivityDetector};

/// WebRTC VAD aggressiveness mode.
///
/// Higher modes are more aggressive at filtering out non-speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebRtcVadMode {
    /// Quality mode — least aggressive, fewest false negatives.
    Quality,
    /// Low bitrate mode.
    LowBitrate,
    /// Aggressive mode.
    Aggressive,
    /// Very aggressive mode — most aggressive, fewest false positives.
    VeryAggressive,
}

impl From<WebRtcVadMode> for webrtc_vad::VadMode {
    fn from(mode: WebRtcVadMode) -> Self {
        match mode {
            WebRtcVadMode::Quality => webrtc_vad::VadMode::Quality,
            WebRtcVadMode::LowBitrate => webrtc_vad::VadMode::LowBitrate,
            WebRtcVadMode::Aggressive => webrtc_vad::VadMode::Aggressive,
            WebRtcVadMode::VeryAggressive => webrtc_vad::VadMode::VeryAggressive,
        }
    }
}

/// Default frame duration for WebRTC VAD (30ms gives best results).
const DEFAULT_FRAME_DURATION_MS: u32 = 30;

/// Voice activity detector backed by Google's WebRTC VAD.
///
/// Supports sample rates of 8000, 16000, 32000, and 48000 Hz.
/// Frame duration must be 10, 20, or 30 milliseconds.
///
/// Returns binary detection results: `1.0` for speech, `0.0` for silence.
pub struct WebRtcVad {
    vad: webrtc_vad::Vad,
    sample_rate: u32,
    mode: WebRtcVadMode,
    frame_duration_ms: u32,
}

// SAFETY: webrtc_vad::Vad wraps a C pointer that is only accessed via &mut self.
// We never share the pointer across threads simultaneously.
unsafe impl Send for WebRtcVad {}

impl WebRtcVad {
    /// Create a new WebRTC VAD instance with default 30ms frame duration.
    ///
    /// # Errors
    ///
    /// Returns `VadError::InvalidSampleRate` if the sample rate is not supported.
    pub fn new(sample_rate: u32, mode: WebRtcVadMode) -> Result<Self, VadError> {
        Self::with_frame_duration(sample_rate, mode, DEFAULT_FRAME_DURATION_MS)
    }

    /// Create a new WebRTC VAD instance with custom frame duration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (8000, 16000, 32000, or 48000)
    /// * `mode` - Aggressiveness mode
    /// * `frame_duration_ms` - Frame duration in ms (10, 20, or 30)
    ///
    /// # Errors
    ///
    /// Returns `VadError::InvalidSampleRate` if the sample rate is not supported.
    /// Returns `VadError::InvalidFrameSize` if the frame duration is not 10, 20, or 30ms.
    pub fn with_frame_duration(
        sample_rate: u32,
        mode: WebRtcVadMode,
        frame_duration_ms: u32,
    ) -> Result<Self, VadError> {
        validate_sample_rate(sample_rate)?;

        if !matches!(frame_duration_ms, 10 | 20 | 30) {
            return Err(VadError::InvalidFrameSize {
                got: frame_samples(sample_rate, frame_duration_ms),
                expected: frame_samples(sample_rate, 30),
            });
        }

        let mut vad = webrtc_vad::Vad::new_with_rate(to_sample_rate(sample_rate));
        vad.set_mode(mode.into());
        Ok(Self {
            vad,
            sample_rate,
            mode,
            frame_duration_ms,
        })
    }
}

impl VoiceActivityDetector for WebRtcVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: self.sample_rate,
            frame_size: frame_samples(self.sample_rate, self.frame_duration_ms),
            frame_duration_ms: self.frame_duration_ms,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        if sample_rate != self.sample_rate {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        // webrtc-vad requires frames of exactly 10, 20, or 30ms
        let valid_frame_sizes = [
            frame_samples(sample_rate, 10),
            frame_samples(sample_rate, 20),
            frame_samples(sample_rate, 30),
        ];

        if !valid_frame_sizes.contains(&samples.len()) {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: valid_frame_sizes[0], // suggest 10ms
            });
        }

        let is_voice = self
            .vad
            .is_voice_segment(samples)
            .map_err(|()| VadError::BackendError("webrtc-vad processing error".into()))?;

        Ok(if is_voice { 1.0 } else { 0.0 })
    }

    fn reset(&mut self) {
        let mut vad = webrtc_vad::Vad::new_with_rate(to_sample_rate(self.sample_rate));
        vad.set_mode(self.mode.into());
        self.vad = vad;
    }
}

fn to_sample_rate(rate: u32) -> webrtc_vad::SampleRate {
    match rate {
        8000 => webrtc_vad::SampleRate::Rate8kHz,
        16000 => webrtc_vad::SampleRate::Rate16kHz,
        32000 => webrtc_vad::SampleRate::Rate32kHz,
        48000 => webrtc_vad::SampleRate::Rate48kHz,
        _ => unreachable!("sample rate validated before calling to_sample_rate"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_with_valid_rates() {
        for &rate in &[8000, 16000, 32000, 48000] {
            let vad = WebRtcVad::new(rate, WebRtcVadMode::Quality);
            assert!(vad.is_ok());
        }
    }

    #[test]
    fn create_with_invalid_rate() {
        let vad = WebRtcVad::new(44100, WebRtcVadMode::Quality);
        assert!(vad.is_err());
    }

    #[test]
    fn process_silence() {
        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
        // 10ms of silence at 16kHz = 160 samples
        let silence = vec![0i16; 160];
        let result = vad.process(&silence, 16000).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn process_wrong_sample_rate() {
        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
        let samples = vec![0i16; 160];
        let result = vad.process(&samples, 8000);
        assert!(result.is_err());
    }

    #[test]
    fn process_invalid_frame_size() {
        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
        let samples = vec![0i16; 100]; // not 160, 320, or 480
        let result = vad.process(&samples, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn reset_works() {
        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Aggressive).unwrap();
        let silence = vec![0i16; 160];
        let _ = vad.process(&silence, 16000).unwrap();
        vad.reset();
        // Should still work after reset
        let result = vad.process(&silence, 16000).unwrap();
        assert_eq!(result, 0.0);
    }
}
