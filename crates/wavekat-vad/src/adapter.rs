//! Frame adapter for matching audio frames to VAD backend requirements.
//!
//! Different VAD backends have different frame size requirements. This module
//! provides an adapter that buffers incoming audio and produces frames of the
//! exact size required by each backend.

use crate::{VadCapabilities, VadError, VoiceActivityDetector};

/// Adapts audio frames to match a VAD backend's requirements.
///
/// Buffers incoming samples and produces frames of the exact size
/// required by the wrapped detector. Also handles sample rate validation.
pub struct FrameAdapter {
    /// The wrapped VAD detector.
    inner: Box<dyn VoiceActivityDetector>,
    /// Capabilities of the inner detector.
    capabilities: VadCapabilities,
    /// Buffer for accumulating samples.
    buffer: Vec<i16>,
}

impl FrameAdapter {
    /// Create a new frame adapter wrapping a VAD detector.
    pub fn new(inner: Box<dyn VoiceActivityDetector>) -> Self {
        let capabilities = inner.capabilities();
        Self {
            inner,
            capabilities,
            buffer: Vec::new(),
        }
    }

    /// Returns the capabilities of the wrapped detector.
    pub fn capabilities(&self) -> &VadCapabilities {
        &self.capabilities
    }

    /// Returns the required sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.capabilities.sample_rate
    }

    /// Returns the required frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.capabilities.frame_size
    }

    /// Process audio samples, buffering until a complete frame is available.
    ///
    /// Returns `Some(probability)` when a complete frame was processed,
    /// or `None` if more samples are needed.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (any length)
    /// * `sample_rate` - Sample rate of the input audio
    ///
    /// # Errors
    /// Returns an error if the sample rate doesn't match the detector's requirements.
    pub fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<Option<f32>, VadError> {
        if sample_rate != self.capabilities.sample_rate {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        self.buffer.extend_from_slice(samples);

        if self.buffer.len() >= self.capabilities.frame_size {
            let frame: Vec<i16> = self.buffer.drain(..self.capabilities.frame_size).collect();
            let probability = self.inner.process(&frame, sample_rate)?;
            Ok(Some(probability))
        } else {
            Ok(None)
        }
    }

    /// Process all complete frames in the buffer.
    ///
    /// Returns a vector of probabilities, one for each complete frame processed.
    /// Useful when you want to process multiple frames at once.
    pub fn process_all(&mut self, samples: &[i16], sample_rate: u32) -> Result<Vec<f32>, VadError> {
        if sample_rate != self.capabilities.sample_rate {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        self.buffer.extend_from_slice(samples);

        let mut results = Vec::new();
        while self.buffer.len() >= self.capabilities.frame_size {
            let frame: Vec<i16> = self.buffer.drain(..self.capabilities.frame_size).collect();
            let probability = self.inner.process(&frame, sample_rate)?;
            results.push(probability);
        }

        Ok(results)
    }

    /// Returns the last probability from processing, or 0.0 if no frame was complete.
    ///
    /// This is a convenience method for real-time processing where you only
    /// care about the most recent result.
    pub fn process_latest(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        let results = self.process_all(samples, sample_rate)?;
        Ok(results.into_iter().last().unwrap_or(0.0))
    }

    /// Reset the adapter and the wrapped detector.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inner.reset();
    }

    /// Returns the number of samples currently buffered.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock VAD for testing
    struct MockVad {
        sample_rate: u32,
        frame_size: usize,
        call_count: usize,
    }

    impl MockVad {
        fn new(sample_rate: u32, frame_size: usize) -> Self {
            Self {
                sample_rate,
                frame_size,
                call_count: 0,
            }
        }
    }

    impl VoiceActivityDetector for MockVad {
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities {
                sample_rate: self.sample_rate,
                frame_size: self.frame_size,
                frame_duration_ms: (self.frame_size as u32 * 1000) / self.sample_rate,
            }
        }

        fn process(&mut self, samples: &[i16], _sample_rate: u32) -> Result<f32, VadError> {
            assert_eq!(samples.len(), self.frame_size);
            self.call_count += 1;
            Ok(0.5)
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }
    }

    #[test]
    fn test_adapter_buffers_samples() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send less than a full frame
        let result = adapter.process(&[0i16; 256], 16000).unwrap();
        assert!(result.is_none());
        assert_eq!(adapter.buffered_samples(), 256);

        // Send more to complete the frame
        let result = adapter.process(&[0i16; 256], 16000).unwrap();
        assert!(result.is_some());
        assert_eq!(adapter.buffered_samples(), 0);
    }

    #[test]
    fn test_adapter_handles_multiple_frames() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send two complete frames worth
        let results = adapter.process_all(&[0i16; 1024], 16000).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_adapter_wrong_sample_rate() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        let result = adapter.process(&[0i16; 512], 48000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(48000))));
    }

    #[test]
    fn test_adapter_reset() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Buffer some samples
        let _ = adapter.process(&[0i16; 256], 16000);
        assert_eq!(adapter.buffered_samples(), 256);

        // Reset
        adapter.reset();
        assert_eq!(adapter.buffered_samples(), 0);
    }

    #[test]
    fn test_process_latest() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send multiple frames (1600 = 3 full frames + 64 left over)
        let result = adapter.process_latest(&[0i16; 1600], 16000).unwrap();
        assert_eq!(result, 0.5); // Mock returns 0.5
        assert_eq!(adapter.buffered_samples(), 64); // 1600 - 3*512 = 64 left over
    }
}
