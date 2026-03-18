use crate::error::VadError;

/// Supported sample rates for voice activity detection.
pub const SUPPORTED_SAMPLE_RATES: &[u32] = &[8000, 16000, 32000, 48000];

/// Validate that a sample rate is supported.
pub fn validate_sample_rate(sample_rate: u32) -> Result<(), VadError> {
    if SUPPORTED_SAMPLE_RATES.contains(&sample_rate) {
        Ok(())
    } else {
        Err(VadError::InvalidSampleRate(sample_rate))
    }
}

/// Calculate the number of samples for a given frame duration in milliseconds.
pub fn frame_samples(sample_rate: u32, duration_ms: u32) -> usize {
    (sample_rate as usize * duration_ms as usize) / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_sample_rates() {
        for &rate in SUPPORTED_SAMPLE_RATES {
            assert!(validate_sample_rate(rate).is_ok());
        }
    }

    #[test]
    fn invalid_sample_rates() {
        assert!(validate_sample_rate(44100).is_err());
        assert!(validate_sample_rate(22050).is_err());
        assert!(validate_sample_rate(0).is_err());
    }

    #[test]
    fn frame_samples_calculation() {
        // 16kHz, 10ms = 160 samples
        assert_eq!(frame_samples(16000, 10), 160);
        // 16kHz, 20ms = 320 samples
        assert_eq!(frame_samples(16000, 20), 320);
        // 16kHz, 30ms = 480 samples
        assert_eq!(frame_samples(16000, 30), 480);
        // 8kHz, 10ms = 80 samples
        assert_eq!(frame_samples(8000, 10), 80);
        // 48kHz, 30ms = 1440 samples
        assert_eq!(frame_samples(48000, 30), 1440);
    }
}
