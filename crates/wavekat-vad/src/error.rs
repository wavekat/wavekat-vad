/// Errors that can occur during voice activity detection.
#[derive(Debug, thiserror::Error)]
pub enum VadError {
    /// The provided sample rate is not supported by the backend.
    #[error("unsupported sample rate: {0} Hz")]
    InvalidSampleRate(u32),

    /// The provided frame size does not match the expected size for the backend.
    #[error("invalid frame size: got {got} samples, expected {expected}")]
    InvalidFrameSize {
        /// The number of samples provided.
        got: usize,
        /// The number of samples expected.
        expected: usize,
    },

    /// An error originating from the underlying VAD backend.
    #[error("backend error: {0}")]
    BackendError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = VadError::InvalidSampleRate(44100);
        assert_eq!(err.to_string(), "unsupported sample rate: 44100 Hz");

        let err = VadError::InvalidFrameSize {
            got: 100,
            expected: 160,
        };
        assert_eq!(
            err.to_string(),
            "invalid frame size: got 100 samples, expected 160"
        );

        let err = VadError::BackendError("something went wrong".into());
        assert_eq!(err.to_string(), "backend error: something went wrong");
    }
}
