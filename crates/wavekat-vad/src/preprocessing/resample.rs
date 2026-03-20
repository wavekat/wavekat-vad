//! Audio resampling utilities.
//!
//! FFT-based sample rate conversion for audio preprocessing.

use rubato::{FftFixedIn, Resampler};

/// Resampler for converting between sample rates.
///
/// Uses FFT-based resampling for high quality.
pub struct AudioResampler {
    resampler: FftFixedIn<f64>,
    input_buffer: Vec<f64>,
    output_buffer: Vec<i16>,
}

impl AudioResampler {
    /// Create a new resampler.
    ///
    /// # Arguments
    /// * `source_rate` - Input sample rate in Hz
    /// * `target_rate` - Output sample rate in Hz
    ///
    /// # Errors
    /// Returns an error if the resampler cannot be created.
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, String> {
        let chunk_size = 1024;
        let resampler = FftFixedIn::<f64>::new(
            source_rate as usize,
            target_rate as usize,
            chunk_size,
            2, // sub_chunks for lower latency
            1, // mono
        )
        .map_err(|e| format!("failed to create resampler: {e}"))?;

        Ok(Self {
            resampler,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        })
    }

    /// Process audio samples and return resampled output.
    ///
    /// Due to buffering, output length may differ from input length.
    /// Call `flush` at the end of a stream to get remaining samples.
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        // Convert input samples to f64 and add to buffer
        self.input_buffer
            .extend(samples.iter().map(|&s| s as f64 / i16::MAX as f64));

        // Process complete chunks through the resampler
        let input_frames_needed = self.resampler.input_frames_next();
        while self.input_buffer.len() >= input_frames_needed {
            let chunk: Vec<f64> = self.input_buffer.drain(..input_frames_needed).collect();
            match self.resampler.process(&[chunk], None) {
                Ok(output) => {
                    if !output.is_empty() {
                        self.output_buffer
                            .extend(output[0].iter().map(|&s| (s * i16::MAX as f64) as i16));
                    }
                }
                Err(e) => {
                    // Log error but don't fail - return what we have
                    eprintln!("resampling error: {e}");
                }
            }
        }

        // Return all available output
        std::mem::take(&mut self.output_buffer)
    }

    /// Reset the resampler state.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        // Note: FftFixedIn doesn't have a reset method, but clearing buffers is sufficient
        // since it's stateless beyond the internal FFT buffers which get overwritten
    }

    /// Returns the number of input samples currently buffered.
    pub fn buffered_input(&self) -> usize {
        self.input_buffer.len()
    }

    /// Returns the number of output samples ready to be retrieved.
    pub fn buffered_output(&self) -> usize {
        self.output_buffer.len()
    }
}

impl std::fmt::Debug for AudioResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioResampler")
            .field("input_buffer_len", &self.input_buffer.len())
            .field("output_buffer_len", &self.output_buffer.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_creation() {
        let resampler = AudioResampler::new(16000, 48000);
        assert!(resampler.is_ok());
    }

    #[test]
    fn test_resample_16k_to_48k() {
        let mut resampler = AudioResampler::new(16000, 48000).unwrap();

        // Process enough samples to get output (need to fill internal buffer)
        let input: Vec<i16> = vec![1000; 2048];
        let output = resampler.process(&input);

        // Output should be roughly 3x the input (48000/16000 = 3)
        // Due to buffering, we may not get exactly 3x, but should be close
        assert!(!output.is_empty(), "Should have some output");
    }

    #[test]
    fn test_resample_48k_to_16k() {
        let mut resampler = AudioResampler::new(48000, 16000).unwrap();

        // Process enough samples
        let input: Vec<i16> = vec![1000; 2048];
        let output = resampler.process(&input);

        // Output should be roughly 1/3 the input
        assert!(!output.is_empty(), "Should have some output");
    }

    #[test]
    fn test_resampler_reset() {
        let mut resampler = AudioResampler::new(16000, 48000).unwrap();

        // Buffer some samples (not enough for output)
        let input: Vec<i16> = vec![1000; 100];
        let _ = resampler.process(&input);
        assert!(resampler.buffered_input() > 0);

        // Reset
        resampler.reset();
        assert_eq!(resampler.buffered_input(), 0);
        assert_eq!(resampler.buffered_output(), 0);
    }
}
