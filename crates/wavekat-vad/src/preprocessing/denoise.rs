//! RNNoise-based noise suppression wrapper.
//!
//! This module wraps the `nnnoiseless` crate (a pure Rust port of RNNoise)
//! to provide stationary noise suppression for audio streams.
//!
//! # Requirements
//!
//! - Sample rate: 48 kHz (required by RNNoise)
//! - Frame size: 480 samples (10ms at 48kHz)
//!
//! The denoiser handles frame buffering internally, so you can pass
//! any chunk size and it will accumulate/process accordingly.

use nnnoiseless::DenoiseState;

/// Expected sample rate for the denoiser (48 kHz).
pub const DENOISE_SAMPLE_RATE: u32 = 48000;

/// Frame size expected by RNNoise (480 samples = 10ms at 48kHz).
const FRAME_SIZE: usize = 480;

/// RNNoise-based noise suppressor.
///
/// Wraps `nnnoiseless::DenoiseState` with frame buffering to handle
/// arbitrary input chunk sizes.
pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Input buffer accumulating samples until we have FRAME_SIZE
    input_buffer: Vec<f32>,
    /// Output buffer holding processed samples
    output_buffer: Vec<f32>,
    /// Whether this is the first frame (discard due to fade-in artifacts)
    first_frame: bool,
}

impl std::fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Denoiser")
            .field("input_buffer_len", &self.input_buffer.len())
            .field("output_buffer_len", &self.output_buffer.len())
            .field("first_frame", &self.first_frame)
            .finish_non_exhaustive()
    }
}

impl Denoiser {
    /// Create a new denoiser.
    ///
    /// # Panics
    ///
    /// Panics if sample_rate is not 48000 Hz.
    pub fn new(sample_rate: u32) -> Self {
        assert_eq!(
            sample_rate, DENOISE_SAMPLE_RATE,
            "Denoiser requires 48kHz sample rate, got {sample_rate}Hz"
        );

        Self {
            state: DenoiseState::new(),
            input_buffer: Vec::with_capacity(FRAME_SIZE),
            output_buffer: Vec::new(),
            first_frame: true,
        }
    }

    /// Process audio samples through the noise suppressor.
    ///
    /// Input samples are i16 values. Returns denoised samples.
    /// Due to frame buffering, output length may differ from input length.
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        // Convert i16 to f32 and add to input buffer
        for &sample in samples {
            self.input_buffer.push(sample as f32);
        }

        // Process complete frames
        while self.input_buffer.len() >= FRAME_SIZE {
            let mut input_frame = [0.0f32; FRAME_SIZE];
            let mut output_frame = [0.0f32; FRAME_SIZE];

            // Copy frame from input buffer
            input_frame.copy_from_slice(&self.input_buffer[..FRAME_SIZE]);
            self.input_buffer.drain(..FRAME_SIZE);

            // Process through RNNoise
            let _vad_prob = self.state.process_frame(&mut output_frame, &input_frame);

            // Skip first frame due to fade-in artifacts
            if self.first_frame {
                self.first_frame = false;
                // Output zeros for the first frame to maintain timing
                self.output_buffer.extend(std::iter::repeat_n(0.0, FRAME_SIZE));
            } else {
                self.output_buffer.extend_from_slice(&output_frame);
            }
        }

        // Convert output buffer to i16 and return
        let result: Vec<i16> = self
            .output_buffer
            .drain(..)
            .map(|s| s.round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        result
    }

    /// Process a complete buffer of samples (must be multiple of FRAME_SIZE).
    ///
    /// This is more efficient when you know your input is frame-aligned.
    pub fn process_aligned(&mut self, samples: &[i16]) -> Vec<i16> {
        assert!(
            samples.len().is_multiple_of(FRAME_SIZE),
            "Input length {} is not a multiple of frame size {}",
            samples.len(),
            FRAME_SIZE
        );

        let mut output = Vec::with_capacity(samples.len());
        let mut input_frame = [0.0f32; FRAME_SIZE];
        let mut output_frame = [0.0f32; FRAME_SIZE];

        for chunk in samples.chunks_exact(FRAME_SIZE) {
            // Convert to f32
            for (i, &sample) in chunk.iter().enumerate() {
                input_frame[i] = sample as f32;
            }

            // Process
            let _vad_prob = self.state.process_frame(&mut output_frame, &input_frame);

            // Handle first frame
            if self.first_frame {
                self.first_frame = false;
                output.extend(std::iter::repeat_n(0i16, FRAME_SIZE));
            } else {
                // Convert back to i16
                for &s in &output_frame {
                    output.push(s.round().clamp(-32768.0, 32767.0) as i16);
                }
            }
        }

        output
    }

    /// Reset the denoiser state.
    pub fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.first_frame = true;
    }

    /// Returns the number of samples currently buffered.
    pub fn buffered_samples(&self) -> usize {
        self.input_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoiser_creation() {
        let denoiser = Denoiser::new(48000);
        assert_eq!(denoiser.buffered_samples(), 0);
    }

    #[test]
    #[should_panic(expected = "Denoiser requires 48kHz")]
    fn test_denoiser_wrong_sample_rate() {
        Denoiser::new(16000);
    }

    #[test]
    fn test_denoiser_process_single_frame() {
        let mut denoiser = Denoiser::new(48000);

        // Process exactly one frame
        let input: Vec<i16> = vec![0; FRAME_SIZE];
        let output = denoiser.process(&input);

        // First frame outputs zeros (fade-in handling)
        assert_eq!(output.len(), FRAME_SIZE);
    }

    #[test]
    fn test_denoiser_process_multiple_frames() {
        let mut denoiser = Denoiser::new(48000);

        // Process two frames
        let input: Vec<i16> = vec![0; FRAME_SIZE * 2];
        let output = denoiser.process(&input);

        // Should get both frames back
        assert_eq!(output.len(), FRAME_SIZE * 2);
    }

    #[test]
    fn test_denoiser_process_partial_frame() {
        let mut denoiser = Denoiser::new(48000);

        // Process less than one frame
        let input: Vec<i16> = vec![0; 100];
        let output = denoiser.process(&input);

        // No output yet (buffering)
        assert_eq!(output.len(), 0);
        assert_eq!(denoiser.buffered_samples(), 100);

        // Complete the frame
        let input2: Vec<i16> = vec![0; FRAME_SIZE - 100];
        let output2 = denoiser.process(&input2);

        // Now we get output
        assert_eq!(output2.len(), FRAME_SIZE);
        assert_eq!(denoiser.buffered_samples(), 0);
    }

    #[test]
    fn test_denoiser_reset() {
        let mut denoiser = Denoiser::new(48000);

        // Buffer some samples
        let input: Vec<i16> = vec![0; 100];
        denoiser.process(&input);
        assert_eq!(denoiser.buffered_samples(), 100);

        // Reset
        denoiser.reset();
        assert_eq!(denoiser.buffered_samples(), 0);
    }

    #[test]
    fn test_denoiser_aligned() {
        let mut denoiser = Denoiser::new(48000);

        let input: Vec<i16> = vec![0; FRAME_SIZE * 3];
        let output = denoiser.process_aligned(&input);

        assert_eq!(output.len(), FRAME_SIZE * 3);
    }
}
