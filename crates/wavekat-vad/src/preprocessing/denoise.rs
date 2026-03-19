//! RNNoise-based noise suppression wrapper.
//!
//! This module wraps the `nnnoiseless` crate (a pure Rust port of RNNoise)
//! to provide stationary noise suppression for audio streams.
//!
//! # Sample Rate Handling
//!
//! RNNoise internally requires 48kHz audio. This module handles resampling
//! automatically:
//! - At 48kHz: audio is processed directly (most efficient)
//! - At other rates (e.g., 16kHz): audio is upsampled to 48kHz, processed,
//!   then downsampled back to the original rate
//!
//! # Frame Size
//!
//! RNNoise processes 480 samples (10ms at 48kHz) at a time.
//! The denoiser handles frame buffering internally, so you can pass
//! any chunk size and it will accumulate/process accordingly.

use super::resample::AudioResampler;
use nnnoiseless::DenoiseState;

/// Internal sample rate required by RNNoise (48 kHz).
pub const DENOISE_SAMPLE_RATE: u32 = 48000;

/// Frame size expected by RNNoise (480 samples = 10ms at 48kHz).
const FRAME_SIZE: usize = 480;

/// RNNoise-based noise suppressor.
///
/// Wraps `nnnoiseless::DenoiseState` with frame buffering to handle
/// arbitrary input chunk sizes. Automatically resamples to/from 48kHz
/// when the input sample rate differs.
pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Input sample rate.
    sample_rate: u32,
    /// Upsampler: input rate → 48kHz (None if already 48kHz).
    upsampler: Option<AudioResampler>,
    /// Downsampler: 48kHz → input rate (None if already 48kHz).
    downsampler: Option<AudioResampler>,
    /// Input buffer accumulating samples until we have FRAME_SIZE (at 48kHz).
    input_buffer: Vec<f32>,
    /// Output buffer holding processed samples (at 48kHz).
    output_buffer: Vec<f32>,
    /// Whether this is the first frame (discard due to fade-in artifacts).
    first_frame: bool,
}

impl std::fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Denoiser")
            .field("sample_rate", &self.sample_rate)
            .field("resampling", &self.upsampler.is_some())
            .field("input_buffer_len", &self.input_buffer.len())
            .field("output_buffer_len", &self.output_buffer.len())
            .field("first_frame", &self.first_frame)
            .finish_non_exhaustive()
    }
}

impl Denoiser {
    /// Create a new denoiser.
    ///
    /// # Arguments
    /// * `sample_rate` - Input sample rate in Hz. Common values: 16000, 48000.
    ///
    /// If the sample rate is not 48kHz, the denoiser will automatically
    /// resample to 48kHz for processing and back to the original rate.
    ///
    /// # Panics
    ///
    /// Panics if resamplers cannot be created (should not happen with valid rates).
    pub fn new(sample_rate: u32) -> Self {
        let (upsampler, downsampler) = if sample_rate == DENOISE_SAMPLE_RATE {
            (None, None)
        } else {
            let up = AudioResampler::new(sample_rate, DENOISE_SAMPLE_RATE)
                .expect("failed to create upsampler");
            let down = AudioResampler::new(DENOISE_SAMPLE_RATE, sample_rate)
                .expect("failed to create downsampler");
            (Some(up), Some(down))
        };

        Self {
            state: DenoiseState::new(),
            sample_rate,
            upsampler,
            downsampler,
            input_buffer: Vec::with_capacity(FRAME_SIZE),
            output_buffer: Vec::new(),
            first_frame: true,
        }
    }

    /// Returns the sample rate this denoiser was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns true if resampling is being used.
    pub fn is_resampling(&self) -> bool {
        self.upsampler.is_some()
    }

    /// Process audio samples through the noise suppressor.
    ///
    /// Input samples are i16 values at the configured sample rate.
    /// Returns denoised samples at the same sample rate.
    /// Due to frame buffering (and resampling if applicable), output length
    /// may differ from input length.
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        // Step 1: Upsample if needed (e.g., 16kHz → 48kHz)
        let samples_48k: Vec<i16> = if let Some(ref mut upsampler) = self.upsampler {
            upsampler.process(samples)
        } else {
            samples.to_vec()
        };

        // Step 2: Process through RNNoise at 48kHz
        // Convert i16 to f32 and add to input buffer
        for &sample in &samples_48k {
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
                self.output_buffer
                    .extend(std::iter::repeat_n(0.0, FRAME_SIZE));
            } else {
                self.output_buffer.extend_from_slice(&output_frame);
            }
        }

        // Convert output buffer to i16
        let denoised_48k: Vec<i16> = self
            .output_buffer
            .drain(..)
            .map(|s| s.round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        // Step 3: Downsample if needed (e.g., 48kHz → 16kHz)
        if let Some(ref mut downsampler) = self.downsampler {
            downsampler.process(&denoised_48k)
        } else {
            denoised_48k
        }
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
        if let Some(ref mut upsampler) = self.upsampler {
            upsampler.reset();
        }
        if let Some(ref mut downsampler) = self.downsampler {
            downsampler.reset();
        }
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
    fn test_denoiser_creation_48k() {
        let denoiser = Denoiser::new(48000);
        assert_eq!(denoiser.buffered_samples(), 0);
        assert_eq!(denoiser.sample_rate(), 48000);
        assert!(!denoiser.is_resampling());
    }

    #[test]
    fn test_denoiser_creation_16k() {
        let denoiser = Denoiser::new(16000);
        assert_eq!(denoiser.buffered_samples(), 0);
        assert_eq!(denoiser.sample_rate(), 16000);
        assert!(denoiser.is_resampling());
    }

    #[test]
    fn test_denoiser_process_single_frame_48k() {
        let mut denoiser = Denoiser::new(48000);

        // Process exactly one frame
        let input: Vec<i16> = vec![0; FRAME_SIZE];
        let output = denoiser.process(&input);

        // First frame outputs zeros (fade-in handling)
        assert_eq!(output.len(), FRAME_SIZE);
    }

    #[test]
    fn test_denoiser_process_multiple_frames_48k() {
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

    #[test]
    fn test_denoiser_16k_produces_output() {
        let mut denoiser = Denoiser::new(16000);

        // Process enough samples to get output (need extra for resampling buffers)
        // At 16kHz, we need ~160 samples for 10ms, but resampling buffers need more
        let input: Vec<i16> = vec![0; 2048];
        let output = denoiser.process(&input);

        // Should produce some output (exact amount depends on resampler buffering)
        // Due to multiple buffering stages, first call may not produce full output
        assert!(
            output.len() > 0 || denoiser.buffered_samples() > 0,
            "Should either produce output or buffer samples"
        );
    }

    #[test]
    fn test_denoiser_16k_continuous_processing() {
        let mut denoiser = Denoiser::new(16000);

        // Process several chunks to fill all buffers
        let chunk: Vec<i16> = vec![0; 320]; // 20ms at 16kHz
        let mut total_output = 0;

        for _ in 0..20 {
            let output = denoiser.process(&chunk);
            total_output += output.len();
        }

        // After 400ms of input, we should have substantial output
        assert!(
            total_output > 5000,
            "Expected significant output, got {total_output}"
        );
    }
}
