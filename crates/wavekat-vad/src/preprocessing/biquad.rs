//! Biquad filter implementation for audio preprocessing.

use std::f64::consts::PI;

/// Second-order (biquad) IIR filter.
///
/// Implements the standard biquad difference equation:
/// ```text
/// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
/// ```
///
/// Coefficients are normalized (a0 = 1).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    // Normalized coefficients
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State variables (delay line)
    x1: f64, // x[n-1]
    x2: f64, // x[n-2]
    y1: f64, // y[n-1]
    y2: f64, // y[n-2]
}

impl BiquadFilter {
    /// Create a second-order Butterworth high-pass filter.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Panics
    /// Panics if cutoff_hz >= sample_rate / 2 (Nyquist limit).
    pub fn highpass_butterworth(cutoff_hz: f32, sample_rate: u32) -> Self {
        let fs = sample_rate as f64;
        let fc = cutoff_hz as f64;

        assert!(
            fc < fs / 2.0,
            "cutoff frequency must be below Nyquist frequency"
        );

        // Butterworth Q factor for second-order filter
        let q = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.7071

        // Angular frequency
        let omega = 2.0 * PI * fc / fs;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q);

        // High-pass coefficients (before normalization)
        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize by a0
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process a single sample through the filter.
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }

    /// Process a buffer of i16 samples in place.
    pub fn process_i16(&mut self, samples: &mut [i16]) {
        for sample in samples.iter_mut() {
            let x = *sample as f64;
            let y = self.process_sample(x);
            // Clamp to i16 range
            *sample = y.round().clamp(-32768.0, 32767.0) as i16;
        }
    }

    /// Process a buffer of i16 samples, returning a new buffer.
    pub fn process_i16_to_vec(&mut self, samples: &[i16]) -> Vec<i16> {
        samples
            .iter()
            .map(|&s| {
                let y = self.process_sample(s as f64);
                y.round().clamp(-32768.0, 32767.0) as i16
            })
            .collect()
    }

    /// Reset the filter state (clear delay line).
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highpass_creation() {
        let filter = BiquadFilter::highpass_butterworth(80.0, 16000);
        // Just verify it doesn't panic and coefficients are reasonable
        assert!(filter.b0.is_finite());
        assert!(filter.a1.is_finite());
    }

    #[test]
    #[should_panic(expected = "cutoff frequency must be below Nyquist")]
    fn test_highpass_invalid_cutoff() {
        // Cutoff at Nyquist should panic
        BiquadFilter::highpass_butterworth(8000.0, 16000);
    }

    #[test]
    fn test_highpass_attenuates_dc() {
        let mut filter = BiquadFilter::highpass_butterworth(100.0, 16000);

        // DC signal (constant value) should be attenuated to near zero
        let dc_samples: Vec<i16> = vec![10000; 1000];
        let output = filter.process_i16_to_vec(&dc_samples);

        // After settling, output should be near zero
        let last_100: i32 = output[900..].iter().map(|&s| s.abs() as i32).sum();
        let avg = last_100 / 100;
        assert!(
            avg < 100,
            "DC should be heavily attenuated, got avg abs: {avg}"
        );
    }

    #[test]
    fn test_highpass_passes_high_frequencies() {
        let mut filter = BiquadFilter::highpass_butterworth(100.0, 16000);

        // Generate 1kHz sine wave (well above 100Hz cutoff)
        let sample_rate = 16000.0;
        let freq = 1000.0;
        let samples: Vec<i16> = (0..1600)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (10000.0 * (2.0 * PI * freq * t).sin()) as i16
            })
            .collect();

        let output = filter.process_i16_to_vec(&samples);

        // High frequency should pass through with minimal attenuation
        // Check RMS of last portion (after filter settles)
        let input_rms: f64 = (samples[800..]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / 800.0)
            .sqrt();
        let output_rms: f64 = (output[800..]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / 800.0)
            .sqrt();

        // Should retain at least 90% of energy at 1kHz (10x above cutoff)
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.9,
            "1kHz should pass with >90% amplitude, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_highpass_attenuates_low_frequencies() {
        let mut filter = BiquadFilter::highpass_butterworth(200.0, 16000);

        // Generate 50Hz sine wave (well below 200Hz cutoff)
        let sample_rate = 16000.0;
        let freq = 50.0;
        let samples: Vec<i16> = (0..3200) // 200ms for low freq to settle
            .map(|i| {
                let t = i as f64 / sample_rate;
                (10000.0 * (2.0 * PI * freq * t).sin()) as i16
            })
            .collect();

        let output = filter.process_i16_to_vec(&samples);

        // Low frequency should be attenuated
        let input_rms: f64 = (samples[1600..]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / 1600.0)
            .sqrt();
        let output_rms: f64 = (output[1600..]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / 1600.0)
            .sqrt();

        // At 50Hz with 200Hz cutoff (2 octaves below), expect ~12dB attenuation
        // That's about 25% amplitude or less
        let ratio = output_rms / input_rms;
        assert!(
            ratio < 0.35,
            "50Hz should be attenuated to <35% at 200Hz cutoff, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_reset() {
        let mut filter = BiquadFilter::highpass_butterworth(100.0, 16000);

        // Process some samples
        let samples: Vec<i16> = vec![1000; 100];
        filter.process_i16_to_vec(&samples);

        // Reset and verify state is cleared
        filter.reset();
        assert_eq!(filter.x1, 0.0);
        assert_eq!(filter.x2, 0.0);
        assert_eq!(filter.y1, 0.0);
        assert_eq!(filter.y2, 0.0);
    }
}
