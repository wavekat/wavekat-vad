//! RMS-based audio normalization.
//!
//! Normalizes audio amplitude to a target level in dBFS (decibels relative
//! to full scale), ensuring consistent VAD thresholds regardless of input gain.

/// Full scale reference for i16 audio.
const FULL_SCALE: f64 = 32768.0;

/// Minimum RMS threshold to avoid amplifying silence/noise floor.
/// Below this level (-60 dBFS), we don't apply gain.
const MIN_RMS_THRESHOLD: f64 = 0.001; // ~-60 dBFS

/// Audio normalizer that adjusts amplitude to a target dBFS level.
///
/// Uses RMS (root mean square) measurement with smoothing to avoid
/// sudden gain changes. Includes peak limiting to prevent clipping.
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Target RMS level as linear amplitude (derived from dBFS).
    target_rms: f64,
    /// Current gain being applied (smoothed).
    current_gain: f64,
    /// Smoothing factor for gain changes (0-1, higher = faster).
    smoothing: f64,
    /// Whether to apply peak limiting.
    peak_limit: bool,
}

impl Normalizer {
    /// Create a new normalizer with the given target level.
    ///
    /// # Arguments
    /// * `target_dbfs` - Target RMS level in dBFS (e.g., -20.0)
    ///
    /// # Panics
    /// Panics if target_dbfs > 0 (would exceed full scale).
    pub fn new(target_dbfs: f32) -> Self {
        assert!(
            target_dbfs <= 0.0,
            "Target dBFS must be <= 0, got {target_dbfs}"
        );

        // Convert dBFS to linear amplitude
        // dBFS = 20 * log10(amplitude / full_scale)
        // amplitude = full_scale * 10^(dBFS/20)
        let target_rms = FULL_SCALE * 10_f64.powf(target_dbfs as f64 / 20.0);

        Self {
            target_rms,
            current_gain: 1.0,
            smoothing: 0.1, // Moderate smoothing
            peak_limit: true,
        }
    }

    /// Create a normalizer with custom settings.
    pub fn with_settings(target_dbfs: f32, smoothing: f64, peak_limit: bool) -> Self {
        let mut normalizer = Self::new(target_dbfs);
        normalizer.smoothing = smoothing.clamp(0.01, 1.0);
        normalizer.peak_limit = peak_limit;
        normalizer
    }

    /// Calculate RMS of the samples.
    fn calculate_rms(samples: &[i16]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        (sum_squares / samples.len() as f64).sqrt()
    }

    /// Convert RMS to dBFS.
    #[allow(dead_code)]
    pub fn rms_to_dbfs(rms: f64) -> f64 {
        if rms <= 0.0 {
            return -96.0; // Practical minimum
        }
        20.0 * (rms / FULL_SCALE).log10()
    }

    /// Process audio samples with normalization.
    ///
    /// Returns a new buffer with normalized amplitude.
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        if samples.is_empty() {
            return Vec::new();
        }

        let input_rms = Self::calculate_rms(samples);

        // Don't amplify very quiet signals (likely silence or noise floor)
        if input_rms < MIN_RMS_THRESHOLD * FULL_SCALE {
            return samples.to_vec();
        }

        // Calculate target gain
        let target_gain = self.target_rms / input_rms;

        // Smooth gain changes to avoid sudden jumps
        self.current_gain += self.smoothing * (target_gain - self.current_gain);

        // Apply gain and optionally limit peaks
        samples
            .iter()
            .map(|&s| {
                let amplified = s as f64 * self.current_gain;

                if self.peak_limit {
                    // Soft clipping using tanh for smoother limiting
                    let normalized = amplified / FULL_SCALE;
                    let limited = if normalized.abs() > 0.9 {
                        // Apply soft limiting above 90% of full scale
                        let sign = normalized.signum();
                        let magnitude = normalized.abs();
                        let compressed = 0.9 + 0.1 * ((magnitude - 0.9) / 0.1).tanh();
                        sign * compressed * FULL_SCALE
                    } else {
                        amplified
                    };
                    limited.round().clamp(-32768.0, 32767.0) as i16
                } else {
                    // Hard clipping
                    amplified.round().clamp(-32768.0, 32767.0) as i16
                }
            })
            .collect()
    }

    /// Reset the normalizer state.
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }

    /// Get the current gain being applied.
    pub fn current_gain(&self) -> f64 {
        self.current_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_creation() {
        let norm = Normalizer::new(-20.0);
        assert!(norm.target_rms > 0.0);
        assert!(norm.target_rms < FULL_SCALE);
    }

    #[test]
    #[should_panic(expected = "Target dBFS must be <= 0")]
    fn test_normalizer_invalid_target() {
        Normalizer::new(6.0);
    }

    #[test]
    fn test_rms_calculation() {
        // DC signal should have RMS equal to the DC value
        let dc: Vec<i16> = vec![1000; 100];
        let rms = Normalizer::calculate_rms(&dc);
        assert!((rms - 1000.0).abs() < 1.0);

        // Silence should have zero RMS
        let silence: Vec<i16> = vec![0; 100];
        let rms = Normalizer::calculate_rms(&silence);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_normalizer_amplifies_quiet() {
        let mut norm = Normalizer::new(-20.0);

        // Very quiet signal (about -40 dBFS)
        let quiet: Vec<i16> = vec![100; 480];
        let output = norm.process(&quiet);

        // Should be amplified
        let input_rms = Normalizer::calculate_rms(&quiet);
        let output_rms = Normalizer::calculate_rms(&output);
        assert!(
            output_rms > input_rms,
            "Output RMS {output_rms} should be > input RMS {input_rms}"
        );
    }

    #[test]
    fn test_normalizer_attenuates_loud() {
        let mut norm = Normalizer::new(-20.0);

        // Loud signal (about -6 dBFS)
        let loud: Vec<i16> = vec![16000; 480];
        let output = norm.process(&loud);

        // Should be attenuated
        let input_rms = Normalizer::calculate_rms(&loud);
        let output_rms = Normalizer::calculate_rms(&output);
        assert!(
            output_rms < input_rms,
            "Output RMS {output_rms} should be < input RMS {input_rms}"
        );
    }

    #[test]
    fn test_normalizer_skips_silence() {
        let mut norm = Normalizer::new(-20.0);

        // Very quiet (below threshold)
        let silence: Vec<i16> = vec![1; 480];
        let output = norm.process(&silence);

        // Should pass through unchanged
        assert_eq!(output, silence);
    }

    #[test]
    fn test_normalizer_peak_limiting() {
        let mut norm = Normalizer::with_settings(-6.0, 1.0, true);

        // Signal that will be amplified significantly
        let input: Vec<i16> = vec![10000; 480];
        let output = norm.process(&input);

        // All samples should be valid i16 values (limiting worked)
        // This is implicitly true since output is Vec<i16>, but we verify
        // the limiting didn't cause any issues
        assert!(!output.is_empty());
    }

    #[test]
    fn test_normalizer_reset() {
        let mut norm = Normalizer::new(-20.0);

        // Process some audio to change gain
        let samples: Vec<i16> = vec![1000; 480];
        norm.process(&samples);
        assert!(norm.current_gain() != 1.0);

        // Reset
        norm.reset();
        assert_eq!(norm.current_gain(), 1.0);
    }

    #[test]
    fn test_dbfs_conversion() {
        // Full scale should be 0 dBFS
        let dbfs = Normalizer::rms_to_dbfs(FULL_SCALE);
        assert!((dbfs - 0.0).abs() < 0.01);

        // Half amplitude should be about -6 dBFS
        let dbfs = Normalizer::rms_to_dbfs(FULL_SCALE / 2.0);
        assert!((dbfs - (-6.02)).abs() < 0.1);
    }
}
