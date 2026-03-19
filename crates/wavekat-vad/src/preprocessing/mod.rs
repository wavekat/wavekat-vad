//! Audio preprocessing pipeline for improving VAD accuracy.
//!
//! This module provides configurable preprocessing stages that clean audio
//! before voice activity detection. Each stage is optional and can be
//! enabled/disabled via [`PreprocessorConfig`].
//!
//! # Example
//!
//! ```
//! use wavekat_vad::preprocessing::{Preprocessor, PreprocessorConfig};
//!
//! let config = PreprocessorConfig {
//!     high_pass_hz: Some(80.0),
//!     ..Default::default()
//! };
//!
//! let mut preprocessor = Preprocessor::new(&config, 16000);
//! let samples: Vec<i16> = vec![0; 320]; // 20ms at 16kHz
//! let cleaned = preprocessor.process(&samples);
//! ```

mod biquad;

#[cfg(feature = "denoise")]
mod denoise;

pub use biquad::BiquadFilter;

#[cfg(feature = "denoise")]
pub use denoise::{Denoiser, DENOISE_SAMPLE_RATE};

use serde::{Deserialize, Serialize};

/// Configuration for the audio preprocessor.
///
/// All fields are optional. Set to `None` or `false` to disable a stage.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// High-pass filter cutoff frequency in Hz.
    ///
    /// Removes low-frequency noise (HVAC, rumble) that can cause false positives.
    /// Recommended: 80Hz for raw mic input, 200Hz for telephony.
    /// Set to `None` to disable.
    #[serde(default)]
    pub high_pass_hz: Option<f32>,

    /// Enable RNNoise-based noise suppression.
    ///
    /// Suppresses stationary background noise while preserving speech.
    /// Requires the `denoise` feature flag and 48kHz sample rate.
    #[serde(default)]
    pub denoise: bool,

    /// Target level for RMS normalization in dBFS.
    ///
    /// Normalizes audio amplitude so VAD thresholds work consistently.
    /// Recommended: -20.0 dBFS. Set to `None` to disable.
    #[serde(default)]
    pub normalize_dbfs: Option<f32>,
}

impl PreprocessorConfig {
    /// No preprocessing — pass audio through unchanged.
    pub fn none() -> Self {
        Self::default()
    }

    /// Preset for raw microphone input.
    ///
    /// Enables high-pass filter at 80Hz to remove room rumble.
    /// With `denoise` feature, also enables noise suppression.
    pub fn raw_mic() -> Self {
        Self {
            high_pass_hz: Some(80.0),
            #[cfg(feature = "denoise")]
            denoise: true,
            #[cfg(not(feature = "denoise"))]
            denoise: false,
            normalize_dbfs: None,
        }
    }

    /// Preset for telephony audio.
    ///
    /// Light high-pass at 200Hz (telephony is already bandpass filtered).
    pub fn telephony() -> Self {
        Self {
            high_pass_hz: Some(200.0),
            ..Default::default()
        }
    }

    /// Returns true if any preprocessing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.high_pass_hz.is_some() || self.denoise || self.normalize_dbfs.is_some()
    }
}

/// Audio preprocessor that applies configured processing stages.
///
/// Each instance maintains its own filter state, so you should create
/// one `Preprocessor` per audio stream (or per VAD config in vad-lab).
///
/// # Processing Order
///
/// 1. High-pass filter (removes low-frequency noise)
/// 2. Noise suppression (RNNoise, requires 48kHz)
/// 3. Normalization (not yet implemented)
pub struct Preprocessor {
    high_pass: Option<BiquadFilter>,
    #[cfg(feature = "denoise")]
    denoiser: Option<Denoiser>,
    sample_rate: u32,
    enabled: bool,
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Preprocessor");
        s.field("high_pass", &self.high_pass.is_some());
        #[cfg(feature = "denoise")]
        s.field("denoiser", &self.denoiser.is_some());
        s.field("sample_rate", &self.sample_rate);
        s.field("enabled", &self.enabled);
        s.finish()
    }
}

impl Preprocessor {
    /// Create a new preprocessor with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Preprocessing configuration
    /// * `sample_rate` - Audio sample rate in Hz
    ///
    /// # Notes
    ///
    /// Noise suppression (`denoise: true`) requires 48kHz sample rate.
    /// If enabled at a different sample rate, it will be silently disabled.
    pub fn new(config: &PreprocessorConfig, sample_rate: u32) -> Self {
        let high_pass = config.high_pass_hz.map(|cutoff| {
            BiquadFilter::highpass_butterworth(cutoff, sample_rate)
        });

        #[cfg(feature = "denoise")]
        let denoiser = if config.denoise && sample_rate == DENOISE_SAMPLE_RATE {
            Some(Denoiser::new(sample_rate))
        } else {
            if config.denoise && sample_rate != DENOISE_SAMPLE_RATE {
                // Silently disable - caller should use 48kHz for denoising
            }
            None
        };

        #[cfg(feature = "denoise")]
        let denoise_enabled = denoiser.is_some();
        #[cfg(not(feature = "denoise"))]
        let denoise_enabled = false;

        let enabled = high_pass.is_some() || denoise_enabled || config.normalize_dbfs.is_some();

        Self {
            high_pass,
            #[cfg(feature = "denoise")]
            denoiser,
            sample_rate,
            enabled,
        }
    }

    /// Returns the sample rate this preprocessor was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns true if any preprocessing stages are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns true if noise suppression is active.
    #[cfg(feature = "denoise")]
    pub fn is_denoising(&self) -> bool {
        self.denoiser.is_some()
    }

    /// Process audio samples and return the preprocessed result.
    ///
    /// Returns a new `Vec<i16>` with the processed samples.
    /// If no preprocessing is enabled, returns a clone of the input.
    pub fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        if !self.enabled {
            return samples.to_vec();
        }

        let mut output = samples.to_vec();

        // Stage 1: High-pass filter
        if let Some(ref mut filter) = self.high_pass {
            filter.process_i16(&mut output);
        }

        // Stage 2: Noise suppression
        #[cfg(feature = "denoise")]
        if let Some(ref mut denoiser) = self.denoiser {
            output = denoiser.process(&output);
        }

        // Stage 3: Normalization (TODO)

        output
    }

    /// Reset all filter states.
    ///
    /// Call this when starting a new audio stream or after a long pause.
    pub fn reset(&mut self) {
        if let Some(ref mut filter) = self.high_pass {
            filter.reset();
        }
        #[cfg(feature = "denoise")]
        if let Some(ref mut denoiser) = self.denoiser {
            denoiser.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PreprocessorConfig::default();
        assert_eq!(config.high_pass_hz, None);
        assert!(!config.denoise);
        assert_eq!(config.normalize_dbfs, None);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_config_presets() {
        let none = PreprocessorConfig::none();
        assert!(!none.is_enabled());

        let raw_mic = PreprocessorConfig::raw_mic();
        assert!(raw_mic.is_enabled());
        assert_eq!(raw_mic.high_pass_hz, Some(80.0));

        let telephony = PreprocessorConfig::telephony();
        assert!(telephony.is_enabled());
        assert_eq!(telephony.high_pass_hz, Some(200.0));
    }

    #[test]
    fn test_config_serde() {
        let config = PreprocessorConfig {
            high_pass_hz: Some(100.0),
            denoise: true,
            normalize_dbfs: Some(-20.0),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: PreprocessorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_config_serde_defaults() {
        // Empty JSON should deserialize to defaults
        let json = "{}";
        let config: PreprocessorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config, PreprocessorConfig::default());
    }

    #[test]
    fn test_preprocessor_disabled() {
        let config = PreprocessorConfig::none();
        let mut preprocessor = Preprocessor::new(&config, 16000);

        assert!(!preprocessor.is_enabled());

        let input: Vec<i16> = vec![100, 200, 300, 400, 500];
        let output = preprocessor.process(&input);
        assert_eq!(input, output);
    }

    #[test]
    fn test_preprocessor_highpass() {
        let config = PreprocessorConfig {
            high_pass_hz: Some(100.0),
            ..Default::default()
        };
        let mut preprocessor = Preprocessor::new(&config, 16000);

        assert!(preprocessor.is_enabled());
        assert_eq!(preprocessor.sample_rate(), 16000);

        // DC offset should be removed
        let dc_input: Vec<i16> = vec![5000; 500];
        let output = preprocessor.process(&dc_input);

        // After settling, output should be near zero
        let last_avg: i32 = output[400..].iter().map(|&s| s.abs() as i32).sum::<i32>() / 100;
        assert!(last_avg < 500, "DC should be attenuated, got avg: {last_avg}");
    }

    #[test]
    fn test_preprocessor_reset() {
        let config = PreprocessorConfig::raw_mic();
        let mut preprocessor = Preprocessor::new(&config, 16000);

        // Process some audio
        let samples: Vec<i16> = vec![1000; 100];
        preprocessor.process(&samples);

        // Reset should not panic
        preprocessor.reset();
    }

    #[cfg(feature = "denoise")]
    #[test]
    fn test_preprocessor_denoise_requires_48khz() {
        // At 16kHz, denoising should be disabled
        let config = PreprocessorConfig {
            denoise: true,
            ..Default::default()
        };
        let preprocessor = Preprocessor::new(&config, 16000);
        assert!(!preprocessor.is_denoising());

        // At 48kHz, denoising should be enabled
        let preprocessor = Preprocessor::new(&config, 48000);
        assert!(preprocessor.is_denoising());
    }

    #[cfg(feature = "denoise")]
    #[test]
    fn test_preprocessor_denoise() {
        let config = PreprocessorConfig {
            denoise: true,
            ..Default::default()
        };
        let mut preprocessor = Preprocessor::new(&config, 48000);

        assert!(preprocessor.is_enabled());
        assert!(preprocessor.is_denoising());

        // Process some audio (silence)
        let input: Vec<i16> = vec![0; 960]; // 20ms at 48kHz
        let output = preprocessor.process(&input);

        // Output length may differ slightly due to frame buffering
        assert!(!output.is_empty());
    }
}
