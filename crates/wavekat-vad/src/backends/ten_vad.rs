//! TEN-VAD backend via FFI to prebuilt C library.
//!
//! This backend wraps [TEN-VAD](https://github.com/TEN-framework/ten-vad) by Agora,
//! a real-time voice activity detector using a neural network with mel-filterbank
//! and pitch estimation preprocessing.
//!
//! # Audio Requirements
//!
//! - **Sample rate:** 16000 Hz only
//! - **Frame size:** Configurable hop size (recommended 160 or 256 samples)
//!
//! # Runtime Dependency
//!
//! The TEN-VAD C library loads its ONNX model from `onnx_model/ten-vad.onnx`
//! relative to the current working directory. Use [`setup_model_dir`] before
//! creating an instance to ensure the model file is in the expected location.
//!
//! # License
//!
//! TEN-VAD is licensed under Apache 2.0 with additional conditions that include
//! a non-compete clause against Agora's offerings. This feature is intended for
//! internal experimentation only and must not be published to crates.io.

use crate::error::VadError;
use crate::{VadCapabilities, VoiceActivityDetector};
use std::ffi::c_void;
use std::os::raw::c_int;

type TenVadHandle = *mut c_void;

extern "C" {
    fn ten_vad_create(handle: *mut TenVadHandle, hop_size: usize, threshold: f32) -> c_int;
    fn ten_vad_process(
        handle: TenVadHandle,
        audio_data: *const i16,
        audio_data_length: usize,
        out_probability: *mut f32,
        out_flag: *mut c_int,
    ) -> c_int;
    fn ten_vad_destroy(handle: *mut TenVadHandle) -> c_int;
}

/// Embedded TEN-VAD ONNX model bytes.
///
/// The C library loads the model from the filesystem, so we embed it here
/// and write it to the expected path at runtime via [`setup_model_dir`].
/// The model is downloaded or copied at build time by build.rs (same pattern as Silero).
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ten_vad.onnx"));

/// Ensure the TEN-VAD ONNX model file exists at the path expected by the C library.
///
/// The C library loads `onnx_model/ten-vad.onnx` relative to the current working
/// directory. This function creates that directory and writes the embedded model
/// file if it doesn't already exist.
///
/// Call this once before creating any [`TenVad`] instances.
///
/// # Errors
///
/// Returns `VadError::BackendError` if the model file cannot be written.
pub fn setup_model_dir() -> Result<(), VadError> {
    let model_dir = std::path::Path::new("onnx_model");
    let model_path = model_dir.join("ten-vad.onnx");

    if model_path.exists() {
        return Ok(());
    }

    std::fs::create_dir_all(model_dir)
        .map_err(|e| VadError::BackendError(format!("failed to create onnx_model dir: {e}")))?;
    std::fs::write(&model_path, MODEL_BYTES)
        .map_err(|e| VadError::BackendError(format!("failed to write TEN-VAD model: {e}")))?;

    Ok(())
}

/// Voice activity detector backed by the TEN-VAD C library.
///
/// Wraps the prebuilt TEN-VAD shared library via FFI. Returns continuous
/// speech probabilities in the range `[0.0, 1.0]`.
///
/// # Example
///
/// ```no_run
/// use wavekat_vad::backends::ten_vad::{TenVad, setup_model_dir};
/// use wavekat_vad::VoiceActivityDetector;
///
/// setup_model_dir().unwrap();
/// let mut vad = TenVad::new(256, 0.5).unwrap();
/// let samples = vec![0i16; 256]; // 16ms at 16kHz
/// let probability = vad.process(&samples, 16000).unwrap();
/// println!("Speech probability: {probability:.3}");
/// ```
pub struct TenVad {
    /// Opaque handle to the C library instance.
    handle: TenVadHandle,
    /// Hop size (samples per frame).
    hop_size: usize,
    /// Detection threshold for the binary flag (we only use probability).
    threshold: f32,
}

// SAFETY: The TEN-VAD C library handle is used single-threaded (via &mut self).
// The handle is an opaque pointer to heap-allocated C++ state that is not
// shared across threads.
unsafe impl Send for TenVad {}

impl TenVad {
    /// Create a new TEN-VAD instance.
    ///
    /// Call [`setup_model_dir`] before this to ensure the ONNX model is available.
    ///
    /// # Arguments
    ///
    /// * `hop_size` — Number of samples per frame (recommended: 160 for 10ms, 256 for 16ms at 16kHz)
    /// * `threshold` — Detection threshold `[0.0, 1.0]` for the binary voice/no-voice flag
    ///
    /// # Errors
    ///
    /// Returns `VadError::BackendError` if the C library fails to initialize.
    pub fn new(hop_size: usize, threshold: f32) -> Result<Self, VadError> {
        let mut handle: TenVadHandle = std::ptr::null_mut();
        let ret = unsafe { ten_vad_create(&mut handle, hop_size, threshold) };
        if ret != 0 || handle.is_null() {
            return Err(VadError::BackendError(
                "ten_vad_create failed (is onnx_model/ten-vad.onnx accessible?)".into(),
            ));
        }
        Ok(Self {
            handle,
            hop_size,
            threshold,
        })
    }
}

impl VoiceActivityDetector for TenVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: 16000,
            frame_size: self.hop_size,
            frame_duration_ms: (self.hop_size as u32 * 1000) / 16000,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        if sample_rate != 16000 {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        if samples.len() != self.hop_size {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: self.hop_size,
            });
        }

        let mut probability: f32 = 0.0;
        let mut flag: c_int = 0;

        let ret = unsafe {
            ten_vad_process(
                self.handle,
                samples.as_ptr(),
                samples.len(),
                &mut probability,
                &mut flag,
            )
        };

        if ret != 0 {
            return Err(VadError::BackendError("ten_vad_process failed".into()));
        }

        Ok(probability.clamp(0.0, 1.0))
    }

    fn reset(&mut self) {
        // TEN-VAD has no public reset API, so we destroy and recreate the handle.
        unsafe {
            ten_vad_destroy(&mut self.handle);
        }
        self.handle = std::ptr::null_mut();

        let mut new_handle: TenVadHandle = std::ptr::null_mut();
        let ret = unsafe { ten_vad_create(&mut new_handle, self.hop_size, self.threshold) };
        if ret == 0 && !new_handle.is_null() {
            self.handle = new_handle;
        }
        // If recreation fails, handle stays null and next process() call will fail.
        // This is acceptable since reset() cannot return an error per the trait.
    }
}

impl Drop for TenVad {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ten_vad_destroy(&mut self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        setup_model_dir().expect("failed to set up model dir");
    }

    #[test]
    fn create_ten_vad() {
        setup();
        let vad = TenVad::new(256, 0.5);
        assert!(vad.is_ok(), "failed to create TEN VAD: {:?}", vad.err());
    }

    #[test]
    fn create_with_hop_160() {
        setup();
        let vad = TenVad::new(160, 0.5);
        assert!(vad.is_ok());
    }

    #[test]
    fn capabilities() {
        setup();
        let vad = TenVad::new(256, 0.5).unwrap();
        let caps = vad.capabilities();
        assert_eq!(caps.sample_rate, 16000);
        assert_eq!(caps.frame_size, 256);
        assert_eq!(caps.frame_duration_ms, 16);
    }

    #[test]
    fn process_silence() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();
        let silence = vec![0i16; 256];
        let result = vad.process(&silence, 16000).unwrap();
        assert!(
            result >= 0.0 && result <= 1.0,
            "probability out of range: {result}"
        );
        // Silence should generally have low probability
        assert!(
            result < 0.5,
            "expected low probability for silence, got {result}"
        );
    }

    #[test]
    fn process_wrong_sample_rate() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();
        let samples = vec![0i16; 256];
        let result = vad.process(&samples, 8000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(8000))));
    }

    #[test]
    fn process_wrong_frame_size() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();
        let samples = vec![0i16; 100];
        let result = vad.process(&samples, 16000);
        assert!(matches!(
            result,
            Err(VadError::InvalidFrameSize {
                got: 100,
                expected: 256
            })
        ));
    }

    #[test]
    fn probability_in_range() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();
        // Low-amplitude noise signal
        let samples: Vec<i16> = (0..256).map(|i| (i % 100) as i16 * 50).collect();
        let result = vad.process(&samples, 16000).unwrap();
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn reset_works() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();

        // Process some audio
        let samples: Vec<i16> = (0..256).map(|i| i as i16 * 10).collect();
        let _ = vad.process(&samples, 16000).unwrap();

        // Reset and process silence
        vad.reset();
        let silence = vec![0i16; 256];
        let result = vad.process(&silence, 16000).unwrap();
        assert!(result < 0.5);
    }

    #[test]
    fn multiple_frames() {
        setup();
        let mut vad = TenVad::new(256, 0.5).unwrap();
        let silence = vec![0i16; 256];

        for _ in 0..10 {
            let result = vad.process(&silence, 16000).unwrap();
            assert!(result >= 0.0 && result <= 1.0);
        }
    }
}
