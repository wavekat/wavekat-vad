//! Earshot VAD backend — pure Rust, no ONNX runtime, no C dependencies.
//!
//! This backend wraps the [`earshot`](https://crates.io/crates/earshot) crate,
//! a small recurrent neural voice activity detector implemented entirely in
//! safe-ish pure Rust. Because it links no native code, enabling this backend
//! pulls in **no** ONNX Runtime, no model download at build time, and — most
//! importantly for hosts that also link WebRTC — no `libfvad`/`webrtc-vad`
//! C symbols.
//!
//! # Audio requirements
//!
//! - **Sample rate:** 16000 Hz only
//! - **Frame size:** exactly 256 samples (16 ms)
//! - **Format:** 16-bit signed integers (i16), mono
//!
//! Anything else is rejected with a typed error — [`VadError::InvalidSampleRate`]
//! for the wrong rate, [`VadError::InvalidFrameSize`] for the wrong length.
//! Use [`FrameAdapter`](crate::FrameAdapter) if your transport delivers audio
//! in chunks that are not 256 samples (e.g. 20 ms / 320-sample packets).
//!
//! # Output
//!
//! [`process`](VoiceActivityDetector::process) returns the **raw, unthresholded**
//! score in `0.0..=1.0`, where `0.0` is silence and `1.0` is speech. This crate
//! deliberately does no thresholding, hangover, endpointing, or call control —
//! that is the caller's policy decision. `0.5` is the value the upstream crate
//! suggests as a starting threshold.
//!
//! # Streaming state
//!
//! Earshot is stateful: it keeps a 768-sample overlapping analysis window plus
//! three frames of recurrent feature context. Consequences:
//!
//! - One [`EarshotVad`] per audio stream. State is owned by the value, with no
//!   interior mutability and no globals, so two detectors never influence each
//!   other.
//! - Frames must be fed **in order** and without gaps.
//! - Call [`reset()`](VoiceActivityDetector::reset) when starting a new stream;
//!   it restores exactly the state of a freshly constructed detector.
//!
//! # Example
//!
//! ```
//! use wavekat_vad::backends::earshot::EarshotVad;
//! use wavekat_vad::VoiceActivityDetector;
//!
//! let mut vad = EarshotVad::new();
//! let samples = vec![0i16; 256]; // 16 ms at 16 kHz
//! let score = vad.process(&samples, 16000).unwrap();
//! println!("Speech score: {score:.3}");
//! ```

use crate::error::VadError;
use crate::{ProcessTimings, VadCapabilities, VoiceActivityDetector};
use earshot::{DefaultPredictor, Detector};
use std::time::{Duration, Instant};

/// The only sample rate Earshot accepts, in Hz.
pub const SAMPLE_RATE: u32 = 16_000;

/// The exact frame size Earshot accepts, in samples.
pub const FRAME_SIZE: usize = 256;

/// Frame duration in milliseconds (`FRAME_SIZE` at `SAMPLE_RATE`).
pub const FRAME_DURATION_MS: u32 = 16;

/// Voice activity detector backed by the pure-Rust [`earshot`] crate.
///
/// Create one per audio stream. See the [module documentation](self) for
/// audio requirements and streaming semantics.
pub struct EarshotVad {
    /// Boxed because `Detector` keeps ~8 KiB of state inline; `default_boxed`
    /// builds it directly on the heap instead of via an 8 KiB stack temporary.
    detector: Box<Detector<DefaultPredictor>>,
    inference_time: Duration,
    frames: u64,
}

impl EarshotVad {
    /// Create a new Earshot detector with fresh state.
    ///
    /// Infallible: there is no model to load and no runtime to initialise.
    pub fn new() -> Self {
        Self {
            detector: Detector::default_boxed(),
            inference_time: Duration::ZERO,
            frames: 0,
        }
    }
}

impl Default for EarshotVad {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceActivityDetector for EarshotVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: SAMPLE_RATE,
            frame_size: FRAME_SIZE,
            frame_duration_ms: FRAME_DURATION_MS,
        }
    }

    /// Score one 256-sample frame of 16 kHz audio.
    ///
    /// Returns the raw score in `0.0..=1.0`. No allocation is performed.
    ///
    /// # Errors
    ///
    /// - [`VadError::InvalidSampleRate`] if `sample_rate != 16000`.
    /// - [`VadError::InvalidFrameSize`] if `samples.len() != 256`.
    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        if sample_rate != SAMPLE_RATE {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }
        if samples.len() != FRAME_SIZE {
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: FRAME_SIZE,
            });
        }

        let start = Instant::now();
        let score = self.detector.predict_i16(samples);
        self.inference_time += start.elapsed();
        self.frames += 1;

        Ok(score)
    }

    /// Restore the detector to the state of a freshly constructed instance.
    ///
    /// Clears the analysis window and the recurrent feature context. Does not
    /// clear accumulated [`timings()`](VoiceActivityDetector::timings).
    fn reset(&mut self) {
        self.detector.reset();
    }

    fn timings(&self) -> ProcessTimings {
        ProcessTimings {
            stages: vec![("inference", self.inference_time)],
            frames: self.frames,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-speech: a sum of sines with a fixed seed-free
    /// formula, so every run produces byte-identical input.
    fn tone(offset: usize, len: usize) -> Vec<i16> {
        (offset..offset + len)
            .map(|n| {
                let t = n as f32 / SAMPLE_RATE as f32;
                let v = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.4
                    + (2.0 * std::f32::consts::PI * 700.0 * t).sin() * 0.25
                    + (2.0 * std::f32::consts::PI * 1900.0 * t).sin() * 0.1;
                (v * 12000.0) as i16
            })
            .collect()
    }

    #[test]
    fn capabilities_are_16khz_256_samples() {
        let vad = EarshotVad::new();
        assert_eq!(
            vad.capabilities(),
            VadCapabilities {
                sample_rate: 16_000,
                frame_size: 256,
                frame_duration_ms: 16,
            }
        );
    }

    #[test]
    fn wrong_sample_rate_is_rejected() {
        let mut vad = EarshotVad::new();
        let frame = vec![0i16; FRAME_SIZE];
        for rate in [8_000u32, 32_000, 44_100, 48_000, 0] {
            let err = vad.process(&frame, rate).unwrap_err();
            assert!(
                matches!(err, VadError::InvalidSampleRate(r) if r == rate),
                "rate {rate} produced {err:?}"
            );
        }
    }

    #[test]
    fn wrong_frame_length_is_rejected() {
        let mut vad = EarshotVad::new();
        for len in [0usize, 1, 160, 255, 257, 320, 512] {
            let err = vad.process(&vec![0i16; len], SAMPLE_RATE).unwrap_err();
            match err {
                VadError::InvalidFrameSize { got, expected } => {
                    assert_eq!(got, len);
                    assert_eq!(expected, FRAME_SIZE);
                }
                other => panic!("len {len} produced {other:?}"),
            }
        }
    }

    #[test]
    fn sample_rate_is_checked_before_frame_size() {
        // A call that is wrong in both ways reports the rate, deterministically.
        let mut vad = EarshotVad::new();
        let err = vad.process(&[0i16; 10], 48_000).unwrap_err();
        assert!(
            matches!(err, VadError::InvalidSampleRate(48_000)),
            "{err:?}"
        );
    }

    #[test]
    fn scores_are_finite_and_in_unit_range() {
        let mut vad = EarshotVad::new();
        let audio = tone(0, FRAME_SIZE * 40);
        for (i, frame) in audio.chunks_exact(FRAME_SIZE).enumerate() {
            let score = vad.process(frame, SAMPLE_RATE).unwrap();
            assert!(score.is_finite(), "frame {i} score {score} is not finite");
            assert!(
                (0.0..=1.0).contains(&score),
                "frame {i} score {score} outside 0.0..=1.0"
            );
        }
        // Silence must also stay in range (and must not produce the -1.0
        // sentinel the upstream crate returns for malformed frames).
        let mut vad = EarshotVad::new();
        for _ in 0..40 {
            let score = vad.process(&[0i16; FRAME_SIZE], SAMPLE_RATE).unwrap();
            assert!(score.is_finite() && (0.0..=1.0).contains(&score), "{score}");
        }
    }

    #[test]
    fn reset_restores_a_fresh_detectors_output_sequence() {
        let audio = tone(0, FRAME_SIZE * 24);
        let run = |vad: &mut EarshotVad| -> Vec<f32> {
            audio
                .chunks_exact(FRAME_SIZE)
                .map(|f| vad.process(f, SAMPLE_RATE).unwrap())
                .collect()
        };

        let mut fresh = EarshotVad::new();
        let expected = run(&mut fresh);

        let mut reused = EarshotVad::new();
        // Dirty the recurrent state with different audio first.
        let noise = tone(7_777, FRAME_SIZE * 13);
        for f in noise.chunks_exact(FRAME_SIZE) {
            reused.process(f, SAMPLE_RATE).unwrap();
        }
        let dirty = run(&mut reused);
        assert_ne!(
            dirty, expected,
            "test is vacuous: carried state did not change the output"
        );

        reused.reset();
        let after_reset = run(&mut reused);
        assert_eq!(
            after_reset, expected,
            "reset() did not restore fresh-detector behaviour"
        );
    }

    #[test]
    fn instances_are_state_isolated() {
        let audio_a = tone(0, FRAME_SIZE * 16);
        let audio_b = tone(31_337, FRAME_SIZE * 16);

        // Baseline: each stream scored by its own untouched detector.
        let mut solo_a = EarshotVad::new();
        let expected_a: Vec<f32> = audio_a
            .chunks_exact(FRAME_SIZE)
            .map(|f| solo_a.process(f, SAMPLE_RATE).unwrap())
            .collect();
        let mut solo_b = EarshotVad::new();
        let expected_b: Vec<f32> = audio_b
            .chunks_exact(FRAME_SIZE)
            .map(|f| solo_b.process(f, SAMPLE_RATE).unwrap())
            .collect();

        // Interleaved: two detectors alive at once, alternating frames.
        let mut vad_a = EarshotVad::new();
        let mut vad_b = EarshotVad::new();
        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        for (fa, fb) in audio_a
            .chunks_exact(FRAME_SIZE)
            .zip(audio_b.chunks_exact(FRAME_SIZE))
        {
            got_a.push(vad_a.process(fa, SAMPLE_RATE).unwrap());
            got_b.push(vad_b.process(fb, SAMPLE_RATE).unwrap());
        }

        assert_eq!(got_a, expected_a, "detector A was perturbed by detector B");
        assert_eq!(got_b, expected_b, "detector B was perturbed by detector A");
        assert_ne!(
            expected_a, expected_b,
            "test is vacuous: the two streams score identically"
        );
    }

    #[test]
    fn timings_accumulate_and_survive_reset() {
        let mut vad = EarshotVad::new();
        assert_eq!(vad.timings().frames, 0);

        for _ in 0..5 {
            vad.process(&[0i16; FRAME_SIZE], SAMPLE_RATE).unwrap();
        }
        let t = vad.timings();
        assert_eq!(t.frames, 5);
        assert_eq!(t.stages.len(), 1);
        assert_eq!(t.stages[0].0, "inference");

        // Rejected calls must not be counted as processed frames.
        let _ = vad.process(&[0i16; 10], SAMPLE_RATE);
        let _ = vad.process(&[0i16; FRAME_SIZE], 8_000);
        assert_eq!(vad.timings().frames, 5);

        vad.reset();
        assert_eq!(vad.timings().frames, 5, "reset() must not clear timings");
    }

    #[test]
    fn works_behind_the_frame_adapter() {
        use crate::FrameAdapter;

        let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
        assert_eq!(adapter.frame_size(), FRAME_SIZE);
        assert_eq!(adapter.sample_rate(), SAMPLE_RATE);

        // 20 ms transport packets (320 samples) over a 16 ms frame size.
        let audio = tone(0, 320 * 20);
        let mut scored = 0usize;
        for chunk in audio.chunks(320) {
            adapter
                .process_each(chunk, SAMPLE_RATE, |s| {
                    assert!(s.is_finite() && (0.0..=1.0).contains(&s), "{s}");
                    scored += 1;
                })
                .unwrap();
            assert!(adapter.buffered_samples() < FRAME_SIZE);
        }
        assert_eq!(scored, (320 * 20) / FRAME_SIZE);
    }
}
