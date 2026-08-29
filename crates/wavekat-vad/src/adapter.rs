//! Frame adapter for matching audio frames to VAD backend requirements.
//!
//! Different VAD backends have different frame size requirements. This module
//! provides an adapter that buffers incoming audio and produces frames of the
//! exact size required by each backend.

use crate::{ProcessTimings, VadCapabilities, VadError, VoiceActivityDetector};

/// Adapts audio frames to match a VAD backend's requirements.
///
/// Buffers incoming samples and produces frames of the exact size
/// required by the wrapped detector. Also handles sample rate validation.
pub struct FrameAdapter {
    /// The wrapped VAD detector.
    inner: Box<dyn VoiceActivityDetector>,
    /// Capabilities of the inner detector.
    capabilities: VadCapabilities,
    /// Carry buffer holding the trailing partial frame.
    ///
    /// # Invariant
    ///
    /// Between calls this holds **fewer than `frame_size`** samples. It is
    /// allocated once, with exactly `frame_size` capacity, and never grows
    /// again, so the steady-state processing path performs no allocation.
    buffer: Vec<i16>,
}

impl FrameAdapter {
    /// Create a new frame adapter wrapping a VAD detector.
    pub fn new(inner: Box<dyn VoiceActivityDetector>) -> Self {
        let capabilities = inner.capabilities();
        // The carry buffer never exceeds `frame_size` samples, so this is the
        // only allocation it will ever make.
        let buffer = Vec::with_capacity(capabilities.frame_size);
        Self {
            inner,
            capabilities,
            buffer,
        }
    }

    /// Returns the capabilities of the wrapped detector.
    pub fn capabilities(&self) -> &VadCapabilities {
        &self.capabilities
    }

    /// Returns the required sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.capabilities.sample_rate
    }

    /// Returns the required frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.capabilities.frame_size
    }

    /// Zero-allocation core: score every complete frame, invoking `on_score`
    /// once per frame in stream order.
    ///
    /// Every other `process*` method on this type is a thin wrapper around
    /// this one. Complete frames are read **directly out of `samples`** — they
    /// are never copied into the carry buffer — so a steady stream of audio is
    /// processed without touching the allocator at all. Only the trailing
    /// partial frame is copied into the carry buffer for the next call.
    ///
    /// Samples are never dropped, duplicated, or reordered: the concatenation
    /// of every `samples` slice ever passed in is split into consecutive
    /// `frame_size` frames, and the remainder is carried.
    ///
    /// # Invariant
    ///
    /// After every call — including one that returns an error —
    /// [`buffered_samples()`](Self::buffered_samples) is strictly less than
    /// [`frame_size()`](Self::frame_size).
    ///
    /// # Arguments
    /// * `samples` — Audio samples (any length, including zero)
    /// * `sample_rate` — Sample rate of the input audio
    /// * `on_score` — Called once per complete frame with its raw score
    ///
    /// # Errors
    ///
    /// [`VadError::InvalidSampleRate`] if `sample_rate` does not match the
    /// wrapped detector, or whatever error the detector returns for a frame.
    /// If the detector fails partway through, the frames already scored have
    /// been reported through `on_score`, and the samples belonging to the
    /// failed frame and everything after it in this call are discarded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "webrtc")]
    /// # {
    /// use wavekat_vad::FrameAdapter;
    /// use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
    ///
    /// let vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
    /// let mut adapter = FrameAdapter::new(Box::new(vad));
    ///
    /// let mut speech_frames = 0usize;
    /// adapter
    ///     .process_each(&vec![0i16; 1000], 16000, |score| {
    ///         if score > 0.5 {
    ///             speech_frames += 1;
    ///         }
    ///     })
    ///     .unwrap();
    /// # }
    /// ```
    pub fn process_each(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
        mut on_score: impl FnMut(f32),
    ) -> Result<(), VadError> {
        if sample_rate != self.capabilities.sample_rate {
            return Err(VadError::InvalidSampleRate(sample_rate));
        }

        let frame_size = self.capabilities.frame_size;
        if frame_size == 0 {
            // A detector that wants zero-length frames cannot be framed for.
            // Reject instead of looping forever.
            return Err(VadError::InvalidFrameSize {
                got: samples.len(),
                expected: 0,
            });
        }

        let mut idx = 0usize;

        // 1. Top up a carried partial frame from the head of `samples`.
        if !self.buffer.is_empty() {
            // Invariant guarantees `self.buffer.len() < frame_size`, so this
            // subtraction cannot underflow.
            let needed = frame_size - self.buffer.len();
            let take = needed.min(samples.len());
            self.buffer.extend_from_slice(&samples[..take]);
            idx = take;

            if self.buffer.len() < frame_size {
                // Consumed the whole input and still short of a frame.
                debug_assert_eq!(idx, samples.len());
                return Ok(());
            }

            // Clear before propagating so the invariant holds even on error.
            let result = self.inner.process(&self.buffer, sample_rate);
            self.buffer.clear();
            on_score(result?);
        }

        // 2. Score complete frames straight out of the caller's slice.
        while samples.len() - idx >= frame_size {
            let score = self
                .inner
                .process(&samples[idx..idx + frame_size], sample_rate)?;
            idx += frame_size;
            on_score(score);
        }

        // 3. Carry the trailing partial frame; the loop above guarantees it is
        //    shorter than `frame_size`.
        if idx < samples.len() {
            self.buffer.extend_from_slice(&samples[idx..]);
        }
        debug_assert!(self.buffer.len() < frame_size);

        Ok(())
    }

    /// Process audio samples, buffering until a complete frame is available.
    ///
    /// Returns `Some(score)` for the **first** complete frame in this call, or
    /// `None` if no frame could be completed.
    ///
    /// # Multi-frame input
    ///
    /// If `samples` spans more than one frame, every complete frame is still
    /// fed to the detector, in order — no audio is dropped and no audio is
    /// over-buffered (see the invariant below). Feeding every frame is
    /// required for correctness: the neural backends are stateful, so skipping
    /// frames would corrupt the stream.
    ///
    /// What is discarded is the *scores* of the frames after the first, since
    /// this signature can only return one. If you need them, use
    /// [`process_each`](Self::process_each) (no allocation),
    /// [`process_all`](Self::process_all) (returns a `Vec`), or
    /// [`process_latest`](Self::process_latest) (most recent score only).
    ///
    /// # Invariant
    ///
    /// After every call, [`buffered_samples()`](Self::buffered_samples) is
    /// strictly less than [`frame_size()`](Self::frame_size).
    ///
    /// # Arguments
    /// * `samples` - Audio samples (any length)
    /// * `sample_rate` - Sample rate of the input audio
    ///
    /// # Errors
    /// Returns an error if the sample rate doesn't match the detector's requirements.
    pub fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<Option<f32>, VadError> {
        let mut first = None;
        self.process_each(samples, sample_rate, |score| {
            if first.is_none() {
                first = Some(score);
            }
        })?;
        Ok(first)
    }

    /// Process all complete frames and collect their scores.
    ///
    /// Returns a vector of scores, one for each complete frame processed.
    ///
    /// This is the **allocating** convenience wrapper around
    /// [`process_each`](Self::process_each): it allocates one `Vec` sized to
    /// the number of frames this call will produce. On a real-time audio path,
    /// prefer [`process_each`](Self::process_each) or
    /// [`process_latest`](Self::process_latest), which allocate nothing.
    pub fn process_all(&mut self, samples: &[i16], sample_rate: u32) -> Result<Vec<f32>, VadError> {
        // `frame_size == 0` is rejected by `process_each`; size the result at
        // zero rather than dividing by it.
        let expected = (self.buffer.len() + samples.len())
            .checked_div(self.capabilities.frame_size)
            .unwrap_or(0);

        let mut results = Vec::with_capacity(expected);
        self.process_each(samples, sample_rate, |score| results.push(score))?;
        Ok(results)
    }

    /// Returns the last score from processing, or 0.0 if no frame was complete.
    ///
    /// This is a convenience method for real-time processing where you only
    /// care about the most recent result. It allocates nothing: the score is
    /// tracked through [`process_each`](Self::process_each)'s callback rather
    /// than collected into a `Vec`.
    pub fn process_latest(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        let mut latest = 0.0f32;
        self.process_each(samples, sample_rate, |score| latest = score)?;
        Ok(latest)
    }

    /// Reset the adapter and the wrapped detector.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inner.reset();
    }

    /// Returns the number of samples currently buffered.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Returns accumulated processing timings from the inner detector.
    pub fn timings(&self) -> ProcessTimings {
        self.inner.timings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    /// Every sample the inner detector saw, concatenated in call order.
    type Seen = Arc<Mutex<Vec<i16>>>;

    // Mock VAD for testing
    struct MockVad {
        sample_rate: u32,
        frame_size: usize,
        call_count: usize,
        /// Records the exact samples handed to the detector, in order.
        seen: Seen,
        /// If set, `process` fails once `call_count` reaches this value.
        fail_at_call: Option<usize>,
    }

    impl MockVad {
        fn new(sample_rate: u32, frame_size: usize) -> Self {
            Self {
                sample_rate,
                frame_size,
                call_count: 0,
                seen: Seen::default(),
                fail_at_call: None,
            }
        }

        fn failing_at(mut self, call: usize) -> Self {
            self.fail_at_call = Some(call);
            self
        }

        fn seen_handle(&self) -> Seen {
            Arc::clone(&self.seen)
        }
    }

    impl VoiceActivityDetector for MockVad {
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities {
                sample_rate: self.sample_rate,
                frame_size: self.frame_size,
                frame_duration_ms: (self.frame_size as u32 * 1000) / self.sample_rate,
            }
        }

        fn process(&mut self, samples: &[i16], _sample_rate: u32) -> Result<f32, VadError> {
            assert_eq!(samples.len(), self.frame_size);
            if self.fail_at_call == Some(self.call_count) {
                return Err(VadError::BackendError("mock failure".into()));
            }
            self.seen
                .lock()
                .expect("seen lock")
                .extend_from_slice(samples);
            self.call_count += 1;
            Ok(0.5)
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }
    }

    /// Mock whose score identifies the frame: it returns the frame's first
    /// sample, so scores can be mapped back to input positions.
    struct EchoVad {
        sample_rate: u32,
        frame_size: usize,
    }

    impl VoiceActivityDetector for EchoVad {
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities {
                sample_rate: self.sample_rate,
                frame_size: self.frame_size,
                frame_duration_ms: (self.frame_size as u32 * 1000) / self.sample_rate,
            }
        }

        fn process(&mut self, samples: &[i16], _sample_rate: u32) -> Result<f32, VadError> {
            assert_eq!(samples.len(), self.frame_size);
            Ok(samples[0] as f32)
        }

        fn reset(&mut self) {}
    }

    /// A distinguishable ramp: sample `n` has value `n`, wrapped into i16.
    fn ramp(len: usize) -> Vec<i16> {
        (0..len).map(|n| (n % 30_000) as i16).collect()
    }

    #[test]
    fn test_adapter_buffers_samples() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send less than a full frame
        let result = adapter.process(&[0i16; 256], 16000).unwrap();
        assert!(result.is_none());
        assert_eq!(adapter.buffered_samples(), 256);

        // Send more to complete the frame
        let result = adapter.process(&[0i16; 256], 16000).unwrap();
        assert!(result.is_some());
        assert_eq!(adapter.buffered_samples(), 0);
    }

    #[test]
    fn test_adapter_handles_multiple_frames() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send two complete frames worth
        let results = adapter.process_all(&[0i16; 1024], 16000).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_adapter_wrong_sample_rate() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        let result = adapter.process(&[0i16; 512], 48000);
        assert!(matches!(result, Err(VadError::InvalidSampleRate(48000))));
    }

    #[test]
    fn test_adapter_reset() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Buffer some samples
        let _ = adapter.process(&[0i16; 256], 16000);
        assert_eq!(adapter.buffered_samples(), 256);

        // Reset
        adapter.reset();
        assert_eq!(adapter.buffered_samples(), 0);
    }

    #[test]
    fn test_process_latest() {
        let mock = MockVad::new(16000, 512);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // Send multiple frames (1600 = 3 full frames + 64 left over)
        let result = adapter.process_latest(&[0i16; 1600], 16000).unwrap();
        assert_eq!(result, 0.5); // Mock returns 0.5
        assert_eq!(adapter.buffered_samples(), 64); // 1600 - 3*512 = 64 left over
    }

    // ------------------------------------------------------------------
    // Framing: exact sample ordering, no loss / duplication / reordering
    // ------------------------------------------------------------------

    /// Drives `chunks` through the adapter and asserts the detector saw
    /// exactly the whole-frame prefix of the input stream, in order.
    fn assert_exact_framing(frame_size: usize, chunk_sizes: &[usize]) {
        let total: usize = chunk_sizes.iter().sum();
        let audio = ramp(total);

        let mock = MockVad::new(16000, frame_size);
        let seen = mock.seen_handle();
        let mut adapter = FrameAdapter::new(Box::new(mock));

        let mut offset = 0usize;
        let mut scored = 0usize;
        for &size in chunk_sizes {
            adapter
                .process_each(&audio[offset..offset + size], 16000, |_| scored += 1)
                .expect("process_each");
            offset += size;

            // Carry invariant, checked after every single call.
            assert!(
                adapter.buffered_samples() < frame_size,
                "carry invariant violated: {} buffered, frame_size {frame_size}",
                adapter.buffered_samples()
            );
            // Everything fed in is either scored or carried, nothing else.
            let consumed = seen.lock().unwrap().len();
            assert_eq!(
                consumed + adapter.buffered_samples(),
                offset,
                "samples lost or duplicated after {offset} input samples"
            );
        }

        let expected_frames = total / frame_size;
        assert_eq!(scored, expected_frames, "wrong number of frames scored");

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.len(),
            expected_frames * frame_size,
            "wrong number of samples reached the detector"
        );
        // The load-bearing assertion: exact ORDER, not just counts.
        assert_eq!(
            &seen[..],
            &audio[..expected_frames * frame_size],
            "samples were reordered, dropped, or duplicated"
        );
        assert_eq!(adapter.buffered_samples(), total - seen.len());
    }

    #[test]
    fn framing_one_sample_at_a_time() {
        assert_exact_framing(256, &[1; 1030]);
    }

    #[test]
    fn framing_320_sample_transport_chunks() {
        // 20 ms packets into a 16 ms (256-sample) frame size.
        assert_exact_framing(256, &[320; 25]);
    }

    #[test]
    fn framing_several_complete_frames_per_chunk() {
        // Exact multiples: never any carry.
        assert_exact_framing(256, &[1024; 8]);
    }

    #[test]
    fn framing_partial_carry_across_calls() {
        // Deliberately awkward sizes, including empty and sub-frame chunks.
        assert_exact_framing(256, &[100, 0, 7, 511, 1, 900, 3, 256, 5, 1024, 63]);
    }

    #[test]
    fn framing_chunk_smaller_than_remaining_need() {
        // Two consecutive partial top-ups that still do not complete a frame.
        assert_exact_framing(512, &[100, 100, 100, 100, 100, 100]);
    }

    // ------------------------------------------------------------------
    // Carry invariant: buffered_samples() < frame_size, always
    // ------------------------------------------------------------------

    #[test]
    fn process_upholds_carry_invariant_on_multi_frame_input() {
        // Regression: the previous implementation appended everything to the
        // carry buffer and drained a single frame, leaving 768 of these 1024
        // samples buffered — three whole frames past the invariant.
        let mock = MockVad::new(16000, 256);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        assert!(adapter.process(&[0i16; 1024], 16000).unwrap().is_some());
        assert_eq!(
            adapter.buffered_samples(),
            0,
            "1024 samples at frame_size 256 must leave nothing buffered"
        );

        // The call that used to underflow.
        assert!(adapter.process(&[0i16; 1024], 16000).unwrap().is_some());
        assert!(adapter.buffered_samples() < 256);

        // Same, but with a genuine remainder carried in first.
        adapter.reset();
        assert!(adapter.process(&[0i16; 1000], 16000).unwrap().is_some());
        assert_eq!(adapter.buffered_samples(), 1000 - 3 * 256);
        assert!(adapter.process(&[0i16; 1024], 16000).unwrap().is_some());
        assert!(adapter.buffered_samples() < 256);
    }

    /// The downstream consumer's exact shape: 20 ms / 320-sample ingress
    /// chunks into Earshot's 16 ms / 256-sample frames, one `process()` call
    /// per chunk, for the length of a phone call.
    ///
    /// Regression: the previous implementation drained at most one frame per
    /// call, so a 320-into-256 stream accumulated 64 extra samples on every
    /// call — the carry buffer grew without bound and the audio it emitted
    /// fell further and further behind real time.
    #[test]
    fn repeated_320_sample_process_calls_do_not_grow_the_carry_buffer() {
        let mock = MockVad::new(16000, 256);
        let seen = mock.seen_handle();
        let mut adapter = FrameAdapter::new(Box::new(mock));

        // 3000 chunks == 60 seconds of call audio.
        for i in 1..=3000usize {
            adapter.process(&[0i16; 320], 16000).unwrap();
            assert!(
                adapter.buffered_samples() < 256,
                "carry buffer grew to {} after {i} chunks",
                adapter.buffered_samples()
            );
            // No backlog: everything in is either already scored or carried.
            assert_eq!(
                seen.lock().unwrap().len() + adapter.buffered_samples(),
                i * 320
            );
        }
    }

    #[test]
    fn carry_invariant_holds_across_every_api_and_chunk_size() {
        for frame_size in [1usize, 2, 160, 256, 512] {
            for chunk in [0usize, 1, 3, 255, 256, 257, 320, 1024, 4096] {
                let mut adapter = FrameAdapter::new(Box::new(MockVad::new(16000, frame_size)));
                let audio = ramp(chunk);
                for _ in 0..4 {
                    adapter.process(&audio, 16000).unwrap();
                    assert!(adapter.buffered_samples() < frame_size, "process");
                    adapter.process_all(&audio, 16000).unwrap();
                    assert!(adapter.buffered_samples() < frame_size, "process_all");
                    adapter.process_latest(&audio, 16000).unwrap();
                    assert!(adapter.buffered_samples() < frame_size, "process_latest");
                    adapter.process_each(&audio, 16000, |_| {}).unwrap();
                    assert!(adapter.buffered_samples() < frame_size, "process_each");
                }
            }
        }
    }

    #[test]
    fn carry_invariant_holds_when_the_detector_errors() {
        // Fail on the very first frame, right after a partial top-up.
        let mock = MockVad::new(16000, 256).failing_at(0);
        let mut adapter = FrameAdapter::new(Box::new(mock));

        adapter.process_each(&ramp(200), 16000, |_| {}).unwrap();
        assert_eq!(adapter.buffered_samples(), 200);

        let err = adapter
            .process_each(&ramp(100), 16000, |_| {})
            .expect_err("detector should have failed");
        assert!(matches!(err, VadError::BackendError(_)), "{err:?}");
        assert!(
            adapter.buffered_samples() < 256,
            "carry invariant violated after an inner error: {}",
            adapter.buffered_samples()
        );

        // And on a mid-slice frame, where the failure happens in the
        // read-straight-from-the-slice loop.
        let mock = MockVad::new(16000, 256).failing_at(2);
        let mut adapter = FrameAdapter::new(Box::new(mock));
        let err = adapter
            .process_each(&ramp(2000), 16000, |_| {})
            .expect_err("detector should have failed");
        assert!(matches!(err, VadError::BackendError(_)), "{err:?}");
        assert!(adapter.buffered_samples() < 256);
    }

    // ------------------------------------------------------------------
    // Documented semantics of the wrapper APIs
    // ------------------------------------------------------------------

    #[test]
    fn process_returns_the_first_frame_score_but_feeds_every_frame() {
        let echo = EchoVad {
            sample_rate: 16000,
            frame_size: 4,
        };
        let mut adapter = FrameAdapter::new(Box::new(echo));

        // ramp(): frames start at 0, 4, 8, 12 -> scores 0.0, 4.0, 8.0, 12.0.
        let first = adapter.process(&ramp(16), 16000).unwrap();
        assert_eq!(first, Some(0.0), "process() must return the FIRST score");
        assert_eq!(adapter.buffered_samples(), 0);

        // Nothing was buffered, so the stream continues where it left off:
        // if later frames had been dropped instead of fed, the next call's
        // first frame would not line up with sample 16.
        let mock = MockVad::new(16000, 4);
        let seen = mock.seen_handle();
        let mut adapter = FrameAdapter::new(Box::new(mock));
        let audio = ramp(16);
        adapter.process(&audio, 16000).unwrap();
        assert_eq!(&seen.lock().unwrap()[..], &audio[..], "frames were dropped");
    }

    #[test]
    fn process_returns_none_when_no_frame_completes() {
        let mut adapter = FrameAdapter::new(Box::new(MockVad::new(16000, 512)));
        assert_eq!(adapter.process(&[0i16; 0], 16000).unwrap(), None);
        assert_eq!(adapter.process(&[0i16; 511], 16000).unwrap(), None);
        assert_eq!(adapter.buffered_samples(), 511);
    }

    #[test]
    fn process_all_and_process_latest_agree_with_process_each() {
        let audio = ramp(3000);

        let mut a = FrameAdapter::new(Box::new(EchoVad {
            sample_rate: 16000,
            frame_size: 256,
        }));
        let mut via_each = Vec::new();
        a.process_each(&audio, 16000, |s| via_each.push(s)).unwrap();

        let mut b = FrameAdapter::new(Box::new(EchoVad {
            sample_rate: 16000,
            frame_size: 256,
        }));
        let via_all = b.process_all(&audio, 16000).unwrap();

        let mut c = FrameAdapter::new(Box::new(EchoVad {
            sample_rate: 16000,
            frame_size: 256,
        }));
        let via_latest = c.process_latest(&audio, 16000).unwrap();

        assert_eq!(via_each, via_all);
        assert_eq!(via_latest, *via_each.last().unwrap());
        assert_eq!(a.buffered_samples(), b.buffered_samples());
        assert_eq!(b.buffered_samples(), c.buffered_samples());
    }

    #[test]
    fn process_latest_is_zero_when_no_frame_completes() {
        let mut adapter = FrameAdapter::new(Box::new(EchoVad {
            sample_rate: 16000,
            frame_size: 512,
        }));
        assert_eq!(adapter.process_latest(&ramp(100), 16000).unwrap(), 0.0);
    }

    #[test]
    fn every_api_rejects_the_wrong_sample_rate() {
        let mut adapter = FrameAdapter::new(Box::new(MockVad::new(16000, 512)));
        let audio = ramp(512);

        assert!(matches!(
            adapter.process(&audio, 48000),
            Err(VadError::InvalidSampleRate(48000))
        ));
        assert!(matches!(
            adapter.process_all(&audio, 8000),
            Err(VadError::InvalidSampleRate(8000))
        ));
        assert!(matches!(
            adapter.process_latest(&audio, 44100),
            Err(VadError::InvalidSampleRate(44100))
        ));
        assert!(matches!(
            adapter.process_each(&audio, 32000, |_| {}),
            Err(VadError::InvalidSampleRate(32000))
        ));
        // A rejected call must not consume input.
        assert_eq!(adapter.buffered_samples(), 0);
    }

    #[test]
    fn process_each_with_empty_input_is_a_no_op() {
        let mock = MockVad::new(16000, 256);
        let seen = mock.seen_handle();
        let mut adapter = FrameAdapter::new(Box::new(mock));

        adapter
            .process_each(&[], 16000, |_| panic!("scored nothing"))
            .unwrap();
        assert_eq!(adapter.buffered_samples(), 0);

        adapter.process_each(&ramp(100), 16000, |_| {}).unwrap();
        adapter
            .process_each(&[], 16000, |_| panic!("scored nothing"))
            .unwrap();
        assert_eq!(adapter.buffered_samples(), 100);
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn reset_drops_the_carry_and_restarts_framing() {
        let mock = MockVad::new(16000, 256);
        let seen = mock.seen_handle();
        let mut adapter = FrameAdapter::new(Box::new(mock));

        adapter.process_each(&ramp(300), 16000, |_| {}).unwrap();
        assert_eq!(adapter.buffered_samples(), 44);

        adapter.reset();
        assert_eq!(adapter.buffered_samples(), 0);

        let audio = ramp(256);
        adapter.process_each(&audio, 16000, |_| {}).unwrap();
        let seen = seen.lock().unwrap();
        // First 256 from the pre-reset stream, then the post-reset frame
        // starting cleanly at sample 0 again.
        assert_eq!(&seen[..256], &ramp(300)[..256]);
        assert_eq!(&seen[256..], &audio[..]);
    }
}
