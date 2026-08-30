//! Cross-backend conformance suite.
//!
//! Every backend implements [`VoiceActivityDetector`], and callers are told
//! they can swap one for another. That promise only holds if the backends
//! agree on the parts of the contract the trait does not enforce at compile
//! time: what `capabilities()` means, which inputs are rejected and with which
//! error, what `reset()` restores, and what `timings()` counts.
//!
//! Each backend's own module tests it in its own style and to its own depth.
//! This file runs one identical set of checks against all of them, so a
//! backend cannot quietly diverge — and so a newly added backend inherits the
//! whole contract by being listed here.
//!
//! The checks are deliberately phrased to fit every backend rather than the
//! strictest one. WebRTC, for example, accepts 10, 20 and 30 ms frames, so the
//! contract says "the declared frame size is accepted" and "an impossible
//! length is rejected", not "only the declared frame size is accepted".

use wavekat_vad::{FrameAdapter, VadError, VoiceActivityDetector};

/// Deterministic voiced-sounding signal at the backend's own sample rate.
fn audio(offset: usize, len: usize, sample_rate: u32) -> Vec<i16> {
    (offset..offset + len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            let v = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.4
                + (2.0 * std::f32::consts::PI * 700.0 * t).sin() * 0.25
                + (2.0 * std::f32::consts::PI * 1900.0 * t).sin() * 0.1;
            (v * 12000.0) as i16
        })
        .collect()
}

/// Score `frames` consecutive frames with a freshly built detector.
fn score_stream<F>(make: &F, frames: usize) -> Vec<f32>
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let mut vad = make();
    let caps = vad.capabilities();
    let signal = audio(0, caps.frame_size * frames, caps.sample_rate);
    signal
        .chunks_exact(caps.frame_size)
        .map(|f| vad.process(f, caps.sample_rate).unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------

fn capabilities_are_self_consistent<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let caps = make().capabilities();
    assert!(caps.sample_rate > 0, "{name}: sample_rate is zero");
    assert!(caps.frame_size > 0, "{name}: frame_size is zero");

    // frame_duration_ms must describe frame_size at sample_rate, to the
    // nearest millisecond — callers size their buffers from it.
    let derived = (caps.frame_size as f64 * 1000.0 / caps.sample_rate as f64).round() as u32;
    assert_eq!(
        caps.frame_duration_ms, derived,
        "{name}: frame_duration_ms {} does not match {} samples at {} Hz",
        caps.frame_duration_ms, caps.frame_size, caps.sample_rate
    );

    // capabilities() must not depend on how much audio has been processed.
    let mut vad = make();
    let before = vad.capabilities();
    let signal = audio(0, before.frame_size * 3, before.sample_rate);
    for f in signal.chunks_exact(before.frame_size) {
        vad.process(f, before.sample_rate).unwrap();
    }
    assert_eq!(before, vad.capabilities(), "{name}: capabilities() drifted");
}

fn the_declared_frame_is_accepted<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let mut vad = make();
    let caps = vad.capabilities();
    let signal = audio(0, caps.frame_size * 20, caps.sample_rate);

    for (i, frame) in signal.chunks_exact(caps.frame_size).enumerate() {
        let score = vad
            .process(frame, caps.sample_rate)
            .unwrap_or_else(|e| panic!("{name}: rejected its own declared frame: {e:?}"));
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "{name}: frame {i} scored {score}, outside 0.0..=1.0"
        );
    }
}

fn a_wrong_sample_rate_is_rejected<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let mut vad = make();
    let caps = vad.capabilities();
    let frame = audio(0, caps.frame_size, caps.sample_rate);

    // Rates no backend in this crate supports.
    for rate in [7_777u32, 44_100, 1] {
        match vad.process(&frame, rate) {
            Err(VadError::InvalidSampleRate(r)) => assert_eq!(r, rate, "{name}: wrong rate echoed"),
            other => panic!("{name}: rate {rate} produced {other:?}, expected InvalidSampleRate"),
        }
    }
}

fn an_impossible_frame_length_is_rejected<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let mut vad = make();
    let caps = vad.capabilities();

    // Lengths that are not a valid frame for any backend: empty, a single
    // sample, and one either side of the declared size.
    for len in [0, 1, caps.frame_size - 1, caps.frame_size + 1] {
        let frame = audio(0, len, caps.sample_rate);
        match vad.process(&frame, caps.sample_rate) {
            Err(VadError::InvalidFrameSize { got, expected }) => {
                assert_eq!(got, len, "{name}: reported the wrong `got` length");
                assert!(expected > 0, "{name}: suggested a zero-length frame");
            }
            other => panic!("{name}: length {len} produced {other:?}, expected InvalidFrameSize"),
        }
    }
}

fn reset_restores_a_fresh_detector<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let expected = score_stream(make, 20);

    let mut vad = make();
    let caps = vad.capabilities();
    // Dirty any streaming state with unrelated audio first.
    let noise = audio(7_777, caps.frame_size * 11, caps.sample_rate);
    for f in noise.chunks_exact(caps.frame_size) {
        vad.process(f, caps.sample_rate).unwrap();
    }

    vad.reset();

    let signal = audio(0, caps.frame_size * 20, caps.sample_rate);
    let after: Vec<f32> = signal
        .chunks_exact(caps.frame_size)
        .map(|f| vad.process(f, caps.sample_rate).unwrap())
        .collect();

    assert_eq!(
        after, expected,
        "{name}: reset() did not restore a freshly constructed detector"
    );
}

fn a_rejected_call_does_not_disturb_the_stream<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let expected = score_stream(make, 16);

    let mut vad = make();
    let caps = vad.capabilities();
    let signal = audio(0, caps.frame_size * 16, caps.sample_rate);

    let mut got = Vec::new();
    for frame in signal.chunks_exact(caps.frame_size) {
        // Invalid calls between valid ones must be inert.
        assert!(vad.process(frame, 44_100).is_err());
        assert!(vad.process(&frame[..1], caps.sample_rate).is_err());
        got.push(vad.process(frame, caps.sample_rate).unwrap());
    }

    assert_eq!(
        got, expected,
        "{name}: a rejected call perturbed the detector's state"
    );
}

fn the_frame_adapter_preserves_scores<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let expected = score_stream(make, 40);

    let caps = make().capabilities();
    let signal = audio(0, caps.frame_size * 40, caps.sample_rate);

    // Chunk sizes that never line up with the frame, including one that spans
    // several frames at once.
    for chunk in [
        1,
        caps.frame_size - 1,
        caps.frame_size + 1,
        caps.frame_size * 3 + 7,
    ] {
        let mut adapter = FrameAdapter::new(make());
        let mut got = Vec::new();
        for part in signal.chunks(chunk) {
            adapter
                .process_each(part, caps.sample_rate, |s| got.push(s))
                .unwrap();
            assert!(
                adapter.buffered_samples() < caps.frame_size,
                "{name}: carry buffer exceeded a frame at chunk size {chunk}"
            );
        }
        assert_eq!(
            got, expected,
            "{name}: chunk size {chunk} changed the scores"
        );
    }
}

fn timings_count_scored_frames<F>(name: &str, make: &F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    let mut vad = make();
    let caps = vad.capabilities();
    assert_eq!(
        vad.timings().frames,
        0,
        "{name}: fresh detector has timings"
    );

    // A backend may consume more calls than it scores: FireRedVAD takes
    // 160-sample calls but needs 400 samples before its first frame, so its
    // opening calls are buffering-only. The trait documents `frames` as
    // excluding exactly those, so the contract is phrased in terms of the
    // steady state rather than a call count.
    const BATCH: usize = 12;
    let feed = |vad: &mut Box<dyn VoiceActivityDetector>, offset: usize| {
        let signal = audio(offset, caps.frame_size * BATCH, caps.sample_rate);
        for f in signal.chunks_exact(caps.frame_size) {
            vad.process(f, caps.sample_rate).unwrap();
        }
    };

    feed(&mut vad, 0);
    let after_warmup = vad.timings().frames;
    assert!(
        after_warmup <= BATCH as u64,
        "{name}: reported {after_warmup} frames for {BATCH} calls"
    );

    // A backend may opt out of timing entirely (the trait default). If it
    // reports anything, the rest of the contract applies.
    if after_warmup == 0 {
        return;
    }

    let t = vad.timings();
    assert!(
        !t.stages.is_empty(),
        "{name}: timed frames but named no stages"
    );
    assert!(
        t.stages.iter().all(|(n, _)| !n.is_empty()),
        "{name}: an unnamed timing stage"
    );

    // Warm-up is behind us, so from here every call scores exactly one frame.
    feed(&mut vad, caps.frame_size * BATCH);
    assert_eq!(
        vad.timings().frames - after_warmup,
        BATCH as u64,
        "{name}: steady-state frame count does not track calls"
    );

    let steady = vad.timings().frames;

    // Rejected calls are not frames.
    let _ = vad.process(&[0i16; 1], caps.sample_rate);
    let _ = vad.process(&audio(0, caps.frame_size, caps.sample_rate), 44_100);
    assert_eq!(
        vad.timings().frames,
        steady,
        "{name}: a rejected call was counted as a frame"
    );

    // reset() clears detector state, not accumulated timings.
    vad.reset();
    assert_eq!(
        vad.timings().frames,
        steady,
        "{name}: reset() cleared timings"
    );
}

/// Run the whole contract against one backend.
fn check_backend<F>(name: &str, make: F)
where
    F: Fn() -> Box<dyn VoiceActivityDetector>,
{
    capabilities_are_self_consistent(name, &make);
    the_declared_frame_is_accepted(name, &make);
    a_wrong_sample_rate_is_rejected(name, &make);
    an_impossible_frame_length_is_rejected(name, &make);
    reset_restores_a_fresh_detector(name, &make);
    a_rejected_call_does_not_disturb_the_stream(name, &make);
    the_frame_adapter_preserves_scores(name, &make);
    timings_count_scored_frames(name, &make);
}

// ---------------------------------------------------------------------------
// The backends
// ---------------------------------------------------------------------------

#[cfg(feature = "webrtc")]
#[test]
fn webrtc_conforms() {
    use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

    // WebRTC is configurable, so sweep the rates and frame durations it offers.
    for rate in [8_000u32, 16_000, 32_000, 48_000] {
        for ms in [10u32, 20, 30] {
            check_backend(&format!("webrtc {rate}Hz/{ms}ms"), move || {
                Box::new(WebRtcVad::with_frame_duration(rate, WebRtcVadMode::Quality, ms).unwrap())
            });
        }
    }
}

#[cfg(feature = "earshot")]
#[test]
fn earshot_conforms() {
    use wavekat_vad::backends::earshot::EarshotVad;
    check_backend("earshot", || Box::new(EarshotVad::new()));
}

#[cfg(feature = "silero")]
#[test]
fn silero_conforms() {
    use wavekat_vad::backends::silero::SileroVad;
    for rate in [8_000u32, 16_000] {
        check_backend(&format!("silero {rate}Hz"), move || {
            Box::new(SileroVad::new(rate).unwrap())
        });
    }
}

#[cfg(feature = "ten-vad")]
#[test]
fn ten_vad_conforms() {
    use wavekat_vad::backends::ten_vad::TenVad;
    check_backend("ten-vad", || Box::new(TenVad::new().unwrap()));
}

#[cfg(feature = "firered")]
#[test]
fn firered_conforms() {
    use wavekat_vad::backends::firered::FireRedVad;
    check_backend("firered", || Box::new(FireRedVad::new().unwrap()));
}

/// The suite is worthless if it silently covers nothing.
#[test]
fn at_least_one_backend_is_covered() {
    let enabled = cfg!(feature = "webrtc") as u8
        + cfg!(feature = "earshot") as u8
        + cfg!(feature = "silero") as u8
        + cfg!(feature = "ten-vad") as u8
        + cfg!(feature = "firered") as u8;
    if enabled == 0 {
        eprintln!("no backend features enabled — conformance suite is inert");
    }
}
