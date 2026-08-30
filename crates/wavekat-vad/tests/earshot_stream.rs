//! End-to-end streaming tests for the Earshot backend behind `FrameAdapter`.
//!
//! These exercise only the public API, in the shape a real transport uses it:
//! audio arriving in packets that do not line up with the backend's 256-sample
//! frame, fed one packet at a time for the length of a call.
#![cfg(feature = "earshot")]

use wavekat_vad::backends::earshot::{EarshotVad, FRAME_SIZE, SAMPLE_RATE};
use wavekat_vad::{FrameAdapter, VoiceActivityDetector};

/// Deterministic voiced-sounding signal: a fixed harmonic stack.
fn signal(offset: usize, len: usize) -> Vec<i16> {
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

/// Scores from feeding frame-aligned 256-sample frames straight to the backend.
fn direct_scores(audio: &[i16]) -> Vec<f32> {
    let mut vad = EarshotVad::new();
    audio
        .chunks_exact(FRAME_SIZE)
        .map(|f| vad.process(f, SAMPLE_RATE).unwrap())
        .collect()
}

/// Scores from feeding the same audio through the adapter in `chunk_sizes`.
fn adapted_scores(audio: &[i16], chunk_sizes: &[usize]) -> Vec<f32> {
    let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
    let mut scores = Vec::new();
    let mut offset = 0usize;
    for &size in chunk_sizes.iter().cycle() {
        if offset >= audio.len() {
            break;
        }
        let end = (offset + size).min(audio.len());
        adapter
            .process_each(&audio[offset..end], SAMPLE_RATE, |s| scores.push(s))
            .unwrap();
        assert!(
            adapter.buffered_samples() < FRAME_SIZE,
            "carry buffer reached {} samples",
            adapter.buffered_samples()
        );
        offset = end;
    }
    scores
}

/// Deterministic LCG, so a failing chunking reproduces exactly.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

#[test]
fn transport_chunks_score_identically_to_aligned_frames() {
    // The whole point of the adapter: how the audio is packetised on the way
    // in must not change a single score coming out. 20 ms packets over a
    // 16 ms frame never line up, so every frame after the first is assembled
    // across a packet boundary.
    let audio = signal(0, 320 * 60);
    let expected = direct_scores(&audio);
    let got = adapted_scores(&audio, &[320]);

    assert_eq!(got.len(), audio.len() / FRAME_SIZE);
    assert_eq!(
        got, expected,
        "320-sample packets did not reproduce frame-aligned scores"
    );
}

#[test]
fn jittery_chunk_sizes_score_identically_to_aligned_frames() {
    let audio = signal(0, FRAME_SIZE * 200);
    let expected = direct_scores(&audio);

    for seed in 0..6u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let chunk_sizes: Vec<usize> = (0..24)
            .map(|_| ((lcg(&mut state) % 900) + 1) as usize)
            .collect();
        let got = adapted_scores(&audio, &chunk_sizes);
        assert_eq!(
            got, expected,
            "seed {seed}: chunking changed the scores ({chunk_sizes:?})"
        );
    }
}

#[test]
fn a_minute_of_streaming_stays_in_step() {
    // A stream that buffers a little more each packet drifts further behind
    // real time the longer the call runs, so assert progress against the
    // wall-clock position of the stream rather than only at the end.
    let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
    let packet = signal(0, 320);
    let mut scored = 0usize;

    for i in 1..=3000usize {
        // 3000 * 20 ms = 60 s
        adapter
            .process_each(&packet, SAMPLE_RATE, |s| {
                assert!(s.is_finite() && (0.0..=1.0).contains(&s), "score {s}");
                scored += 1;
            })
            .unwrap();

        let fed = i * 320;
        assert!(
            adapter.buffered_samples() < FRAME_SIZE,
            "carry buffer grew to {} after {i} packets",
            adapter.buffered_samples()
        );
        // Every sample fed in has either been scored or is still carried.
        assert_eq!(
            scored * FRAME_SIZE + adapter.buffered_samples(),
            fed,
            "stream fell out of step after {i} packets"
        );
    }

    assert_eq!(scored, (3000 * 320) / FRAME_SIZE);
    assert_eq!(adapter.timings().frames, scored as u64);
}

#[test]
fn a_voiced_region_scores_above_the_silence_around_it() {
    // End-to-end through the adapter: silence, then signal, then silence.
    // The middle has to stand out, or the backend is wired up but deaf.
    let quiet = FRAME_SIZE * 40;
    let loud = FRAME_SIZE * 40;
    let mut audio = vec![0i16; quiet];
    audio.extend_from_slice(&signal(0, loud));
    audio.extend(std::iter::repeat_n(0i16, quiet));

    let scores = adapted_scores(&audio, &[320]);
    assert_eq!(scores.len(), audio.len() / FRAME_SIZE);

    let mean = |r: &[f32]| r.iter().sum::<f32>() / r.len() as f32;
    // Skip the frames straddling each transition; they are legitimately mixed.
    let lead_in = mean(&scores[..35]);
    let voiced = mean(&scores[45..75]);
    let tail = mean(&scores[85..115]);

    assert!(
        voiced > lead_in + 0.2,
        "voiced region {voiced:.3} not separated from lead-in silence {lead_in:.3}"
    );
    assert!(
        voiced > tail + 0.2,
        "voiced region {voiced:.3} not separated from trailing silence {tail:.3}"
    );
}

#[test]
fn reset_restarts_the_stream_through_the_adapter() {
    // A reset adapter must behave exactly like a new one: both the carried
    // partial frame and the backend's recurrent context have to be cleared,
    // or the next call starts mid-stream.
    let audio = signal(0, FRAME_SIZE * 30);

    let mut reused = FrameAdapter::new(Box::new(EarshotVad::new()));
    // Dirty it with unrelated audio, ending on a partial frame.
    reused
        .process_each(&signal(9_001, FRAME_SIZE * 7 + 100), SAMPLE_RATE, |_| {})
        .unwrap();
    assert_eq!(reused.buffered_samples(), 100);

    reused.reset();
    assert_eq!(reused.buffered_samples(), 0);

    let mut after_reset = Vec::new();
    reused
        .process_each(&audio, SAMPLE_RATE, |s| after_reset.push(s))
        .unwrap();

    assert_eq!(
        after_reset,
        direct_scores(&audio),
        "reset() did not restore fresh-stream behaviour"
    );
}

#[test]
fn a_wrong_rate_packet_does_not_disturb_the_stream() {
    // A caller that hands over a mislabelled packet gets an error, and the
    // stream carries on from exactly where it was.
    let audio = signal(0, 320 * 40);
    let expected = direct_scores(&audio);

    let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
    let mut got = Vec::new();
    for chunk in audio.chunks(320) {
        assert!(adapter.process_each(chunk, 48_000, |_| {}).is_err());
        adapter
            .process_each(chunk, SAMPLE_RATE, |s| got.push(s))
            .unwrap();
    }

    assert_eq!(got, expected, "a rejected packet perturbed the stream");
}
