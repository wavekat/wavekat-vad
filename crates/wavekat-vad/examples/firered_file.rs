//! Run FireRedVAD on a WAV file and print speech probabilities.
//!
//! ```sh
//! cargo run --example firered_file --features firered -- path/to/audio.wav
//! ```
//!
//! Accepts any WAV file — automatically resamples to 16kHz and converts to mono.
//!
//! To use this code in your own project, add these dependencies:
//! ```sh
//! cargo add wavekat-vad --features firered
//! cargo add hound
//! ```

use std::env;

use wavekat_vad::backends::firered::FireRedVad;
use wavekat_vad::preprocessing::AudioResampler;
use wavekat_vad::VoiceActivityDetector;

fn main() {
    let path = env::args().nth(1).expect("usage: firered_file <wav-file>");

    let mut reader = hound::WavReader::open(&path).expect("failed to open WAV file");
    let spec = reader.spec();

    println!(
        "File: {path}\n  channels: {}, sample rate: {} Hz, bits: {}",
        spec.channels, spec.sample_rate, spec.bits_per_sample
    );

    // Read all samples (take first channel if stereo)
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.expect("failed to read sample"))
        .collect();

    let duration = samples.len() as f64 / spec.sample_rate as f64;
    println!("  samples: {}, duration: {duration:.2}s", samples.len());

    // Resample to 16kHz if needed
    let target_rate = 16000;
    let samples = if spec.sample_rate != target_rate {
        println!("  resampling {}Hz -> {}Hz", spec.sample_rate, target_rate);
        let mut resampler =
            AudioResampler::new(spec.sample_rate, target_rate).expect("failed to create resampler");
        resampler.process(&samples)
    } else {
        samples
    };

    println!();

    let mut vad = FireRedVad::new().expect("failed to create FireRedVAD");
    let caps = vad.capabilities();

    // Only process the first 10 seconds
    let max_samples = target_rate as usize * 10;
    let samples = &samples[..samples.len().min(max_samples)];

    // Process frame by frame
    for (i, frame) in samples.chunks_exact(caps.frame_size).enumerate() {
        let prob = vad.process(frame, target_rate).expect("VAD failed");
        let time_ms = i as f64 * caps.frame_duration_ms as f64;
        let bar = "#".repeat((prob * 40.0) as usize);
        println!("{time_ms:8.0}ms  {prob:.3}  {bar}");
    }
}
