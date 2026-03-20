//! Detect speech in a WAV file using any available backend.
//!
//! ```sh
//! # WebRTC (default feature):
//! cargo run --example detect_speech -- path/to/audio.wav
//!
//! # Silero:
//! cargo run --example detect_speech --features silero -- --backend silero path/to/audio.wav
//!
//! # TEN-VAD:
//! cargo run --example detect_speech --features ten-vad -- --backend ten-vad path/to/audio.wav
//! ```

use std::env;
use wavekat_vad::{FrameAdapter, VoiceActivityDetector};

fn create_vad(backend: &str) -> Box<dyn VoiceActivityDetector> {
    match backend {
        #[cfg(feature = "webrtc")]
        "webrtc" => {
            use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
            Box::new(
                WebRtcVad::new(16000, WebRtcVadMode::Quality).expect("failed to create WebRTC VAD"),
            )
        }
        #[cfg(feature = "silero")]
        "silero" => {
            use wavekat_vad::backends::silero::SileroVad;
            Box::new(SileroVad::new(16000).expect("failed to create Silero VAD"))
        }
        #[cfg(feature = "ten-vad")]
        "ten-vad" => {
            use wavekat_vad::backends::ten_vad::TenVad;
            Box::new(TenVad::new().expect("failed to create TEN-VAD"))
        }
        other => {
            eprintln!("Unknown or disabled backend: {other}");
            eprintln!("Available backends:");
            #[cfg(feature = "webrtc")]
            eprintln!("  webrtc  (default)");
            #[cfg(feature = "silero")]
            eprintln!("  silero");
            #[cfg(feature = "ten-vad")]
            eprintln!("  ten-vad");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut backend = "webrtc";
    let mut wav_path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" | "-b" => {
                i += 1;
                backend = args.get(i).map(|s| s.as_str()).unwrap_or_else(|| {
                    eprintln!("--backend requires a value");
                    std::process::exit(1);
                });
            }
            arg if !arg.starts_with('-') => {
                wav_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unknown flag: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let wav_path = wav_path.unwrap_or_else(|| {
        eprintln!("Usage: detect_speech [--backend webrtc|silero|ten-vad] <wav-file>");
        std::process::exit(1);
    });

    // Open WAV file
    let mut reader = hound::WavReader::open(&wav_path).expect("failed to open WAV file");
    let spec = reader.spec();
    println!("File: {wav_path}");
    println!(
        "  channels: {}, sample rate: {} Hz, bits: {}",
        spec.channels, spec.sample_rate, spec.bits_per_sample
    );

    // Read samples (first channel only)
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.expect("failed to read sample"))
        .collect();

    // Resample to 16kHz if needed
    let target_rate = 16000;
    let samples = if spec.sample_rate != target_rate {
        println!("  resampling {}Hz -> {}Hz", spec.sample_rate, target_rate);
        use wavekat_vad::preprocessing::AudioResampler;
        let mut resampler =
            AudioResampler::new(spec.sample_rate, target_rate).expect("failed to create resampler");
        resampler.process(&samples)
    } else {
        samples
    };

    let duration = samples.len() as f64 / target_rate as f64;
    println!(
        "  duration: {duration:.2}s ({} samples at {target_rate}Hz)",
        samples.len()
    );
    println!("  backend: {backend}");
    println!();

    // Create VAD with FrameAdapter for automatic frame buffering
    let vad = create_vad(backend);
    let caps = vad.capabilities();
    println!(
        "Frame: {} samples ({}ms)",
        caps.frame_size, caps.frame_duration_ms
    );
    println!();

    let mut adapter = FrameAdapter::new(vad);

    // Process in 20ms chunks (arbitrary, adapter handles buffering)
    let chunk_size = target_rate as usize / 50; // 20ms
    let mut time_ms = 0.0;
    let step_ms = chunk_size as f64 * 1000.0 / target_rate as f64;

    for chunk in samples.chunks(chunk_size) {
        let results = adapter.process_all(chunk, target_rate).unwrap();
        for prob in results {
            let bar = "#".repeat((prob * 40.0) as usize);
            let label = if prob > 0.5 { " SPEECH" } else { "" };
            println!("{time_ms:8.0}ms  {prob:.3}  {bar}{label}");
        }
        time_ms += step_ms;
    }
}
