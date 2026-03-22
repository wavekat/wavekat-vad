//! VAD accuracy validation against the TEN-VAD testset.
//!
//! Downloads 30 labeled audio files from the TEN-VAD repository and evaluates
//! all enabled backends on precision, recall, F1 score, and inference speed.
//!
//! Run with:
//! ```sh
//! cargo test --release -p wavekat-vad --features webrtc,silero,ten-vad \
//!     -- --ignored accuracy_report --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use wavekat_vad::VoiceActivityDetector;

const TESTSET_URL: &str = "https://github.com/TEN-framework/ten-vad/raw/main/testset";
const NUM_FILES: usize = 30;
const THRESHOLD: f32 = 0.5;

// ---------------------------------------------------------------------------
// Testset download & caching
// ---------------------------------------------------------------------------

fn testset_dir() -> PathBuf {
    // Store in workspace target/ so files persist across runs
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("could not find workspace root");
    workspace_root.join("target").join("testset")
}

fn download_file(url: &str, dest: &Path) {
    if dest.exists() {
        return;
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    eprintln!("Downloading {url}");
    let output = std::process::Command::new("curl")
        .args(["-sSL", "--fail", "-o"])
        .arg(dest.as_os_str())
        .arg(url)
        .output()
        .expect("curl not found — install curl to run accuracy tests");
    assert!(
        output.status.success(),
        "Failed to download {url}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn download_testset() -> PathBuf {
    let dir = testset_dir();
    for i in 1..=NUM_FILES {
        let name = format!("testset-audio-{i:02}");
        download_file(
            &format!("{TESTSET_URL}/{name}.wav"),
            &dir.join(format!("{name}.wav")),
        );
        download_file(
            &format!("{TESTSET_URL}/{name}.scv"),
            &dir.join(format!("{name}.scv")),
        );
    }
    dir
}

// ---------------------------------------------------------------------------
// Ground-truth label parsing (.scv format)
// ---------------------------------------------------------------------------

struct Segment {
    start: f32,
    end: f32,
    is_speech: bool,
}

/// Parse a `.scv` label file into time segments.
///
/// Format: `filename,start1,end1,label1,start2,end2,label2,...`
/// where label is 0 (silence) or 1 (speech).
fn parse_scv(path: &Path) -> Vec<Segment> {
    let content = std::fs::read_to_string(path).unwrap();
    let parts: Vec<&str> = content.trim().split(',').map(|s| s.trim()).collect();
    let mut segments = Vec::new();
    let mut i = 1; // skip filename field
    while i + 2 < parts.len() {
        segments.push(Segment {
            start: parts[i].parse().unwrap(),
            end: parts[i + 1].parse().unwrap(),
            is_speech: parts[i + 2] == "1",
        });
        i += 3;
    }
    segments
}

/// Look up the ground-truth label at a given time.
fn label_at_time(segments: &[Segment], time: f32) -> bool {
    segments
        .iter()
        .any(|s| time >= s.start && time < s.end && s.is_speech)
}

// ---------------------------------------------------------------------------
// WAV reading
// ---------------------------------------------------------------------------

fn read_wav_i16(path: &Path) -> (Vec<i16>, u32) {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| {
                let s = s.unwrap();
                (s * 32767.0).clamp(-32768.0, 32767.0) as i16
            })
            .collect(),
    };
    (samples, spec.sample_rate)
}

// ---------------------------------------------------------------------------
// Backend evaluation
// ---------------------------------------------------------------------------

struct BackendResult {
    name: String,
    precision: f32,
    recall: f32,
    f1: f32,
    avg_frame_us: f64,
    frame_size: usize,
    frame_ms: u32,
}

fn evaluate_backend(
    name: &str,
    vad: &mut dyn VoiceActivityDetector,
    testset_dir: &Path,
) -> BackendResult {
    let caps = vad.capabilities();
    let frame_size = caps.frame_size;
    let sample_rate = caps.sample_rate;
    let frame_duration = frame_size as f32 / sample_rate as f32;

    let mut total_tp: u64 = 0;
    let mut total_fp: u64 = 0;
    let mut total_fn: u64 = 0;
    let mut total_time = Duration::ZERO;
    let mut total_frames: u64 = 0;

    for i in 1..=NUM_FILES {
        let file_name = format!("testset-audio-{i:02}");
        let wav_path = testset_dir.join(format!("{file_name}.wav"));
        let scv_path = testset_dir.join(format!("{file_name}.scv"));

        let (samples, wav_rate) = read_wav_i16(&wav_path);
        assert_eq!(
            wav_rate, sample_rate,
            "{file_name}: expected {sample_rate} Hz, got {wav_rate} Hz"
        );

        let segments = parse_scv(&scv_path);
        vad.reset();

        for (frame_idx, chunk) in samples.chunks_exact(frame_size).enumerate() {
            let start = Instant::now();
            let prob = vad.process(chunk, sample_rate).unwrap();
            total_time += start.elapsed();
            total_frames += 1;

            let center_time = (frame_idx as f32 + 0.5) * frame_duration;
            let predicted = prob >= THRESHOLD;
            let actual = label_at_time(&segments, center_time);

            match (predicted, actual) {
                (true, true) => total_tp += 1,
                (true, false) => total_fp += 1,
                (false, true) => total_fn += 1,
                (false, false) => {}
            }
        }
    }

    let precision = if total_tp + total_fp > 0 {
        total_tp as f32 / (total_tp + total_fp) as f32
    } else {
        0.0
    };
    let recall = if total_tp + total_fn > 0 {
        total_tp as f32 / (total_tp + total_fn) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_frame_us = if total_frames > 0 {
        total_time.as_secs_f64() * 1_000_000.0 / total_frames as f64
    } else {
        0.0
    };

    eprintln!(
        "{name}: P={precision:.3} R={recall:.3} F1={f1:.3} \
         frames={total_frames} avg={avg_frame_us:.1}µs"
    );

    BackendResult {
        name: name.to_string(),
        precision,
        recall,
        f1,
        avg_frame_us,
        frame_size,
        frame_ms: caps.frame_duration_ms,
    }
}

// ---------------------------------------------------------------------------
// Test entry point
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn accuracy_report() {
    let testset_dir = download_testset();
    let mut results = Vec::new();

    #[cfg(feature = "webrtc")]
    {
        use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};
        let mut vad = WebRtcVad::new(16000, WebRtcVadMode::Quality).unwrap();
        results.push(evaluate_backend("WebRTC", &mut vad, &testset_dir));
    }

    #[cfg(feature = "silero")]
    {
        use wavekat_vad::backends::silero::SileroVad;
        let mut vad = SileroVad::new(16000).unwrap();
        results.push(evaluate_backend("Silero", &mut vad, &testset_dir));
    }

    #[cfg(feature = "ten-vad")]
    {
        use wavekat_vad::backends::ten_vad::TenVad;
        let mut vad = TenVad::new().unwrap();
        results.push(evaluate_backend("TEN-VAD", &mut vad, &testset_dir));
    }

    assert!(
        !results.is_empty(),
        "No backends enabled — use --features webrtc,silero,ten-vad"
    );

    // Print markdown table (CI parses this to update README)
    println!();
    println!("| Backend | Precision | Recall | F1 Score | Frame Size | Avg Inference |");
    println!("|---------|-----------|--------|----------|------------|---------------|");
    for r in &results {
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {} ({} ms) | {:.1} µs |",
            r.name, r.precision, r.recall, r.f1, r.frame_size, r.frame_ms, r.avg_frame_us,
        );
    }
    println!();

    // Assert minimum F1 thresholds — fail the test on accuracy regression
    for r in &results {
        let min_f1 = match r.name.as_str() {
            "WebRTC" => 0.80,
            _ => 0.85,
        };
        assert!(
            r.f1 > min_f1,
            "{} F1 score {:.3} below minimum threshold {:.3}",
            r.name,
            r.f1,
            min_f1,
        );
    }
}
