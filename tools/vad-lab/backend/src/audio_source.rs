use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::broadcast;

/// Sample rates supported by the VAD backends.
const SUPPORTED_SAMPLE_RATES: &[u32] = &[8000, 16000, 32000, 48000];

/// Default target sample rate when resampling is needed.
/// Using 48kHz to support both VAD backends and noise suppression (which requires 48kHz).
const DEFAULT_TARGET_SAMPLE_RATE: u32 = 48000;

/// Information about an available audio input device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDevice {
    /// Device index (used to select the device).
    pub index: usize,
    /// Human-readable device name.
    pub name: String,
}

/// An audio frame with its timestamp.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Timestamp in milliseconds from the start of the stream.
    pub timestamp_ms: f64,
    /// Audio samples as 16-bit signed integers, mono.
    pub samples: Vec<i16>,
}

/// Returns true if the sample rate is supported by VAD backends.
fn is_supported_sample_rate(rate: u32) -> bool {
    SUPPORTED_SAMPLE_RATES.contains(&rate)
}

/// Find the best target sample rate for resampling.
/// Prefers 16000 Hz as it's the most common for speech processing.
fn get_target_sample_rate() -> u32 {
    DEFAULT_TARGET_SAMPLE_RATE
}

/// List available audio input devices.
pub fn list_devices() -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    if let Ok(input_devices) = host.input_devices() {
        for (index, device) in input_devices.enumerate() {
            let name = device.name().unwrap_or_else(|_| format!("Device {index}"));
            devices.push(AudioDevice { index, name });
        }
    }

    devices
}

/// Result of starting an audio capture.
pub struct CaptureResult {
    /// Receiver for audio frames.
    pub rx: broadcast::Receiver<AudioFrame>,
    /// Sender to subscribe additional receivers.
    pub tx: broadcast::Sender<AudioFrame>,
    /// Send `()` to stop the capture.
    pub stop: tokio::sync::oneshot::Sender<()>,
    /// Actual sample rate of the captured audio.
    pub sample_rate: u32,
}

/// Start capturing audio from a device.
///
/// Uses the device's default input config (native sample rate and channels),
/// then downmixes to mono. Returns the actual sample rate in `CaptureResult`.
/// The capture runs on a dedicated thread (cpal::Stream is !Send on macOS).
pub fn start_capture(
    device_index: usize,
    frame_duration_ms: u32,
) -> Result<CaptureResult, String> {
    let (tx, rx) = broadcast::channel::<AudioFrame>(256);
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<u32, String>>();

    let tx_clone = tx.clone();

    // Build and run the stream entirely on a dedicated thread
    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.input_devices() {
            Ok(mut devices) => match devices.nth(device_index) {
                Some(d) => d,
                None => {
                    let _ = ready_tx.send(Err(format!("device index {device_index} not found")));
                    return;
                }
            },
            Err(e) => {
                let _ = ready_tx.send(Err(format!("failed to enumerate devices: {e}")));
                return;
            }
        };

        let default_config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("failed to get default config: {e}")));
                return;
            }
        };

        let sample_rate = default_config.sample_rate().0;
        let channels = default_config.channels() as usize;

        let config = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Determine if resampling is needed
        let needs_resampling = !is_supported_sample_rate(sample_rate);
        let effective_sample_rate = if needs_resampling {
            get_target_sample_rate()
        } else {
            sample_rate
        };

        tracing::info!(
            device_sample_rate = sample_rate,
            effective_sample_rate,
            channels,
            needs_resampling,
            "capturing with device default config"
        );

        // Create resampler if needed (outside the closure to avoid borrowing issues)
        let resampler: Option<std::sync::Mutex<FftFixedIn<f64>>> = if needs_resampling {
            // Use a chunk size that works well for real-time audio
            // rubato works with f64 samples
            let chunk_size = 1024;
            match FftFixedIn::<f64>::new(
                sample_rate as usize,
                effective_sample_rate as usize,
                chunk_size,
                2, // sub_chunks for lower latency
                1, // mono
            ) {
                Ok(r) => {
                    tracing::info!(
                        from = sample_rate,
                        to = effective_sample_rate,
                        "created resampler"
                    );
                    Some(std::sync::Mutex::new(r))
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("failed to create resampler: {e}")));
                    return;
                }
            }
        } else {
            None
        };

        let samples_per_frame =
            (effective_sample_rate as usize * frame_duration_ms as usize) / 1000;
        let effective_sr = effective_sample_rate as f64;
        let mut output_buffer = Vec::with_capacity(samples_per_frame);
        let mut total_output_samples: u64 = 0;

        // Buffer for accumulating input samples before resampling
        let mut resample_input_buffer: Vec<f64> = Vec::new();

        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix to mono: average all channels per sample
                let mono_samples: Vec<f32> = data
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();

                // Either resample or use directly
                let processed_samples: Vec<i16> = if let Some(ref resampler_mutex) = resampler {
                    // Accumulate samples for resampling
                    resample_input_buffer.extend(mono_samples.iter().map(|&s| s as f64));

                    let mut resampler = resampler_mutex.lock().unwrap();
                    let input_frames_needed = resampler.input_frames_next();

                    let mut resampled = Vec::new();

                    // Process complete chunks
                    while resample_input_buffer.len() >= input_frames_needed {
                        let chunk: Vec<f64> =
                            resample_input_buffer.drain(..input_frames_needed).collect();
                        match resampler.process(&[chunk], None) {
                            Ok(output) => {
                                if !output.is_empty() {
                                    resampled.extend(
                                        output[0]
                                            .iter()
                                            .map(|&s| (s * i16::MAX as f64) as i16),
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::error!("resampling error: {e}");
                            }
                        }
                    }

                    resampled
                } else {
                    // No resampling needed, convert directly to i16
                    mono_samples
                        .iter()
                        .map(|&s| (s * i16::MAX as f32) as i16)
                        .collect()
                };

                // Buffer samples into frames
                for sample in processed_samples {
                    output_buffer.push(sample);

                    if output_buffer.len() >= samples_per_frame {
                        let timestamp_ms = (total_output_samples as f64 / effective_sr) * 1000.0;
                        let frame = AudioFrame {
                            timestamp_ms,
                            samples: output_buffer.clone(),
                        };
                        let _ = tx_clone.send(frame);
                        total_output_samples += output_buffer.len() as u64;
                        output_buffer.clear();
                    }
                }
            },
            |err| {
                tracing::error!("audio capture error: {err}");
            },
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("failed to build input stream: {e}")));
                return;
            }
        };

        if let Err(e) = stream.play() {
            let _ = ready_tx.send(Err(format!("failed to start stream: {e}")));
            return;
        }

        let _ = ready_tx.send(Ok(effective_sample_rate));

        // Block until stop signal, keeping stream alive
        let _ = stop_rx.blocking_recv();
        drop(stream);
    });

    // Wait for the stream to be ready
    match ready_rx.recv() {
        Ok(Ok(sample_rate)) => Ok(CaptureResult {
            rx,
            tx,
            stop: stop_tx,
            sample_rate,
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("capture thread exited unexpectedly".into()),
    }
}

/// Load a WAV file and produce audio frames, simulating real-time playback.
pub async fn play_file(
    path: &Path,
    frame_duration_ms: u32,
    tx: broadcast::Sender<AudioFrame>,
) -> Result<u32, String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("failed to open WAV file: {e}"))?;
    let spec = reader.spec();
    let file_sample_rate = spec.sample_rate;

    let all_samples_i16: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .map(|s| (s * i16::MAX as f32) as i16)
            .collect(),
    };

    // Determine if resampling is needed
    let needs_resampling = !is_supported_sample_rate(file_sample_rate);
    let effective_sample_rate = if needs_resampling {
        get_target_sample_rate()
    } else {
        file_sample_rate
    };

    tracing::info!(
        file_sample_rate,
        effective_sample_rate,
        needs_resampling,
        "loading WAV file"
    );

    // Resample if needed
    let processed_samples: Vec<i16> = if needs_resampling {
        // Convert to f64 for rubato
        let samples_f64: Vec<f64> = all_samples_i16
            .iter()
            .map(|&s| s as f64 / i16::MAX as f64)
            .collect();

        // Use a reasonable chunk size for batch processing
        let chunk_size = 1024;
        let mut resampler = FftFixedIn::<f64>::new(
            file_sample_rate as usize,
            effective_sample_rate as usize,
            chunk_size,
            2,
            1, // mono
        )
        .map_err(|e| format!("failed to create resampler: {e}"))?;

        let mut resampled = Vec::new();
        let mut offset = 0;

        while offset < samples_f64.len() {
            let input_frames_needed = resampler.input_frames_next();
            let end = (offset + input_frames_needed).min(samples_f64.len());
            let mut chunk: Vec<f64> = samples_f64[offset..end].to_vec();

            // Pad with zeros if we don't have enough samples
            if chunk.len() < input_frames_needed {
                chunk.resize(input_frames_needed, 0.0);
            }

            match resampler.process(&[chunk], None) {
                Ok(output) => {
                    if !output.is_empty() {
                        resampled.extend(
                            output[0]
                                .iter()
                                .map(|&s| (s * i16::MAX as f64) as i16),
                        );
                    }
                }
                Err(e) => {
                    tracing::error!("resampling error: {e}");
                }
            }

            offset = end;
        }

        resampled
    } else {
        all_samples_i16
    };

    let samples_per_frame =
        (effective_sample_rate as usize * frame_duration_ms as usize) / 1000;
    let frame_duration = tokio::time::Duration::from_millis(frame_duration_ms as u64);
    let mut total_samples: u64 = 0;
    let sr = effective_sample_rate as f64;

    for chunk in processed_samples.chunks(samples_per_frame) {
        let timestamp_ms = (total_samples as f64 / sr) * 1000.0;
        let frame = AudioFrame {
            timestamp_ms,
            samples: chunk.to_vec(),
        };
        let _ = tx.send(frame);
        total_samples += chunk.len() as u64;
        tokio::time::sleep(frame_duration).await;
    }

    Ok(effective_sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_devices_does_not_panic() {
        let _devices = list_devices();
    }
}
