use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::broadcast;

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

        tracing::info!(
            sample_rate,
            channels,
            "capturing with device default config"
        );

        let samples_per_frame = (sample_rate as usize * frame_duration_ms as usize) / 1000;
        let sr = sample_rate as f64;
        let mut buffer = Vec::with_capacity(samples_per_frame);
        let mut total_samples: u64 = 0;

        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix to mono: average all channels per sample
                for chunk in data.chunks(channels) {
                    let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                    let sample_i16 = (mono * i16::MAX as f32) as i16;
                    buffer.push(sample_i16);

                    if buffer.len() >= samples_per_frame {
                        let timestamp_ms = (total_samples as f64 / sr) * 1000.0;
                        let frame = AudioFrame {
                            timestamp_ms,
                            samples: buffer.clone(),
                        };
                        let _ = tx_clone.send(frame);
                        total_samples += buffer.len() as u64;
                        buffer.clear();
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

        let _ = ready_tx.send(Ok(sample_rate));

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
    let sample_rate = spec.sample_rate;
    let samples_per_frame = (sample_rate as usize * frame_duration_ms as usize) / 1000;

    let all_samples: Vec<i16> = match spec.sample_format {
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

    let frame_duration = tokio::time::Duration::from_millis(frame_duration_ms as u64);
    let mut total_samples: u64 = 0;
    let sr = sample_rate as f64;

    for chunk in all_samples.chunks(samples_per_frame) {
        let timestamp_ms = (total_samples as f64 / sr) * 1000.0;
        let frame = AudioFrame {
            timestamp_ms,
            samples: chunk.to_vec(),
        };
        let _ = tx.send(frame);
        total_samples += chunk.len() as u64;
        tokio::time::sleep(frame_duration).await;
    }

    Ok(sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_devices_does_not_panic() {
        let _devices = list_devices();
    }
}
