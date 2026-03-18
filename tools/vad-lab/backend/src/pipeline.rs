use crate::audio_source::AudioFrame;
use crate::session::VadConfig;
use serde::Serialize;
use std::collections::HashMap;
use tokio::sync::{broadcast, mpsc};
use wavekat_vad::VoiceActivityDetector;

/// A VAD result from the pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    /// Config ID that produced this result.
    pub config_id: String,
    /// Timestamp in milliseconds.
    pub timestamp_ms: f64,
    /// Speech probability (0.0 - 1.0).
    pub probability: f32,
}

/// Run the VAD pipeline: fan out audio frames to multiple VAD configs.
///
/// Returns an mpsc receiver that yields results from all configs.
pub fn run_pipeline(
    configs: Vec<VadConfig>,
    mut audio_rx: broadcast::Receiver<AudioFrame>,
    sample_rate: u32,
) -> mpsc::Receiver<PipelineResult> {
    let (result_tx, result_rx) = mpsc::channel::<PipelineResult>(1024);

    tokio::spawn(async move {
        let mut detectors: Vec<(String, Box<dyn VoiceActivityDetector>)> = Vec::new();

        for config in &configs {
            match create_detector(config, sample_rate) {
                Ok(detector) => {
                    detectors.push((config.id.clone(), detector));
                }
                Err(e) => {
                    tracing::error!(config_id = %config.id, "failed to create detector: {e}");
                }
            }
        }

        while let Ok(frame) = audio_rx.recv().await {
            for (config_id, detector) in &mut detectors {
                match detector.process(&frame.samples, sample_rate) {
                    Ok(probability) => {
                        let result = PipelineResult {
                            config_id: config_id.clone(),
                            timestamp_ms: frame.timestamp_ms,
                            probability,
                        };
                        if result_tx.send(result).await.is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            config_id = %config_id,
                            "VAD processing error: {e}"
                        );
                    }
                }
            }
        }
    });

    result_rx
}

/// Create a VAD detector from a config.
fn create_detector(
    config: &VadConfig,
    sample_rate: u32,
) -> Result<Box<dyn VoiceActivityDetector>, String> {
    match config.backend.as_str() {
        "webrtc-vad" => {
            use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

            let mode_str = config
                .params
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("0 - quality");

            // Strip "N - " prefix if present (e.g. "2 - aggressive" -> "aggressive")
            let mode_key = mode_str
                .split_once(" - ")
                .map_or(mode_str, |(_, name)| name);

            let mode = match mode_key {
                "quality" => WebRtcVadMode::Quality,
                "low_bitrate" => WebRtcVadMode::LowBitrate,
                "aggressive" => WebRtcVadMode::Aggressive,
                "very_aggressive" => WebRtcVadMode::VeryAggressive,
                other => return Err(format!("unknown webrtc mode: {other}")),
            };

            let vad = WebRtcVad::new(sample_rate, mode)
                .map_err(|e| format!("failed to create WebRTC VAD: {e}"))?;
            Ok(Box::new(vad))
        }
        other => Err(format!("unknown backend: {other}")),
    }
}

/// Return the list of available backends and their configurable parameters.
pub fn available_backends() -> HashMap<String, Vec<ParamInfo>> {
    let mut backends = HashMap::new();

    backends.insert(
        "webrtc-vad".to_string(),
        vec![ParamInfo {
            name: "mode".to_string(),
            description: "Aggressiveness mode".to_string(),
            param_type: ParamType::Select(vec![
                "0 - quality".to_string(),
                "1 - low_bitrate".to_string(),
                "2 - aggressive".to_string(),
                "3 - very_aggressive".to_string(),
            ]),
            default: serde_json::json!("0 - quality"),
        }],
    );

    backends
}

/// Description of a configurable parameter.
#[derive(Debug, Clone, Serialize)]
pub struct ParamInfo {
    /// Parameter name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Parameter type and constraints.
    pub param_type: ParamType,
    /// Default value.
    pub default: serde_json::Value,
}

/// Type of a configurable parameter.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "options")]
pub enum ParamType {
    /// Select from a list of options.
    Select(Vec<String>),
    /// Float value with min/max range.
    #[allow(dead_code)]
    Float { min: f64, max: f64 },
    /// Integer value with min/max range.
    #[allow(dead_code)]
    Int { min: i64, max: i64 },
}
