use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::audio_source::{self, AudioDevice, AudioFrame};
use crate::pipeline;
use crate::session::VadConfig;

/// Messages sent from the client to the server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    ListDevices,
    ListBackends,
    StartRecording {
        device_index: usize,
        sample_rate: u32,
    },
    StopRecording,
    LoadFile {
        path: String,
    },
    SetConfigs {
        configs: Vec<VadConfig>,
    },
}

/// Messages sent from the server to the client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Devices {
        devices: Vec<AudioDevice>,
    },
    Backends {
        backends: std::collections::HashMap<String, Vec<pipeline::ParamInfo>>,
    },
    Audio {
        timestamp_ms: f64,
        samples: Vec<i16>,
    },
    Vad {
        config_id: String,
        timestamp_ms: f64,
        probability: f32,
    },
    Done,
    Error {
        message: String,
    },
}

fn send_msg(msg: &ServerMessage) -> Message {
    Message::Text(serde_json::to_string(msg).unwrap().into())
}

/// Handle a WebSocket connection.
pub async fn handle_ws(socket: WebSocket) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let mut configs: Vec<VadConfig> = Vec::new();
    let mut stop_tx: Option<tokio::sync::oneshot::Sender<()>> = None;

    let frame_duration_ms: u32 = 20;

    while let Some(Ok(msg)) = ws_rx.next().await {
        let Message::Text(text) = msg else {
            continue;
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(msg) => msg,
            Err(e) => {
                let _ = ws_tx
                    .send(send_msg(&ServerMessage::Error {
                        message: format!("invalid message: {e}"),
                    }))
                    .await;
                continue;
            }
        };

        match client_msg {
            ClientMessage::ListDevices => {
                let devices = audio_source::list_devices();
                let _ = ws_tx
                    .send(send_msg(&ServerMessage::Devices { devices }))
                    .await;
            }

            ClientMessage::ListBackends => {
                let backends = pipeline::available_backends();
                let _ = ws_tx
                    .send(send_msg(&ServerMessage::Backends { backends }))
                    .await;
            }

            ClientMessage::SetConfigs {
                configs: new_configs,
            } => {
                tracing::info!(count = new_configs.len(), "configs updated");
                configs = new_configs;
            }

            ClientMessage::StartRecording {
                device_index,
                sample_rate,
            } => {
                // Stop any existing capture
                if let Some(tx) = stop_tx.take() {
                    let _ = tx.send(());
                }

                match audio_source::start_capture(device_index, sample_rate, frame_duration_ms) {
                    Ok(capture) => {
                        stop_tx = Some(capture.stop);
                        let audio_tx = capture.tx;

                        // Start the pipeline
                        let pipeline_rx = audio_tx.subscribe();
                        let mut result_rx =
                            pipeline::run_pipeline(configs.clone(), pipeline_rx, sample_rate);

                        // Collect messages from both audio and pipeline into one channel
                        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::channel::<ServerMessage>(512);

                        // Forward audio frames
                        let msg_tx_audio = msg_tx.clone();
                        let mut audio_rx = capture.rx;
                        tokio::spawn(async move {
                            while let Ok(frame) = audio_rx.recv().await {
                                let msg = ServerMessage::Audio {
                                    timestamp_ms: frame.timestamp_ms,
                                    samples: frame.samples,
                                };
                                if msg_tx_audio.send(msg).await.is_err() {
                                    break;
                                }
                            }
                        });

                        // Forward VAD results
                        let msg_tx_vad = msg_tx;
                        tokio::spawn(async move {
                            while let Some(result) = result_rx.recv().await {
                                let msg = ServerMessage::Vad {
                                    config_id: result.config_id,
                                    timestamp_ms: result.timestamp_ms,
                                    probability: result.probability,
                                };
                                if msg_tx_vad.send(msg).await.is_err() {
                                    break;
                                }
                            }
                        });

                        // Stream messages to the client until stop
                        loop {
                            tokio::select! {
                                Some(msg) = msg_rx.recv() => {
                                    if ws_tx.send(send_msg(&msg)).await.is_err() {
                                        break;
                                    }
                                }
                                incoming = ws_rx.next() => {
                                    match incoming {
                                        Some(Ok(Message::Text(text))) => {
                                            if let Ok(ClientMessage::StopRecording) = serde_json::from_str(&text) {
                                                if let Some(tx) = stop_tx.take() {
                                                    let _ = tx.send(());
                                                }
                                                tracing::info!("recording stopped");
                                                break;
                                            }
                                        }
                                        None | Some(Err(_)) => break, // client disconnected
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(send_msg(&ServerMessage::Error { message: e }))
                            .await;
                    }
                }
            }

            ClientMessage::StopRecording => {
                if let Some(tx) = stop_tx.take() {
                    let _ = tx.send(());
                }
                tracing::info!("recording stopped");
            }

            ClientMessage::LoadFile { path } => {
                let (audio_tx, _) = broadcast::channel::<AudioFrame>(256);

                let pipeline_rx = audio_tx.subscribe();
                let mut forward_rx = audio_tx.subscribe();

                let (msg_tx, mut msg_rx) = tokio::sync::mpsc::channel::<ServerMessage>(512);

                // Forward audio frames to client
                let msg_tx_audio = msg_tx.clone();
                tokio::spawn(async move {
                    while let Ok(frame) = forward_rx.recv().await {
                        let msg = ServerMessage::Audio {
                            timestamp_ms: frame.timestamp_ms,
                            samples: frame.samples,
                        };
                        if msg_tx_audio.send(msg).await.is_err() {
                            break;
                        }
                    }
                });

                // Play file and run pipeline
                let configs_clone = configs.clone();
                let msg_tx_done = msg_tx;
                tokio::spawn(async move {
                    let file_path = std::path::Path::new(&path);
                    match audio_source::play_file(file_path, frame_duration_ms, audio_tx).await {
                        Ok(sample_rate) => {
                            let mut result_rx =
                                pipeline::run_pipeline(configs_clone, pipeline_rx, sample_rate);
                            while let Some(result) = result_rx.recv().await {
                                let msg = ServerMessage::Vad {
                                    config_id: result.config_id,
                                    timestamp_ms: result.timestamp_ms,
                                    probability: result.probability,
                                };
                                if msg_tx_done.send(msg).await.is_err() {
                                    break;
                                }
                            }
                            let _ = msg_tx_done.send(ServerMessage::Done).await;
                        }
                        Err(e) => {
                            let _ = msg_tx_done.send(ServerMessage::Error { message: e }).await;
                        }
                    }
                });

                // Stream results to client
                while let Some(msg) = msg_rx.recv().await {
                    let is_terminal =
                        matches!(msg, ServerMessage::Done | ServerMessage::Error { .. });
                    if ws_tx.send(send_msg(&msg)).await.is_err() {
                        break;
                    }
                    if is_terminal {
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    if let Some(tx) = stop_tx.take() {
        let _ = tx.send(());
    }
}
