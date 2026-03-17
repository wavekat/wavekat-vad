# Implementation Plan

## Overview

The goal is to deliver the best Rust VAD crate. Before we can finalize the API and publish, we need to experiment with different VAD backends and understand their trade-offs. The plan has two tracks that proceed in parallel:

- **Track A**: Core library crate (`crates/wavekat-vad`) — the product
- **Track B**: Dev tool (`tools/vad-lab`) — helps us experiment and make informed decisions

## Step 1: Scaffold Workspace

Set up the Cargo workspace with both the library crate and vad-lab tool.

- [ ] Create workspace `Cargo.toml`
- [ ] Create `crates/wavekat-vad/` with basic structure (lib.rs, error.rs, frame.rs)
- [ ] Create `tools/vad-lab/backend/` with skeleton axum server
- [ ] Create `tools/vad-lab/frontend/` with React app (Vite + TypeScript)
- [ ] Verify `cargo build` works for the whole workspace

## Step 2: Core Library — Trait + First Backend

Implement the foundation of the library crate.

- [ ] Define `VoiceActivityDetector` trait in `lib.rs`
- [ ] Implement `VadError` with `thiserror` in `error.rs`
- [ ] Implement `AudioFrame` type in `frame.rs`
- [ ] Add `webrtc-vad` backend behind `webrtc` feature flag
- [ ] Unit tests for trait, error types, and webrtc backend
- [ ] Verify: `cargo test`, `cargo clippy`, `cargo fmt`

## Step 3: vad-lab Backend — Audio Pipeline

Build the Rust server that captures audio and runs VAD.

- [ ] Mic device enumeration and capture via `cpal`
- [ ] WAV file loading via `hound`
- [ ] Audio pipeline: fan-out frames to N VAD config instances
- [ ] Each VAD config runs in its own tokio task
- [ ] axum WebSocket server streaming audio + VAD results
- [ ] REST endpoints for config/session CRUD
- [ ] Save recordings as WAV files
- [ ] Session storage (configs + results as JSON)

### WebSocket Protocol

```
Server → Client:
  { type: "devices", devices: [{ id, name, sample_rates }] }
  { type: "audio", timestamp, samples: [...] }
  { type: "vad", timestamp, config_id, probability }
  { type: "done" }

Client → Server:
  { type: "list_devices" }
  { type: "start_recording", device_id, sample_rate }
  { type: "stop_recording" }
  { type: "load_file", path: "..." }
  { type: "set_configs", configs: [...] }
```

## Step 4: vad-lab Frontend — React UI

Build the browser-based visualization and control interface.

- [ ] Project setup: Vite + React + TypeScript + Tailwind + shadcn/ui
- [ ] WebSocket client connecting to backend
- [ ] **Waveform display**: canvas-based, renders audio samples in real-time (live mode) or full timeline (file mode)
- [ ] **VAD timeline overlay**: per-config speech probability rendered on top of / below the waveform
- [ ] **Config panel**: add/remove/edit VAD configurations (backend selector, parameter sliders/inputs)
- [ ] **Device selector**: dropdown of available mic devices from server
- [ ] **Transport controls**: record, stop, load file
- [ ] **Session manager**: save/load sessions (configs + results)
- [ ] Embed built frontend assets into Rust binary via `rust-embed`

### UI Layout (rough)

```
┌─────────────────────────────────────────────────┐
│  [Device: ▼ Built-in Mic]  [Record] [Stop]      │
│  [Load File...]  [Save Session]                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──── Waveform ────────────────────────────┐   │
│  │ ▁▂▃▅▇▇▅▃▂▁▁▁▂▃▅▇▇▅▃▂▁                  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──── webrtc-mode3 ───────────────────────┐    │
│  │ ████████░░░░████████░░░░                 │    │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──── silero-default ─────────────────────┐    │
│  │ ██████████░░██████████░░                 │    │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──── silero-sensitive ───────────────────┐    │
│  │ ████████████████████████░               │    │
│  └──────────────────────────────────────────┘   │
│                                                  │
├─────────────────────────────────────────────────┤
│  Config Panel                                    │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐    │
│  │webrtc-mode3│ │silero-def  │ │silero-sen│    │
│  │Backend: wrt│ │Backend: sil│ │Backend: s│    │
│  │Mode: 3     │ │Threshold:.5│ │Threshold:│    │
│  │            │ │            │ │.3        │    │
│  └────────────┘ └────────────┘ └──────────┘    │
│                                  [+ Add Config]  │
│                                                  │
│  UI components: shadcn/ui (buttons, selects,     │
│  sliders, cards). Custom canvas for waveform     │
│  and VAD timeline rendering.                     │
└─────────────────────────────────────────────────┘
```

## Step 5: Add Silero Backend

- [ ] Add `silero-vad` backend behind `silero` feature flag in the library
- [ ] Download/bundle Silero ONNX model
- [ ] Implement `VoiceActivityDetector` trait for Silero
- [ ] Use vad-lab to compare webrtc vs silero side-by-side

## Step 6: Experiment and Iterate

- [ ] Collect test audio: speech samples, silence, noisy environments
- [ ] Run side-by-side comparisons using vad-lab
- [ ] Tune parameters for each backend
- [ ] Add benchmarks (`criterion`) for latency and throughput
- [ ] Document findings in `docs/experiments/`
- [ ] Decide on API refinements based on experimental results

## Step 7: Additional Backends (optional)

Based on experiment results, decide if more backends are worth pursuing:
- [ ] rnnoise
- [ ] Custom energy-based threshold detector

## Step 8: Finalize and Publish

- [ ] Refine public API based on experiment learnings
- [ ] Comprehensive documentation
- [ ] CI/CD pipeline
- [ ] Publish to crates.io

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Workspace layout | `crates/` + `tools/` | Keeps library deps clean, tool can be heavy |
| vad-lab UI | Web (React) | Need waveform viz, interactive config, multi-timeline overlay — too complex for TUI |
| Audio capture | Server-side (cpal) | Lower latency, no browser audio quirks, raw PCM at exact sample rates |
| Frontend delivery | Embedded in Rust binary | Single command to run, no separate frontend server needed |
| Config format | JSON | Easy to generate programmatically, native to the web frontend |
| Result storage | JSON sessions | Preserves structure, easy to load back in vad-lab |
