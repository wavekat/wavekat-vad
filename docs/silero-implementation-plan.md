# Silero VAD Implementation Plan

## Overview

Implement the Silero VAD backend using ONNX Runtime (`ort` crate). Silero is a neural network-based VAD that returns continuous speech probability (0.0-1.0), unlike WebRTC's binary output.

## Key Differences from WebRTC

| Aspect | WebRTC | Silero |
|--------|--------|--------|
| Output | Binary (0 or 1) | Continuous (0.0-1.0) |
| Sample rates | 8k, 16k, 32k, 48k | 8k, 16k only |
| Frame size | 10/20/30ms | 32ms (256@8k, 512@16k) |
| State | Minimal | LSTM hidden state + context |
| Speed | Very fast | Slower (neural network) |

## Files to Modify

1. `crates/wavekat-vad/Cargo.toml` - Add `ndarray` dependency
2. `crates/wavekat-vad/src/backends/silero.rs` - Full implementation
3. `tools/vad-lab/backend/Cargo.toml` - Enable `silero` feature
4. `tools/vad-lab/backend/src/pipeline.rs` - Wire up Silero backend

## Files to Create

1. `crates/wavekat-vad/models/silero_vad.onnx` - Download from snakers4/silero-vad
2. `crates/wavekat-vad/models/README.md` - Download instructions

## Implementation Steps

### Step 1: Download ONNX Model

Download Silero VAD v5 model from GitHub:
```bash
mkdir -p crates/wavekat-vad/models
curl -L -o crates/wavekat-vad/models/silero_vad.onnx \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
```

### Step 2: Update Library Dependencies

Add `ndarray` to `crates/wavekat-vad/Cargo.toml`:
```toml
silero = ["dep:ort", "dep:ndarray"]
ndarray = { version = "0.16", optional = true }
```

### Step 3: Implement SileroVad

**Struct definition:**
```rust
pub struct SileroVad {
    session: ort::Session,
    sample_rate: u32,
    chunk_size: usize,       // 256 @ 8kHz, 512 @ 16kHz
    state: Array3<f32>,      // LSTM state [2, 1, 128]
    context: Vec<f32>,       // Last 64 samples
}
```

**Key implementation details:**
- Embed model via `include_bytes!("../../models/silero_vad.onnx")`
- Only support 8kHz and 16kHz (Silero limitation)
- Maintain LSTM hidden state (`h`, `c`) between calls
- Prepend 64-sample context to each input chunk
- Reset clears state and context to zeros

### Step 4: Wire into vad-lab Pipeline

Add to `create_detector()`:
```rust
"silero-vad" => {
    use wavekat_vad::backends::silero::SileroVad;
    let vad = SileroVad::new(sample_rate)
        .map_err(|e| format!("failed to create Silero VAD: {e}"))?;
    Ok(Box::new(vad))
}
```

Add to `available_backends()`:
```rust
backends.insert("silero-vad".to_string(), vec![]);
```

### Step 5: Enable Feature in vad-lab

Update `tools/vad-lab/backend/Cargo.toml`:
```toml
wavekat-vad = { ..., features = ["webrtc", "silero", "denoise"] }
```

### Step 6: Add Tests

- Valid/invalid sample rate construction
- Process silence (expect low probability)
- Process wrong sample rate (expect error)
- Invalid frame size (expect error)
- State persistence across calls
- Reset clears state

## Sample Rate Consideration

Silero only supports 8kHz and 16kHz. The vad-lab currently uses device sample rates which may be 44.1kHz or 48kHz. Options:

1. **Error on unsupported rates** (initial approach) - Simple, forces user to configure appropriately
2. **Internal resampling** (future enhancement) - Resample to 16kHz internally

For this implementation, we'll error on unsupported rates and document the limitation.

## Verification

1. `cargo build -p wavekat-vad --features silero` - Build succeeds
2. `cargo test -p wavekat-vad --features silero` - Tests pass
3. `cargo clippy --workspace -- -D warnings` - No warnings
4. Run vad-lab, select "silero-vad" backend, verify it processes audio and returns probabilities
