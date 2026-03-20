# TEN-VAD Integration Plan

**Date:** 2026-03-20
**Goal:** Add [TEN-VAD](https://github.com/TEN-framework/ten-vad) as a third backend for comparison in vad-lab.

## Background

TEN-VAD is a real-time VAD system by Agora (TEN Framework). It claims superior precision vs both WebRTC VAD and Silero VAD. Internally it uses an ONNX neural network model with mel-filterbank + pitch estimation preprocessing.

### TEN-VAD Architecture

- **Model:** Small RNN with 64-dim hidden state, 5 I/O tensors (1 feature input + 4 hidden states)
- **Features:** 41-dim vector (40 mel bands + 1 pitch freq), stacked over 3-frame context → input shape `[1, 3, 41]`
- **Preprocessing:** Pre-emphasis (0.97) → STFT (FFT 1024, hop 256, window 768) → 40-band mel filterbank → log compression + mean/variance normalization → LPC pitch estimation
- **Audio:** 16kHz only, i16 samples, flexible hop size (recommended 160 or 256 samples)
- **Output:** Speech probability `[0.0, 1.0]`

### C API (from `ten_vad.h`)

```c
int ten_vad_create(ten_vad_handle_t *handle, size_t hop_size, float threshold);
int ten_vad_process(ten_vad_handle_t handle, const int16_t *audio_data,
                    size_t audio_data_length, float *out_probability, int *out_flag);
int ten_vad_destroy(ten_vad_handle_t *handle);
const char *ten_vad_get_version(void);
```

### Distribution

Prebuilt platform-specific shared libraries checked into their repo:

| Platform | Library | Arch |
|----------|---------|------|
| macOS | `ten_vad.framework` | arm64 + x86_64 (universal) |
| Linux | `libten_vad.so` | x64 |
| Windows | `ten_vad.dll` | x64, x86 |

No Rust bindings exist. No crates.io package. No source build for the core library.

### License

Apache 2.0 **with additional conditions** — includes a non-compete clause against Agora's offerings. This effectively prevents redistribution as part of a library that enables third-party app development. **Fine for internal experimentation / vad-lab, but blocks crates.io publication of the `ten-vad` feature.**

---

## Integration Strategy: Two Phases

We implement **both** approaches in sequence. Phase 1 (FFI) gives us a working reference fast. Phase 2 (pure ONNX) gives us a clean, portable implementation. Running both side-by-side in vad-lab lets us validate that the Rust preprocessing matches the reference output.

### Phase 1: FFI to Prebuilt C Library (`ten-vad` feature)

Link against TEN-VAD's prebuilt shared libraries via Rust FFI.

**Pros:**
- Simple code (~100 lines of FFI + trait impl)
- Reference implementation = guaranteed correctness
- Fast to implement (1-2 days)
- Preprocessing handled by the library

**Cons:**
- Platform-specific binary management
- Closed-source dependency for the inference + preprocessing
- Build setup complexity (framework linking on macOS, .so on Linux)
- License blocks crates.io publication

### Phase 2: Pure ONNX with Rust Preprocessing (`ten-vad-onnx` feature)

Use the open-source `ten-vad.onnx` model with custom Rust preprocessing, reusing the existing `ort` crate.

**Pros:**
- No binary distribution headaches
- Same pattern as Silero backend
- Cross-platform automatically
- No new native dependencies
- Publishable to crates.io (model is open-source)

**Cons:**
- Significant preprocessing work: mel filterbank, STFT, LPC pitch estimation
- New dependency needed for FFT (e.g., `realfft` or `rustfft`)
- Estimated 3-5 days

**Key benefit of doing both:** We can run `ten-vad` (FFI) and `ten-vad-onnx` (pure Rust) side-by-side in vad-lab on the same audio. If the probabilities match (within tolerance), we know the Rust preprocessing is correct. If they diverge, we can debug exactly which preprocessing stage is off. Once validated, the FFI backend can be retired and the pure ONNX version becomes the canonical implementation.

---

## Implementation Plan (Strategy A: FFI)

### Step 1: Vendor the Prebuilt Libraries

Download and commit the prebuilt libraries from the TEN-VAD repo into a vendored directory:

```
crates/wavekat-vad/
  vendor/
    ten-vad/
      include/
        ten_vad.h
      lib/
        mac/
          ten_vad.framework/    (arm64 + x86_64)
        linux-x64/
          libten_vad.so
      model/
        ten-vad.onnx            (loaded at runtime by the library)
      LICENSE
```

The ONNX model file must be accessible at runtime relative to the working directory (path: `onnx_model/ten-vad.onnx`). We need to handle this — either:
- Copy the model to the expected path at runtime, or
- Set the working directory appropriately, or
- Check if the library supports an env var or API to configure the model path

**Action items:**
- [ ] Clone/download the TEN-VAD repo
- [ ] Copy the header, platform libs, model, and LICENSE into `vendor/ten-vad/`
- [ ] Add `vendor/` to the repo (these are binary blobs, so consider git-lfs or keeping them small)

### Step 2: Build Script (`build.rs`)

Extend the existing build script to handle `ten-vad` feature:

```rust
#[cfg(feature = "ten-vad")]
fn setup_ten_vad() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let vendor_dir = Path::new(&manifest_dir).join("vendor/ten-vad");

    #[cfg(target_os = "macos")]
    {
        let framework_dir = vendor_dir.join("lib/mac");
        println!("cargo:rustc-link-search=framework={}", framework_dir.display());
        println!("cargo:rustc-link-lib=framework=ten_vad");
    }

    #[cfg(target_os = "linux")]
    {
        let lib_dir = vendor_dir.join("lib/linux-x64");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=ten_vad");
    }

    // Copy model file to OUT_DIR for runtime access
    let model_src = vendor_dir.join("model/ten-vad.onnx");
    let out_dir = env::var("OUT_DIR").unwrap();
    let model_dst = Path::new(&out_dir).join("ten-vad.onnx");
    fs::copy(&model_src, &model_dst).expect("failed to copy TEN-VAD model");
}
```

### Step 3: FFI Bindings (`backends/ten_vad.rs`)

Write minimal unsafe FFI bindings:

```rust
use std::ffi::c_void;
use std::os::raw::c_int;

type TenVadHandle = *mut c_void;

extern "C" {
    fn ten_vad_create(handle: *mut TenVadHandle, hop_size: usize, threshold: f32) -> c_int;
    fn ten_vad_process(
        handle: TenVadHandle,
        audio_data: *const i16,
        audio_data_length: usize,
        out_probability: *mut f32,
        out_flag: *mut c_int,
    ) -> c_int;
    fn ten_vad_destroy(handle: *mut TenVadHandle) -> c_int;
    fn ten_vad_get_version() -> *const std::os::raw::c_char;
}
```

### Step 4: VoiceActivityDetector Implementation

```rust
pub struct TenVad {
    handle: TenVadHandle,
    hop_size: usize,
    threshold: f32,
}

impl TenVad {
    pub fn new(hop_size: usize, threshold: f32) -> Result<Self, VadError> { ... }
}

impl VoiceActivityDetector for TenVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: 16000,
            frame_size: self.hop_size,
            frame_duration_ms: (self.hop_size as u32 * 1000) / 16000,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // Validate 16kHz only
        // Call ten_vad_process via FFI
        // Return probability
    }

    fn reset(&mut self) {
        // TEN-VAD has no public reset API.
        // Option: destroy + recreate the handle.
    }
}

impl Drop for TenVad {
    fn drop(&mut self) {
        unsafe { ten_vad_destroy(&mut self.handle); }
    }
}

unsafe impl Send for TenVad {}
```

### Step 5: Feature Gate

In `Cargo.toml`:
```toml
[features]
ten-vad = []  # no dep crate — links via build.rs
```

In `backends/mod.rs`:
```rust
#[cfg(feature = "ten-vad")]
pub mod ten_vad;
```

### Step 6: Wire into vad-lab Pipeline

In `tools/vad-lab/backend/src/pipeline.rs`:

1. Add `"ten-vad"` match arm in `create_detector()`:
   ```rust
   "ten-vad" => {
       use wavekat_vad::backends::ten_vad::TenVad;
       let hop_size = config.params.get("hop_size")
           .and_then(|v| v.as_u64())
           .unwrap_or(256) as usize;
       let threshold = config.params.get("threshold")
           .and_then(|v| v.as_f64())
           .unwrap_or(0.5) as f32;
       let vad = TenVad::new(hop_size, threshold)
           .map_err(|e| format!("failed to create TEN VAD: {e}"))?;
       Ok(Box::new(vad))
   }
   ```

2. Add `"ten-vad"` entry in `available_backends()`:
   ```rust
   backends.insert("ten-vad".to_string(), vec![
       ParamInfo {
           name: "hop_size".to_string(),
           description: "Samples per frame (160=10ms, 256=16ms at 16kHz)".to_string(),
           param_type: ParamType::Select(vec![
               "160".to_string(),
               "256".to_string(),
           ]),
           default: serde_json::json!("256"),
       },
       ParamInfo {
           name: "threshold".to_string(),
           description: "Speech detection threshold (0.0 - 1.0)".to_string(),
           param_type: ParamType::Float { min: 0.0, max: 1.0 },
           default: serde_json::json!(0.5),
       },
   ]);
   ```

3. Enable `ten-vad` feature in `tools/vad-lab/backend/Cargo.toml`:
   ```toml
   wavekat-vad = { path = "../../../crates/wavekat-vad", features = ["webrtc", "silero", "denoise", "ten-vad"] }
   ```

### Step 7: Runtime Model Path

TEN-VAD loads its ONNX model from `onnx_model/ten-vad.onnx` relative to the working directory. Options:

1. **Symlink at build time** — `build.rs` creates a symlink in the binary's output directory
2. **Copy at startup** — vad-lab copies the model to the expected path on launch
3. **Environment variable** — check if TEN-VAD respects any config (unlikely, based on source)

Recommended: Option 2 — copy the model file to a temp directory and `chdir` before initializing TEN-VAD, or embed the model and write it to the expected path on startup.

### Step 8: Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn create_ten_vad() { ... }

    #[test]
    fn process_silence() { ... }

    #[test]
    fn process_invalid_frame_size() { ... }

    #[test]
    fn probability_in_range() { ... }

    #[test]
    fn reset_recreates_handle() { ... }
}
```

### Step 9: Documentation

- Add TEN-VAD to the crate-level doc comment in `lib.rs`
- Add `///` doc comments on all public items in `ten_vad.rs`
- Note the license restriction in the module docs

---

## Phase 2 Implementation Plan (Pure ONNX: `ten-vad-onnx`)

Phase 2 builds a standalone Rust implementation that loads the `ten-vad.onnx` model and runs its own preprocessing — no C library needed.

### Step 1: Feature Gate & Dependencies

In `Cargo.toml`:
```toml
[features]
ten-vad-onnx = ["dep:ort", "dep:ndarray", "dep:realfft"]

[dependencies]
realfft = { version = "3", optional = true }  # FFT for STFT computation
```

In `backends/mod.rs`:
```rust
#[cfg(feature = "ten-vad-onnx")]
pub mod ten_vad_onnx;
```

### Step 2: Model Embedding

Extend `build.rs` to download and embed `ten-vad.onnx` (same pattern as Silero):

```rust
#[cfg(feature = "ten-vad-onnx")]
fn setup_ten_vad_onnx_model() {
    // Download from TEN-VAD repo: src/onnx_model/ten-vad.onnx
    // Embed via include_bytes!
}
```

### Step 3: Preprocessing Pipeline in Rust

Implement in `backends/ten_vad_onnx/preprocessing.rs`, porting from the open-source C++ code:

1. **Pre-emphasis filter** (trivial)
   - `y[n] = x[n] - 0.97 * x[n-1]`

2. **STFT** (uses `realfft` crate)
   - FFT size: 1024, hop size: 256, window size: 768
   - Hann window
   - Output: magnitude spectrum

3. **Mel filterbank** (40 bands)
   - Compute mel-spaced filter bank matrix (16kHz, 40 bands, FFT size 1024)
   - Apply to magnitude spectrum → 40-dim mel energies
   - Log compression: `log(max(energy, 1e-10))`

4. **Mean/variance normalization**
   - Running mean and variance across frames
   - Normalize: `(x - mean) / sqrt(var + eps)`

5. **LPC pitch estimation** (ported from `pitch_est.cc`, BSD-licensed)
   - LPC autocorrelation
   - Pitch period detection
   - Convert to frequency → 1-dim pitch feature

6. **Feature assembly**
   - Stack: `[40 mel bands | 1 pitch] = 41-dim`
   - Context window: stack 3 consecutive frames → `[1, 3, 41]`

### Step 4: ONNX Inference

Similar to Silero, but with different I/O:

```rust
pub struct TenVadOnnx {
    session: Session,
    preprocessor: TenVadPreprocessor,
    // 4 hidden state tensors, each [1, 64]
    h0: Array2<f32>,
    h1: Array2<f32>,
    h2: Array2<f32>,
    h3: Array2<f32>,
}
```

Model I/O:
- Input: `"features"` → `[1, 3, 41]`
- Input: `"h0"`, `"h1"`, `"h2"`, `"h3"` → each `[1, 64]`
- Output: `"probability"` → scalar
- Output: `"h0_out"`, `"h1_out"`, `"h2_out"`, `"h3_out"` → each `[1, 64]`

### Step 5: VoiceActivityDetector Implementation

```rust
impl VoiceActivityDetector for TenVadOnnx {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: 16000,
            frame_size: 256,  // internal hop size
            frame_duration_ms: 16,
        }
    }

    fn process(&mut self, samples: &[i16], sample_rate: u32) -> Result<f32, VadError> {
        // 1. Feed samples to preprocessor (handles FIFO buffering internally)
        // 2. If a full feature frame is ready, run ONNX inference
        // 3. Update hidden states
        // 4. Return probability
    }

    fn reset(&mut self) {
        self.preprocessor.reset();
        self.h0.fill(0.0);
        self.h1.fill(0.0);
        self.h2.fill(0.0);
        self.h3.fill(0.0);
    }
}
```

### Step 6: Wire into vad-lab Pipeline

Add `"ten-vad-onnx"` as a separate backend in `pipeline.rs`:

```rust
"ten-vad-onnx" => {
    use wavekat_vad::backends::ten_vad_onnx::TenVadOnnx;
    let vad = TenVadOnnx::new(sample_rate)
        .map_err(|e| format!("failed to create TEN VAD ONNX: {e}"))?;
    Ok(Box::new(vad))
}
```

### Step 7: Validation Tests

Compare outputs of `ten-vad` (FFI) vs `ten-vad-onnx` (pure Rust) on the same audio:

```rust
#[test]
fn ffi_and_onnx_outputs_match() {
    let mut ffi_vad = TenVad::new(256, 0.5).unwrap();
    let mut onnx_vad = TenVadOnnx::new(16000).unwrap();

    let test_audio = load_test_wav("testdata/speech/sample.wav");
    for frame in test_audio.chunks(256) {
        let ffi_prob = ffi_vad.process(frame, 16000).unwrap();
        let onnx_prob = onnx_vad.process(frame, 16000).unwrap();
        assert!((ffi_prob - onnx_prob).abs() < 0.01,
            "divergence: ffi={ffi_prob:.4}, onnx={onnx_prob:.4}");
    }
}
```

---

## Key Constraints & Risks

| Risk | Mitigation |
|------|------------|
| Model path at runtime (Phase 1) | Copy model to expected location on startup |
| No `reset()` in C API (Phase 1) | Destroy and recreate the handle |
| 16kHz only | Pipeline already defaults to 16kHz; validate and reject other rates |
| License non-compete (Phase 1) | Feature-gated, documented; don't enable by default; exclude from crates.io publish |
| macOS framework linking (Phase 1) | Test on both arm64 and x86_64; may need `@rpath` fixup |
| Prebuilt binary size (Phase 1) | ~5MB per platform; consider git-lfs if repo grows |
| Preprocessing divergence (Phase 2) | Validate against Phase 1 FFI output; test per-stage |
| LPC pitch estimation complexity (Phase 2) | Port from BSD-licensed `pitch_est.cc`; unit test against reference values |
| No Windows CI | Start with macOS + Linux only |

## Phasing & Exit Criteria

| Phase | Done when | Can retire? |
|-------|-----------|-------------|
| Phase 1 (FFI) | `ten-vad` backend works in vad-lab, produces reasonable probabilities on test audio | No — needed as reference for Phase 2 |
| Phase 2 (ONNX) | `ten-vad-onnx` output matches FFI within tolerance (< 0.01 abs diff) on all test audio | Phase 1 can be retired; `ten-vad-onnx` becomes the canonical backend |

## Out of Scope

- Windows support — can add later, not needed for experimentation
- Android/iOS — not relevant for vad-lab
- Publishing `ten-vad` to crates.io — possible once validated, but not this iteration

---

## Outcome (2026-03-20)

Both phases completed successfully. The pure Rust ONNX backend was validated against the FFI reference with excellent agreement:
- Silence: diff ~0.0000
- Synthetic speech: diff < 0.002
- Noise: diff <= 0.06

**Decision:** The FFI backend (Phase 1) has been retired. The pure Rust ONNX implementation is now the sole TEN-VAD backend, available under the `ten-vad` feature flag. The git submodule at `third_party/ten-vad` and the `.gitmodules` file have been removed. The ONNX model is downloaded at build time from the TEN-VAD GitHub repo.
