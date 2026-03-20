# wavekat-vad — crates.io Publication Plan

## Current State

The library crate (`crates/wavekat-vad`) is code-complete with 3 backends, preprocessing pipeline, 73 unit tests, and clean clippy/fmt. This document covers everything needed to publish v0.1.0 to crates.io.

---

## Dependency License Audit

All Rust crate dependencies are permissive and compatible with Apache-2.0:

| Crate | License | Notes |
|-------|---------|-------|
| `webrtc-vad` | MIT | Permissive |
| `ort` | MIT OR Apache-2.0 | Dual-licensed |
| `ndarray` | MIT OR Apache-2.0 | Dual-licensed |
| `nnnoiseless` | BSD-3-Clause | Requires attribution (retain copyright notice) |
| `rubato` | MIT | Permissive |
| `realfft` | MIT | Permissive |
| `thiserror` | MIT OR Apache-2.0 | Dual-licensed |

### ONNX Model Licenses

| Model | License | Concern? |
|-------|---------|----------|
| Silero VAD (`snakers4/silero-vad`) | MIT | No concerns |
| TEN-VAD (`TEN-framework/ten-vad`) | Apache-2.0 **with non-compete clause** | **YES** |

**TEN-VAD model has restrictive licensing.** The TEN-framework license includes a non-compete clause prohibiting deployment that competes with Agora's offerings, and restricts deployment to "solely for your benefit and the benefit of your direct End Users." This is **not standard open-source** despite the Apache-2.0 label.

### Actions Required

1. Document TEN-VAD model license restrictions clearly in README and crate docs
2. The `ten-vad` feature is already opt-in (not in default features) — good
3. Models are downloaded by users at build time (not bundled) — good, license applies directly between user and model author
4. Add a `NOTICE` file or license section mentioning `nnnoiseless` BSD-3-Clause attribution

---

## Issues to Fix

### 1. License mismatch (blocking)

- `Cargo.toml` declares `license = "MIT"`
- `LICENSE` file and `README.md` both say Apache-2.0
- **Fix**: Change `Cargo.toml` to `license = "Apache-2.0"`

### 2. Missing Cargo.toml metadata (blocking)

crates.io requires or strongly recommends these fields:

```toml
[package]
name = "wavekat-vad"
version = "0.1.0"
edition = "2021"
description = "Unified voice activity detection with multiple backends"
license = "Apache-2.0"
repository = "https://github.com/wavekat/wavekat-vad"
homepage = "https://github.com/wavekat/wavekat-vad"
documentation = "https://docs.rs/wavekat-vad"
readme = "../../README.md"                                    # relative from crate root
keywords = ["vad", "voice-activity-detection", "audio", "speech", "webrtc"]
categories = ["multimedia::audio"]
rust-version = "1.75"                                         # TBD — verify MSRV
```

### 3. Missing CHANGELOG.md (recommended)

Create `CHANGELOG.md` at repo root following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

## [0.1.0] - 2026-03-20

### Added
- `VoiceActivityDetector` trait — unified interface for all backends
- `VadCapabilities` — describes backend audio requirements
- `FrameAdapter` — automatic frame buffering for mismatched sizes
- WebRTC VAD backend (feature: `webrtc`, enabled by default)
- Silero VAD backend (feature: `silero`)
- TEN-VAD backend (feature: `ten-vad`) — note: model has non-standard license terms
- Audio preprocessing pipeline: high-pass filter, normalization, noise suppression
- Automatic ONNX model download at build time (Silero, TEN-VAD)
- Offline build support via `SILERO_MODEL_PATH` / `TEN_VAD_MODEL_PATH` env vars
```

### 4. Lib.rs crate-level docs outdated

- Mentions Silero as "coming soon" — it's implemented
- Missing TEN-VAD backend
- **Fix**: Update the doc comment in `lib.rs`

### 5. `serde` / `serde_json` — keep, but feature-gate

Serde **is used** in the library: `PreprocessorConfig` derives `Serialize`/`Deserialize` with `#[serde(default)]` attributes. `serde_json` is only used in tests.

- **Action**: Put `serde`/`serde_json` behind a `serde` feature flag
- Move `serde_json` to dev-dependencies (only used in tests)
- Gate `Serialize`/`Deserialize` derives with `#[cfg_attr(feature = "serde", ...)]`

### 6. `ureq` build dependency — make conditional

`ureq` is only used inside `#[cfg(feature = "silero")]` and `#[cfg(feature = "ten-vad")]` blocks in `build.rs`. Currently it's always pulled in.

- **Action**: Make `ureq` an optional build-dependency, activated by `silero` and `ten-vad` features

```toml
[build-dependencies]
ureq = { version = "3", optional = true }

[features]
silero = ["dep:ort", "dep:ndarray", "dep:ureq"]
ten-vad = ["dep:ort", "dep:ndarray", "dep:realfft", "dep:ureq"]
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Version history |
| `.github/workflows/ci.yml` | CI pipeline |

---

## CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

env:
  CARGO_TERM_COLOR: always

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo test --workspace
      - run: cargo doc --no-deps --document-private-items

  features:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        features:
          - ""                          # no default features
          - "webrtc"
          - "silero"
          - "ten-vad"
          - "webrtc,silero,ten-vad"
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test -p wavekat-vad --no-default-features --features "${{ matrix.features }}"
```

---

## README Updates

- Add badges at the top (crates.io version, docs.rs, CI status)
- Add "License" section with note about TEN-VAD model restrictions
- Keep vad-lab section as-is

---

## Pre-publish Checklist

```
[ ] Fix license in Cargo.toml → "Apache-2.0"
[ ] Add missing Cargo.toml metadata (repository, keywords, categories, readme)
[ ] Update lib.rs crate docs (Silero implemented, add TEN-VAD)
[ ] Feature-gate serde/serde_json, move serde_json to dev-deps
[ ] Make ureq conditional build-dep (silero/ten-vad only)
[ ] Add TEN-VAD license warning to README and crate docs
[ ] Create CHANGELOG.md
[ ] Create .github/workflows/ci.yml
[ ] Add badges to README.md
[ ] Run `cargo publish --dry-run -p wavekat-vad` to verify packaging
[ ] Verify `cargo doc --open` looks good
[ ] Tag release: git tag v0.1.0
[ ] Publish: `cargo publish -p wavekat-vad`
```

---

## Publish Commands

```sh
# Dry run first
cargo publish --dry-run -p wavekat-vad

# Check what gets included
cargo package --list -p wavekat-vad

# Publish for real
cargo publish -p wavekat-vad
```
