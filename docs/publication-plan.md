# wavekat-vad — crates.io Publication Plan

## Current State

The library crate (`crates/wavekat-vad`) is publication-ready with 3 backends, preprocessing pipeline, 38 unit tests, and clean clippy/fmt. All pre-publish issues have been resolved.

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

Documented in README and crate-level docs (`lib.rs`).

---

## Resolved Issues

All of the following have been addressed:

- [x] Fix license in Cargo.toml → `Apache-2.0`
- [x] Add missing Cargo.toml metadata (repository, homepage, documentation, readme, keywords, categories)
- [x] Update lib.rs crate docs (all 3 backends, feature flags, TEN-VAD license notice)
- [x] Feature-gate serde behind opt-in `serde` feature flag
- [x] Move `serde_json` to dev-dependencies
- [x] Make `ureq` a conditional build-dep (activated by `silero` and `ten-vad` features)
- [x] Add TEN-VAD license warning to README and crate docs
- [x] Add badges to README (crates.io, docs.rs, CI)
- [x] Create `.github/workflows/ci.yml` (fmt, clippy, test, doc + feature matrix)
- [x] `cargo publish --dry-run` passes (20 files, ~162KB)

---

## Release Process

Releases are fully automated via [release-plz](https://release-plz.dev/).

### How it works

1. Merge feature PRs into `main` using **squash merge** with conventional commit messages
2. release-plz automatically opens/updates a **Release PR** with version bump + CHANGELOG
3. When ready to publish, review and merge the Release PR
4. release-plz publishes to crates.io, creates a git tag, and a GitHub Release

### Conventional commit prefixes

| Prefix | Version bump | Example |
|--------|-------------|---------|
| `feat:` | minor (0.x.0) | `feat: add rnnoise backend` |
| `fix:` | patch (0.x.y) | `fix: correct frame size validation` |
| `feat!:` | major/minor | `feat!: redesign VoiceActivityDetector trait` |
| `chore:`, `docs:`, `refactor:`, `test:` | none | `docs: update README examples` |

### Configuration files

| File | Purpose |
|------|---------|
| `.github/workflows/release-plz.yml` | Automated release workflow |
| `.github/workflows/ci.yml` | CI pipeline (fmt, clippy, test, doc) |
| `release-plz.toml` | Excludes `vad-lab` from publishing |

### Required secrets

| Secret | Where | Purpose |
|--------|-------|---------|
| `CARGO_REGISTRY_TOKEN` | GitHub repo → Settings → Secrets → Actions | crates.io publish token |
| `GITHUB_TOKEN` | Automatic | PR creation and GitHub Releases |

### First publish

The very first publish must be done manually (or with the token) since the crate doesn't exist on crates.io yet. After that, release-plz handles everything automatically.

---

## Manual Publish Commands (first time only)

```sh
# Dry run first
cargo publish --dry-run -p wavekat-vad

# Check what gets included
cargo package --list -p wavekat-vad

# Publish for real
cargo publish -p wavekat-vad
```
