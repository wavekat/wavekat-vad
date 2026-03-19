//! Build script for wavekat-vad.
//!
//! Downloads the Silero VAD ONNX model at build time if the `silero` feature is enabled.
//!
//! # Environment Variables
//!
//! - `SILERO_MODEL_PATH`: Path to a local ONNX model file (skips download)
//! - `SILERO_MODEL_URL`: Custom URL to download the model from

use std::env;
use std::fs;
use std::path::Path;

const DEFAULT_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";
const SILERO_MODEL_NAME: &str = "silero_vad.onnx";

fn main() {
    // Only process if silero feature is enabled
    #[cfg(feature = "silero")]
    setup_silero_model();
}

#[cfg(feature = "silero")]
fn setup_silero_model() {
    // Re-run if these env vars change
    println!("cargo:rerun-if-env-changed=SILERO_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=SILERO_MODEL_URL");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_path = Path::new(&out_dir).join(SILERO_MODEL_NAME);

    // Option 1: Use local file if SILERO_MODEL_PATH is set
    if let Ok(local_path) = env::var("SILERO_MODEL_PATH") {
        let local_path = Path::new(&local_path);
        if !local_path.exists() {
            panic!(
                "SILERO_MODEL_PATH points to non-existent file: {}",
                local_path.display()
            );
        }
        println!(
            "cargo:warning=Using local Silero model: {}",
            local_path.display()
        );
        fs::copy(local_path, &model_path).expect("failed to copy local model file");
        println!("cargo:rerun-if-changed={}", local_path.display());
        return;
    }

    // Skip download if already exists
    if model_path.exists() {
        return;
    }

    // Option 2: Use custom URL if SILERO_MODEL_URL is set, otherwise use default
    let model_url = env::var("SILERO_MODEL_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_string());

    println!("cargo:warning=Downloading Silero VAD model from {model_url}");

    let response = ureq::get(&model_url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download Silero model from {model_url}: {e}"));

    let bytes = response
        .into_body()
        .read_to_vec()
        .expect("failed to read model bytes");

    fs::write(&model_path, &bytes).expect("failed to write model file");

    println!(
        "cargo:warning=Silero VAD model downloaded to {}",
        model_path.display()
    );
}
