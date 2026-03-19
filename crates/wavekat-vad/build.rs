//! Build script for wavekat-vad.
//!
//! Downloads the Silero VAD ONNX model at build time if the `silero` feature is enabled.
//! Sets up linking for TEN-VAD prebuilt libraries if the `ten-vad` feature is enabled.
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
    #[cfg(feature = "silero")]
    setup_silero_model();

    #[cfg(feature = "ten-vad")]
    setup_ten_vad();
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

#[cfg(feature = "ten-vad")]
fn setup_ten_vad() {
    println!("cargo:rerun-if-env-changed=TEN_VAD_MODEL_PATH");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    // TEN-VAD lives in a git submodule at the workspace root
    let workspace_root = Path::new(&manifest_dir).join("../..");
    let ten_vad_dir = workspace_root.join("third_party/ten-vad");

    // Re-run if submodule contents change
    println!(
        "cargo:rerun-if-changed={}",
        ten_vad_dir.join("lib").display()
    );

    // --- Link the prebuilt C library ---

    #[cfg(target_os = "macos")]
    {
        let framework_dir = ten_vad_dir.join("lib/macOS");
        let framework_dir = framework_dir
            .canonicalize()
            .expect("ten-vad submodule not found — run: git submodule update --init");
        println!(
            "cargo:rustc-link-search=framework={}",
            framework_dir.display()
        );
        println!("cargo:rustc-link-lib=framework=ten_vad");
        // Set rpath so the dynamic linker can find the framework at runtime
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            framework_dir.display()
        );
    }

    #[cfg(target_os = "linux")]
    {
        let lib_dir = ten_vad_dir.join("lib/Linux/x64");
        let lib_dir = lib_dir
            .canonicalize()
            .expect("ten-vad submodule not found — run: git submodule update --init");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=ten_vad");
        // Set rpath so the dynamic linker can find the .so at runtime
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    // --- Prepare the ONNX model (same pattern as Silero) ---

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_dest = Path::new(&out_dir).join("ten_vad.onnx");

    // Option 1: Use local file if TEN_VAD_MODEL_PATH is set
    if let Ok(local_path) = env::var("TEN_VAD_MODEL_PATH") {
        let local_path = Path::new(&local_path);
        if !local_path.exists() {
            panic!(
                "TEN_VAD_MODEL_PATH points to non-existent file: {}",
                local_path.display()
            );
        }
        println!(
            "cargo:warning=Using local TEN-VAD model: {}",
            local_path.display()
        );
        fs::copy(local_path, &model_dest).expect("failed to copy local model file");
        println!("cargo:rerun-if-changed={}", local_path.display());
        return;
    }

    // Skip if already exists
    if model_dest.exists() {
        return;
    }

    // Option 2: Copy from submodule if available
    let submodule_model = ten_vad_dir.join("src/onnx_model/ten-vad.onnx");
    if submodule_model.exists() {
        println!(
            "cargo:warning=Copying TEN-VAD model from submodule: {}",
            submodule_model.display()
        );
        fs::copy(&submodule_model, &model_dest)
            .expect("failed to copy TEN-VAD model from submodule");
        return;
    }

    // Option 3: Download from GitHub
    let model_url = "https://github.com/TEN-framework/ten-vad/raw/main/src/onnx_model/ten-vad.onnx";
    println!("cargo:warning=Downloading TEN-VAD model from {model_url}");

    let response = ureq::get(model_url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download TEN-VAD model from {model_url}: {e}"));

    let bytes = response
        .into_body()
        .read_to_vec()
        .expect("failed to read TEN-VAD model bytes");

    fs::write(&model_dest, &bytes).expect("failed to write TEN-VAD model file");

    println!(
        "cargo:warning=TEN-VAD model downloaded to {}",
        model_dest.display()
    );
}
