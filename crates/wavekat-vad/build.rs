//! Build script for wavekat-vad.
//!
//! Downloads the Silero VAD ONNX model at build time if the `silero` feature is enabled.
//! Downloads the TEN-VAD ONNX model at build time if the `ten-vad` feature is enabled.
//!
//! # Environment Variables
//!
//! - `SILERO_MODEL_PATH`: Path to a local ONNX model file (skips download)
//! - `SILERO_MODEL_URL`: Custom URL to download the model from

#[allow(unused_imports)]
use std::env;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::Path;

fn main() {
    // docs.rs builds with --network none, so we can't download models.
    // Write empty placeholder files so include_bytes! compiles.
    if env::var("DOCS_RS").is_ok() {
        #[cfg(feature = "silero")]
        {
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let model_path = Path::new(&out_dir).join("silero_vad.onnx");
            if !model_path.exists() {
                fs::write(&model_path, b"").expect("failed to write placeholder model");
            }
        }
        #[cfg(feature = "ten-vad")]
        {
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let model_path = Path::new(&out_dir).join("ten-vad.onnx");
            if !model_path.exists() {
                fs::write(&model_path, b"").expect("failed to write placeholder model");
            }
        }
        #[cfg(feature = "firered")]
        {
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let model_path = Path::new(&out_dir).join("fireredvad_stream_vad_with_cache.onnx");
            if !model_path.exists() {
                fs::write(&model_path, b"").expect("failed to write placeholder model");
            }
            let cmvn_path = Path::new(&out_dir).join("firered_cmvn.ark");
            if !cmvn_path.exists() {
                fs::write(&cmvn_path, b"").expect("failed to write placeholder cmvn");
            }
        }
        return;
    }

    #[cfg(feature = "silero")]
    setup_silero_model();

    #[cfg(feature = "ten-vad")]
    setup_ten_vad_model();

    #[cfg(feature = "firered")]
    setup_firered_model();
}

#[cfg(feature = "silero")]
fn setup_silero_model() {
    const DEFAULT_MODEL_URL: &str =
        "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx";
    const SILERO_MODEL_NAME: &str = "silero_vad.onnx";
    // Bump this when updating the default model URL to invalidate cached downloads.
    const SILERO_MODEL_VERSION: &str = "v6.2.1";

    // Re-run if these env vars change
    println!("cargo:rerun-if-env-changed=SILERO_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=SILERO_MODEL_URL");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_path = Path::new(&out_dir).join(SILERO_MODEL_NAME);
    let version_path = Path::new(&out_dir).join("silero_vad.version");

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

    // Skip download if model exists and version matches
    let cached_version = fs::read_to_string(&version_path).unwrap_or_default();
    if model_path.exists() && cached_version.trim() == SILERO_MODEL_VERSION {
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
    fs::write(&version_path, SILERO_MODEL_VERSION).expect("failed to write version marker");

    println!(
        "cargo:warning=Silero VAD model ({SILERO_MODEL_VERSION}) downloaded to {}",
        model_path.display()
    );
}

#[cfg(feature = "ten-vad")]
fn setup_ten_vad_model() {
    println!("cargo:rerun-if-env-changed=TEN_VAD_MODEL_PATH");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_dest = Path::new(&out_dir).join("ten-vad.onnx");

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

    // Option 2: Download from GitHub
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

#[cfg(feature = "firered")]
fn setup_firered_model() {
    println!("cargo:rerun-if-env-changed=FIRERED_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=FIRERED_CMVN_PATH");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");

    // --- ONNX model ---
    let model_dest = Path::new(&out_dir).join("fireredvad_stream_vad_with_cache.onnx");

    if let Ok(local_path) = env::var("FIRERED_MODEL_PATH") {
        let local_path = Path::new(&local_path);
        if !local_path.exists() {
            panic!(
                "FIRERED_MODEL_PATH points to non-existent file: {}",
                local_path.display()
            );
        }
        println!(
            "cargo:warning=Using local FireRedVAD model: {}",
            local_path.display()
        );
        fs::copy(local_path, &model_dest).expect("failed to copy local model file");
        println!("cargo:rerun-if-changed={}", local_path.display());
    } else if !model_dest.exists() {
        let model_url = "https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx";
        println!("cargo:warning=Downloading FireRedVAD model from {model_url}");

        let response = ureq::get(model_url).call().unwrap_or_else(|e| {
            panic!("failed to download FireRedVAD model from {model_url}: {e}")
        });

        let bytes = response
            .into_body()
            .read_to_vec()
            .expect("failed to read FireRedVAD model bytes");

        fs::write(&model_dest, &bytes).expect("failed to write FireRedVAD model file");
        println!(
            "cargo:warning=FireRedVAD model downloaded to {}",
            model_dest.display()
        );
    }

    // --- CMVN file ---
    let cmvn_dest = Path::new(&out_dir).join("firered_cmvn.ark");

    if let Ok(local_path) = env::var("FIRERED_CMVN_PATH") {
        let local_path = Path::new(&local_path);
        if !local_path.exists() {
            panic!(
                "FIRERED_CMVN_PATH points to non-existent file: {}",
                local_path.display()
            );
        }
        println!(
            "cargo:warning=Using local FireRedVAD CMVN: {}",
            local_path.display()
        );
        fs::copy(local_path, &cmvn_dest).expect("failed to copy local cmvn file");
        println!("cargo:rerun-if-changed={}", local_path.display());
    } else if !cmvn_dest.exists() {
        let cmvn_url = "https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/cmvn.ark";
        println!("cargo:warning=Downloading FireRedVAD CMVN from {cmvn_url}");

        let response = ureq::get(cmvn_url)
            .call()
            .unwrap_or_else(|e| panic!("failed to download FireRedVAD CMVN from {cmvn_url}: {e}"));

        let bytes = response
            .into_body()
            .read_to_vec()
            .expect("failed to read CMVN bytes");

        fs::write(&cmvn_dest, &bytes).expect("failed to write CMVN file");
        println!(
            "cargo:warning=FireRedVAD CMVN downloaded to {}",
            cmvn_dest.display()
        );
    }
}
