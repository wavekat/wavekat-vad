//! Build script for vad-lab.
//!
//! Sets up runtime library paths for native dependencies.

fn main() {
    // TEN-VAD framework needs an rpath so the dynamic linker can find it at runtime.
    // The wavekat-vad build.rs handles link-search and link-lib, but rpath is
    // per-binary and must be set by each crate that produces a binary.
    #[cfg(target_os = "macos")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace_root = std::path::Path::new(&manifest_dir).join("../../..");
        let framework_dir = workspace_root.join("third_party/ten-vad/lib/macOS");
        if let Ok(framework_dir) = framework_dir.canonicalize() {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                framework_dir.display()
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace_root = std::path::Path::new(&manifest_dir).join("../../..");
        let lib_dir = workspace_root.join("third_party/ten-vad/lib/Linux/x64");
        if let Ok(lib_dir) = lib_dir.canonicalize() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
    }
}
