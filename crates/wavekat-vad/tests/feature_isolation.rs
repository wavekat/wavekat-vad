//! Guards the dependency isolation of the `earshot` feature.
//!
//! # Why this exists
//!
//! `webrtc-vad` bundles `libfvad`, which exports unmangled `WebRtcSpl_*`
//! symbols from object files whose names collide with those in LiveKit's
//! `webrtc-sys`. A host binary that links both fails to link at all under
//! strict linkers (e.g. Wild rejects the duplicate definitions). The whole
//! point of the pure-Rust `earshot` backend is to give such hosts a VAD they
//! can enable *without* dragging `webrtc-vad` in.
//!
//! So `--no-default-features --features earshot` must resolve to a dependency
//! graph with no `webrtc-vad`, no ONNX runtime (`ort`/`ort-sys`), and no
//! build-time model downloader (`ureq`). This test asserts that against the
//! real resolver output, so the guarantee cannot regress silently — for
//! example by someone adding `dep:webrtc-vad` to the `earshot` feature list.
//!
//! `scripts/check_earshot_isolation.sh` performs the same check from CI
//! without needing a compiled test binary.

use std::process::Command;

/// Crates that must never appear when only `earshot` is enabled.
///
/// `realfft`/`rustfft` are deliberately **not** listed: they arrive through
/// `rubato`, which is a mandatory dependency of this crate on every feature
/// set. They are pure Rust and link no C symbols, so they are irrelevant to
/// the collision above.
const FORBIDDEN: &[&str] = &[
    // Bundles libfvad; the symbol collision this whole backend exists to avoid.
    "webrtc-vad",
    // ONNX Runtime, pulled by silero / ten-vad / firered.
    "ort",
    "ort-sys",
    // Build-time model downloader.
    "ureq",
    // Remaining backend-only dependencies.
    "ndarray",
    "nnnoiseless",
];

/// Only `normal` and `build` edges matter: those are what a downstream
/// consumer actually links. Dev-dependencies never reach their binary.
const EDGES: [&str; 2] = ["--edges", "normal,build"];

fn cargo_tree(args: &[&str]) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(&cargo)
        .arg("tree")
        .args(EDGES)
        .args(args)
        // Do not inherit the parent build's flags/target dir quirks.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        // CI sets `CARGO_TERM_COLOR: always` at workflow scope, and a spawned
        // `cargo` inherits it. Colour turns every node line into escape
        // sequences this test then has to parse around; ask for none.
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to run `{cargo} tree {}`: {e}", args.join(" ")));

    assert!(
        output.status.success(),
        "`cargo tree {}` failed with {}:\n{}",
        args.join(" "),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo tree emitted non-UTF-8")
}

/// Strips ANSI CSI escape sequences (`ESC [` … final byte) from one line.
///
/// A coloured node line is `"\x1b[2m├──\x1b[0m earshot v1.2.1"`. The name
/// parser below trims leading non-alphanumerics, which eats `ESC` and `[` and
/// then stops at the `2` of `\x1b[2m` — `2` *is* alphanumeric — so the node's
/// name reads as `2m├──\x1b[0m`. Every node parses to garbage and the
/// isolation assertions fail against a dependency tree that is in fact
/// correct: the panic message prints a tree whose first line is
/// `├── earshot v1.2.1` while claiming earshot is absent.
///
/// `cargo_tree` above forces `CARGO_TERM_COLOR=never`, which removes the
/// cause. This keeps the parser correct for any input rather than only for the
/// input we currently arrange to receive.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        // CSI: `ESC [`, parameter bytes, then a final byte in `@`..=`~`.
        if chars.next() != Some('[') {
            continue;
        }
        for c in chars.by_ref() {
            if ('@'..='~').contains(&c) {
                break;
            }
        }
    }
    out
}

/// Returns the crate names in a `cargo tree` rendering, one per node.
fn crate_names(tree: &str) -> Vec<String> {
    tree.lines()
        .filter_map(|line| {
            // Nodes look like "├── webrtc-vad v0.4.0" or "wavekat-vad v0.1.16 (/path)".
            let line = strip_ansi(line);
            let rest = line.trim_start_matches(|c: char| !c.is_alphanumeric() && c != '_');
            let name = rest.split_whitespace().next()?;
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}

/// The parser must survive coloured output, because CI sets
/// `CARGO_TERM_COLOR: always` at workflow scope and every job inherits it.
///
/// Without this, the only thing standing between a green matrix and a silently
/// unparseable tree is an environment variable set in another file.
#[test]
fn crate_names_reads_a_colour_escaped_tree() {
    let coloured = concat!(
        "wavekat-vad v0.1.16 (/w/wavekat-vad)\n",
        "\x1b[2m├──\x1b[0m earshot v1.2.1\n",
        "\x1b[2m└──\x1b[0m thiserror v2.0.18\n",
    );
    assert_eq!(
        crate_names(coloured),
        vec!["wavekat-vad", "earshot", "thiserror"],
        "the tree parser is blind to ANSI colour, which is how CI renders it"
    );
}

#[test]
fn earshot_only_build_excludes_webrtc_vad_and_onnx() {
    let tree = cargo_tree(&[
        "-p",
        "wavekat-vad",
        "--no-default-features",
        "--features",
        "earshot",
    ]);
    let names = crate_names(&tree);

    assert!(
        names.iter().any(|n| n == "earshot"),
        "the `earshot` feature did not pull in the earshot crate:\n{tree}"
    );

    for forbidden in FORBIDDEN {
        assert!(
            !names.iter().any(|n| n == forbidden),
            "`--no-default-features --features earshot` pulled in `{forbidden}`, \
             which defeats the purpose of this backend.\nFull tree:\n{tree}"
        );
    }
}

/// Sanity check: the same parser does find `webrtc-vad` when it is genuinely
/// present, so the assertion above is not vacuously passing on a parse bug.
#[test]
fn the_isolation_check_is_not_vacuous() {
    let tree = cargo_tree(&[
        "-p",
        "wavekat-vad",
        "--no-default-features",
        "--features",
        "webrtc",
    ]);
    let names = crate_names(&tree);
    assert!(
        names.iter().any(|n| n == "webrtc-vad"),
        "parser failed to see webrtc-vad in a tree that must contain it:\n{tree}"
    );
}
