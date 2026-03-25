# Video Script: Adding FireRedVAD to wavekat-vad

**Working title:** New VAD Backend — FireRedVAD in wavekat-vad

---

## INTRO [~0:00]

Hey everyone. Today we're adding a new voice activity detection backend to wavekat-vad — FireRedVAD. It's the most accurate VAD we've tested so far, and it's now available alongside WebRTC, Silero, and TEN-VAD.

Let me show you what it looks like and why we're excited about it.

---

## QUICK RECAP [~0:20]

If you're new here — wavekat-vad is a Rust library for voice activity detection. It gives you a simple, unified interface across multiple VAD engines. You feed in audio, you get back a speech probability. We also have vad-lab, a browser-based tool for comparing backends side by side in real time.

---

## WHAT IS FIREREDVAD [~0:45]

FireRedVAD comes from Xiaohongshu — the company behind Little Red Book. It was released in March 2026 and uses a neural network architecture called DFSMN.

What makes it stand out? The numbers.

On FLEURS-VAD-102, a benchmark that tests across 102 languages:

- F1 score: 97.6 — that's the best we've seen. Silero is at 96.0, TEN-VAD at 95.2.
- False alarm rate: just 2.7% — Silero is at 9.4%, TEN-VAD at 15.5%.
- AUC-ROC: 99.6.

It's also Apache-2.0 licensed with no usage restrictions. TEN-VAD, for comparison, has a non-compete clause. So FireRedVAD is friendlier for production use.

The model is tiny — about 2.2 MB — and it processes audio in 10ms frames. That's finer resolution than Silero's 32ms or TEN-VAD's 16ms, which means more precise speech boundary detection.

---

## DEMO: VAD-LAB COMPARISON [~1:45]

*[Screen: vad-lab running in browser]*

Let me show this in vad-lab. I've got all four backends running on the same audio file. You can see the waveform at the top, and below it, each backend's speech probability over time.

*[Point to results]*

Notice how FireRedVAD's output is smoother and more decisive — it transitions quickly between speech and silence, with fewer hesitations in the middle. WebRTC, being rule-based, gives you binary yes/no with a lot of false negatives. Silero and TEN-VAD are good, but you can see FireRedVAD catches speech segments more consistently with fewer false alarms.

The threshold slider works the same as the other neural backends — 0.5 by default, drag it lower for more sensitivity.

We also added per-stage timing breakdowns in this update, so you can see exactly how much time each processing step takes.

---

## USING IT IN YOUR CODE [~3:00]

Adding FireRedVAD to your project is the same as any other backend. Add the feature flag:

```toml
wavekat-vad = { version = "0.1", features = ["firered"] }
```

We have a ready-to-run example in the repo. Let me run it.

*[Screen: terminal]*

```sh
cargo run --example firered_file --features firered -- testdata/speech/sample.wav
```

*[Show output scrolling — timestamps with probabilities and # bars]*

That's the `firered_file` example. It opens a WAV file, resamples to 16 kHz if needed, and runs FireRedVAD frame by frame — printing the speech probability at each 10ms step. You can see the probability jump up during speech segments and drop back to near zero during silence.

The example is about 70 lines of code. Here's the core of it:

```rust
use wavekat_vad::backends::firered::FireRedVad;
use wavekat_vad::VoiceActivityDetector;

let mut vad = FireRedVad::new().unwrap();
let caps = vad.capabilities();

for frame in samples.chunks_exact(caps.frame_size) {
    let prob = vad.process(frame, 16000).unwrap();
    // ...
}
```

Create the VAD, get the frame size from capabilities, chunk your audio, and call `process`. That's it.

We also have a multi-backend example — `detect_speech` — where you can switch between all four backends with a `--backend` flag:

```sh
cargo run --example detect_speech --features firered -- --backend firered audio.wav
```

The ONNX model downloads automatically at build time and gets embedded in your binary. No external files needed at runtime. If you're building in CI or offline, you can point to a local model file with an environment variable.

One thing to note: FireRedVAD only supports 16 kHz audio. If you're working with 8 kHz telephone audio, you'd need to resample. Silero handles 8k natively if that's your use case.

---

## WHAT MADE THIS INTERESTING [~3:45]

Most VAD models take raw audio samples as input. FireRedVAD is different — it expects preprocessed audio features. Specifically, 80-dimensional log Mel filterbank features with CMVN normalization. This is a Kaldi-style preprocessing pipeline.

The upstream Python implementation uses C++ libraries for this. We reimplemented the entire pipeline in pure Rust — no C++ dependencies, no Python, everything compiles with `cargo build`.

The tricky part was matching every numerical detail exactly. The model is trained on features from a specific pipeline, so if your preprocessing diverges even slightly, you get bad results. We built a validation suite that compares our Rust output against the Python pipeline at every stage — and the final probability difference is 0.000012. Essentially identical.

That was the bulk of the work for this feature. Around 700 lines of Rust for the preprocessing, and a thorough validation setup to prove it works.

---

## TRADEOFFS [~4:45]

Quick summary of how FireRedVAD compares to the other backends:

- **Accuracy**: Best overall. Highest F1, lowest false alarm rate.
- **Frame size**: 10ms — the finest resolution we have.
- **Sample rate**: 16 kHz only. Less flexible than Silero (8k/16k) or WebRTC (8k–48k).
- **Startup**: First two frames return zero while the internal buffer fills up. 30ms startup latency, then it's real-time.
- **License**: Apache-2.0, no restrictions.

If you want the best accuracy and you're working with 16 kHz audio, FireRedVAD is the one to use.

---

## WRAP UP [~5:15]

That's FireRedVAD in wavekat-vad. Fourth backend, best accuracy, pure Rust all the way down.

To try it out: add the `firered` feature flag, or clone the repo and fire up vad-lab to compare all four backends on your own audio.

Links in the description. Thanks for watching, see you next time.

---

## VIDEO METADATA

**Title:** New VAD Backend — FireRedVAD in wavekat-vad

**Description:**
We added FireRedVAD as a new backend to wavekat-vad, our open-source Rust voice activity detection library. FireRedVAD achieves 99.6 AUC-ROC and 97.6 F1 on FLEURS-VAD-102 — the best accuracy of any backend in the library.

In this video:
- What FireRedVAD is and why we added it
- Side-by-side comparison in vad-lab
- How to use it in your code
- Accuracy vs flexibility tradeoffs across all four backends

GitHub: [link]

**Tags:** rust, voice-activity-detection, vad, firered, audio-processing, machine-learning, open-source
