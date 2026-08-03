#!/usr/bin/env bash
# Assert that `--no-default-features --features earshot` resolves to a
# dependency graph with no webrtc-vad, no ONNX runtime, and no build-time
# model downloader.
#
# webrtc-vad bundles libfvad, whose unmangled WebRtcSpl_* symbols collide with
# LiveKit's webrtc-sys and make strict linkers (e.g. Wild) reject the host
# binary outright. The pure-Rust earshot backend exists so such hosts can get a
# VAD without that collision; if enabling `earshot` still pulls webrtc-vad,
# the backend does not do its job.
set -euo pipefail

# realfft/rustfft are intentionally absent from this list: they arrive via
# rubato, a mandatory dependency on every feature set, and are pure Rust.
FORBIDDEN=(webrtc-vad ort ort-sys ureq ndarray nnnoiseless)

# Only normal/build edges matter -- dev-dependencies never reach a consumer.
TREE=$(cargo tree --edges normal,build -p wavekat-vad --no-default-features --features earshot)

echo "$TREE"
echo

if ! grep -qE '(^|[^-a-zA-Z0-9_])earshot v' <<<"$TREE"; then
	echo "FAIL: the 'earshot' feature did not pull in the earshot crate" >&2
	exit 1
fi

status=0
for crate in "${FORBIDDEN[@]}"; do
	if grep -qE "(^|[^-a-zA-Z0-9_])${crate} v" <<<"$TREE"; then
		echo "FAIL: 'earshot'-only build depends on '${crate}'" >&2
		status=1
	fi
done

if [ "$status" -eq 0 ]; then
	echo "OK: earshot-only build is free of ${FORBIDDEN[*]}"
fi
exit "$status"
