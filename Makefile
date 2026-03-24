.PHONY: help setup setup-backend setup-frontend dev dev-frontend dev-backend check test fmt lint doc ci bench accuracy accuracy-update-baseline

help:
	@echo "Available targets:"
	@echo "  setup           Install all dependencies (run once after clone)"
	@echo "  setup-backend   Install cargo-watch"
	@echo "  setup-frontend  Install npm dependencies"
	@echo "  dev-backend     Run vad-lab backend with auto-rebuild"
	@echo "  dev-frontend    Run vad-lab frontend dev server"
	@echo "  dev             Instructions for running both"
	@echo "  check           Check workspace compiles"
	@echo "  test            Run all tests"
	@echo "  fmt             Format code"
	@echo "  lint            Run clippy with warnings as errors"
	@echo "  doc             Build and open docs in browser"
	@echo "  ci              Run all CI checks locally (fmt, clippy, test, doc, features)"
	@echo "  bench           Run criterion benchmarks (inference timing)"
	@echo "  accuracy        Run accuracy test against TEN-VAD testset (downloads ~60 files)"
	@echo "  accuracy-update-baseline  Update best-score baselines after improvements"

# Install all dependencies
setup: setup-backend setup-frontend

# Install cargo-watch for auto-rebuild
setup-backend:
	cargo install cargo-watch

# Install frontend npm dependencies
setup-frontend:
	cd tools/vad-lab/frontend && . "$$NVM_DIR/nvm.sh" && nvm use && npm install

# Run vad-lab backend with auto-rebuild on file changes
dev-backend:
	cargo watch -x 'run -p vad-lab'

# Run vad-lab frontend dev server (uses .nvmrc for Node version)
dev-frontend:
	cd tools/vad-lab/frontend && . "$$NVM_DIR/nvm.sh" && nvm use && npm run dev

# Run both frontend and backend (requires two terminals — use dev-backend + dev-frontend)
dev:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals"

# Check workspace compiles
check:
	cargo check --workspace

# Run all tests
test:
	cargo test --workspace

# Format code
fmt:
	cargo fmt --all

# Lint
lint:
	cargo clippy --workspace -- -D warnings

# Build and open docs in browser
doc:
	cargo doc --no-deps -p wavekat-vad --all-features --open

# Run all CI checks locally (mirrors .github/workflows/ci.yml)
ci:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	cargo test --workspace
	cargo doc --no-deps -p wavekat-vad --all-features
	cargo test -p wavekat-vad --no-default-features --features ""
	cargo test -p wavekat-vad --no-default-features --features "webrtc"
	cargo test -p wavekat-vad --no-default-features --features "silero"
	cargo test -p wavekat-vad --no-default-features --features "ten-vad"
	cargo test -p wavekat-vad --no-default-features --features "firered"
	cargo test -p wavekat-vad --no-default-features --features "serde"
	cargo test -p wavekat-vad --no-default-features --features "webrtc,silero,ten-vad,firered,serde"

# Run criterion benchmarks for per-frame inference timing
bench:
	cargo bench -p wavekat-vad --no-default-features --features "webrtc,silero,ten-vad,firered"

# Run accuracy test against the TEN-VAD testset (30 labeled audio files)
accuracy:
	cargo test --release -p wavekat-vad --no-default-features --features "webrtc,silero,ten-vad,firered" \
		-- --ignored accuracy_report --nocapture

# Update accuracy-baseline.json with current best scores (only raises, never lowers)
accuracy-update-baseline:
	cargo test --release -p wavekat-vad --no-default-features --features "webrtc,silero,ten-vad,firered" \
		-- --ignored accuracy_update_baseline --nocapture
