.PHONY: dev dev-frontend dev-backend check test fmt lint

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
