.PHONY: run down test lint format check logs shell

# Build and start — logs stream to terminal, Ctrl+C stops gracefully
run:
	docker compose up --build

# Stop and remove containers
down:
	docker compose down

# Run lint, format check, and tests
check: lint
	.venv/bin/ruff format --check .
	.venv/bin/pytest

# Run the test suite
test:
	.venv/bin/pytest

# Check for linting issues
lint:
	.venv/bin/ruff check .

# Auto-fix formatting and import order
format:
	.venv/bin/ruff format .

# Tail logs from the aug container
logs:
	docker compose logs -f aug

# Open a shell inside the running aug container
shell:
	docker compose exec aug /bin/bash
